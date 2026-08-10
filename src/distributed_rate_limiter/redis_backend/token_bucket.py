"""Token bucket backed by Redis, refilled inside a Lua script."""

from __future__ import annotations

from typing import Any

from distributed_rate_limiter.base import Clock, Decision, RateLimiter

# KEYS[1] = hash holding {tokens, timestamp}
# ARGV[1] = capacity
# ARGV[2] = refill rate, tokens per second
# ARGV[3] = caller-supplied time, or "" to use the server's clock
#
# This is the algorithm INCR cannot express. The refill reads *two* values
# (token count and the time they were computed), does arithmetic against the
# current time, branches on whether a whole token exists, and writes both back.
# MULTI cannot do it either: it has no way to branch on a value it has not
# seen yet, and it sees nothing until the whole batch has run. A script is the
# only construct that reads, computes, decides, and writes without a gap.
#
# Both hash fields are written together. Writing them separately would let a
# crash in between leave a token count paired with a stale timestamp, and the
# next request would re-grant the elapsed refill -- minting free tokens.
CHECK_SCRIPT = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])

local now
if ARGV[3] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)
else
  now = tonumber(ARGV[3])
end

local key = KEYS[1]
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local last = tonumber(state[2])

-- An unseen client starts with a full bucket, so first-time traffic gets its
-- burst rather than waiting for the bucket to fill.
if tokens == nil or last == nil then
  tokens = capacity
  last = now
end

local elapsed = now - last
if elapsed < 0 then
  elapsed = 0
end

tokens = math.min(capacity, tokens + (elapsed * rate))

-- Refill is applied incrementally, so binary float error accumulates: ten
-- successive 0.1s refills sum to 0.9999999999999999, not 1.0. Without this
-- tolerance a client that waited exactly the retry_after it was given is
-- rejected for being ~1e-9 tokens short. Lua numbers are IEEE doubles, the
-- same as Python floats, so the drift is identical to the in-memory version.
local epsilon = 1e-9
local allowed = 0
local retry_after = 0.0

if tokens + epsilon >= 1.0 then
  allowed = 1
  tokens = tokens - 1.0
else
  retry_after = (1.0 - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)

-- The key must outlive a refill from empty to full. Expiring sooner would
-- delete the state of a throttled client, and the next request would find no
-- key and start again with a full bucket -- a free reset for anyone willing
-- to pause. The window algorithms can expire at the end of their window;
-- this one has no window, so the TTL is the time to refill completely, plus
-- a margin.
local ttl = ((capacity / rate) * 2) + 1
redis.call('PEXPIRE', key, math.ceil(ttl * 1000))

local reset_after = (capacity - tokens) / rate
local remaining = math.floor(tokens)

return {allowed, remaining, tostring(reset_after), tostring(retry_after)}
"""


class RedisTokenBucket(RateLimiter):
    """Token bucket whose state is shared across every application instance.

    Behaviourally identical to InMemoryTokenBucket -- it passes the same
    shared scenario suite -- with `{tokens, timestamp}` held in a Redis hash
    and the refill computed inside a Lua script so the read-modify-write
    cannot be interleaved.

    As in memory, burst size and sustained rate stay independent: `capacity`
    is how large a burst is tolerated, `refill_rate` what is sustained.
    """

    def __init__(
        self,
        client: Any,
        capacity: int,
        refill_rate: float,
        *,
        prefix: str = "drl:tb",
        clock: Clock | None = None,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")
        self.client = client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.limit = capacity
        self.window = capacity / refill_rate
        self.prefix = prefix
        self._clock = clock
        self._script = client.register_script(CHECK_SCRIPT)

    @classmethod
    def from_limit_window(
        cls,
        client: Any,
        limit: int,
        window: float,
        *,
        prefix: str = "drl:tb",
        clock: Clock | None = None,
    ) -> "RedisTokenBucket":
        """Build a bucket equivalent to `limit` requests per `window` seconds."""
        if window <= 0:
            raise ValueError("window must be > 0")
        return cls(
            client,
            capacity=limit,
            refill_rate=limit / window,
            prefix=prefix,
            clock=clock,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def check(self, key: str) -> Decision:
        now_arg = "" if self._clock is None else repr(self._clock())
        allowed, remaining, reset_after, retry_after = await self._script(
            keys=[self._key(key)],
            args=[self.capacity, self.refill_rate, now_arg],
        )

        if not int(allowed):
            wait = float(retry_after)
            return Decision(
                allowed=False,
                limit=self.capacity,
                remaining=0,
                reset_after=wait,
                retry_after=wait,
            )
        return Decision(
            allowed=True,
            limit=self.capacity,
            remaining=int(remaining),
            reset_after=float(reset_after),
        )

    async def reset(self, key: str) -> None:
        await self.client.delete(self._key(key))
