"""Leaky bucket backed by Redis, drained inside a Lua script."""

from __future__ import annotations

from typing import Any

from distributed_rate_limiter.base import Clock, Decision, RateLimiter

# KEYS[1] = hash holding {level, timestamp}
# ARGV[1] = capacity
# ARGV[2] = leak rate, requests per second
# ARGV[3] = caller-supplied time, or "" to use the server's clock
# ARGV[4] = max queueing delay in seconds, or -1 for no ceiling
#
# Same shape as the token bucket and the same reason for a script: drain is a
# read-modify-write over two values with a branch in the middle.
#
# The queue timeout is decided in here rather than by the caller. Rejecting
# after the level had already been incremented would charge a client for
# queue space it never used, quietly serving below the configured rate.
CHECK_SCRIPT = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local max_delay = tonumber(ARGV[4])

local now
if ARGV[3] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)
else
  now = tonumber(ARGV[3])
end

local key = KEYS[1]
local state = redis.call('HMGET', key, 'level', 'ts')
local level = tonumber(state[1])
local last = tonumber(state[2])

if level == nil or last == nil then
  level = 0.0
  last = now
end

local elapsed = now - last
if elapsed < 0 then
  elapsed = 0
end

level = level - (elapsed * rate)
if level < 0 then
  level = 0
end

local epsilon = 1e-9

-- Queue full: shaping stops at capacity, past which an honest rejection
-- beats an unbounded queue and unbounded latency.
if level + 1.0 - epsilon > capacity then
  local wait = (level + 1.0 - capacity) / rate
  redis.call('HSET', key, 'level', level, 'ts', now)
  redis.call('PEXPIRE', key, math.ceil((((capacity / rate) * 2) + 1) * 1000))
  return {0, 0, tostring(wait), tostring(wait), '0'}
end

-- This request queues behind `level` others and leaves once they have drained.
local delay = level / rate

if max_delay >= 0 and delay > max_delay + epsilon then
  local wait = delay - max_delay
  redis.call('HSET', key, 'level', level, 'ts', now)
  redis.call('PEXPIRE', key, math.ceil((((capacity / rate) * 2) + 1) * 1000))
  return {0, 0, tostring(wait), tostring(wait), '0'}
end

level = level + 1.0
redis.call('HSET', key, 'level', level, 'ts', now)

-- As with the token bucket, the key must outlive a full drain. Expiring
-- sooner would hand a throttled client an empty queue for free.
redis.call('PEXPIRE', key, math.ceil((((capacity / rate) * 2) + 1) * 1000))

local remaining = math.floor(capacity - level)
if remaining < 0 then
  remaining = 0
end

return {1, remaining, tostring(level / rate), '0', tostring(delay)}
"""


class RedisLeakyBucket(RateLimiter):
    """Traffic shaper whose queue state is shared across instances.

    Behaviourally identical to InMemoryLeakyBucket -- it passes the same
    shared scenario suite -- with `{level, timestamp}` in a Redis hash.

    Callers must honour ``Decision.delay``; ignoring it turns the shaper back
    into a meter. `max_delay` bounds how long a request may be held before it
    is rejected instead, which matters more here than in memory: a delay
    longer than the client's timeout is a connection held for nothing.
    """

    def __init__(
        self,
        client: Any,
        capacity: int,
        leak_rate: float,
        *,
        max_delay: float | None = None,
        prefix: str = "drl:lb",
        clock: Clock | None = None,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if leak_rate <= 0:
            raise ValueError("leak_rate must be > 0")
        if max_delay is not None and max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        self.client = client
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.max_delay = max_delay
        self.limit = capacity
        self.window = capacity / leak_rate
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
        max_delay: float | None = None,
        prefix: str = "drl:lb",
        clock: Clock | None = None,
    ) -> "RedisLeakyBucket":
        """Build a bucket equivalent to `limit` requests per `window` seconds."""
        if window <= 0:
            raise ValueError("window must be > 0")
        return cls(
            client,
            capacity=limit,
            leak_rate=limit / window,
            max_delay=max_delay,
            prefix=prefix,
            clock=clock,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def check(self, key: str) -> Decision:
        now_arg = "" if self._clock is None else repr(self._clock())
        max_delay = -1.0 if self.max_delay is None else self.max_delay

        allowed, remaining, reset_after, retry_after, delay = await self._script(
            keys=[self._key(key)],
            args=[self.capacity, self.leak_rate, now_arg, max_delay],
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
            delay=float(delay),
        )

    async def reset(self, key: str) -> None:
        await self.client.delete(self._key(key))
