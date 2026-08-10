"""Sliding window counter backed by Redis."""

from __future__ import annotations

from typing import Any

from distributed_rate_limiter.base import Clock, Decision, RateLimiter

# KEYS[1] = hash holding {idx, cur, prev}
# ARGV[1] = limit
# ARGV[2] = window length in seconds
# ARGV[3] = caller-supplied time, or "" to use the server's clock
#
# All three fields live in one hash rather than in two index-suffixed keys.
# That mirrors the in-memory (index, current, previous) tuple, and keeps this
# a single-key script: a two-key version would have to hash-tag its keys to
# stay correct on Redis Cluster, which is a constraint worth not having.
#
# Rolling the window forward is a read-modify-write with a branch, so it
# needs a script for the same reason the buckets do.
CHECK_SCRIPT = """
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local now
if ARGV[3] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)
else
  now = tonumber(ARGV[3])
end

local key = KEYS[1]
local index = math.floor(now / window)
local start = index * window
local elapsed = now - start
local overlap = (window - elapsed) / window

local state = redis.call('HMGET', key, 'idx', 'cur', 'prev')
local idx = tonumber(state[1])
local cur = tonumber(state[2])
local prev = tonumber(state[3])

if idx == nil or cur == nil or prev == nil then
  idx = index
  cur = 0
  prev = 0
end

local delta = index - idx
if delta == 1 then
  -- Exactly one window on: what was current becomes previous.
  prev = cur
  cur = 0
elseif delta ~= 0 then
  -- Two or more windows of silence (or a clock that moved backwards):
  -- nothing left to carry.
  prev = 0
  cur = 0
end

local estimate = (prev * overlap) + cur

-- The key must survive into the *next* window, where this window's count is
-- read as `prev`. One window of TTL would drop that history and hand the
-- client a clean slate at every boundary -- exactly the fixed window's cliff,
-- which is what this algorithm exists to avoid. This is the fourth distinct
-- TTL rule in this package: never refresh (fixed window), always refresh
-- (log), outlive a full refill (buckets), span two windows (here).
local ttl_ms = math.ceil(window * 2000) + 1000

if estimate >= limit then
  local boundary = start + window - now
  local headroom = limit - cur
  local wait

  if headroom <= 0 or prev <= 0 then
    -- The current window alone has reached the limit; no amount of decay in
    -- the previous window's contribution helps.
    wait = boundary
  else
    -- Solve prev * (window - e) / window + cur < limit for the elapsed time
    -- e, nudged past the crossing because the check above denies on >=.
    local clears_at = start + (window * (1 - (headroom / prev))) + (window * 1e-9)
    wait = clears_at - now
    if wait < 0 then wait = 0 end
    if wait > boundary then wait = boundary end
  end

  redis.call('HSET', key, 'idx', index, 'cur', cur, 'prev', prev)
  redis.call('PEXPIRE', key, ttl_ms)
  return {0, 0, tostring(wait), tostring(wait)}
end

cur = cur + 1
redis.call('HSET', key, 'idx', index, 'cur', cur, 'prev', prev)
redis.call('PEXPIRE', key, ttl_ms)

local remaining = limit - math.ceil((prev * overlap) + cur)
if remaining < 0 then
  remaining = 0
end

return {1, remaining, tostring(start + window - now), '0'}
"""


class RedisSlidingWindowCounter(RateLimiter):
    """Approximates the sliding window log with two shared counters.

    Behaviourally identical to InMemorySlidingWindowCounter -- it passes the
    same shared scenario suite -- with `{index, current, previous}` in a Redis
    hash.

    The approximation's characteristics carry over unchanged: O(1) state per
    key, no boundary cliff, and a small permissive bias measured at roughly
    1% against the exact log under Poisson traffic.
    """

    def __init__(
        self,
        client: Any,
        limit: int,
        window: float,
        *,
        prefix: str = "drl:swc",
        clock: Clock | None = None,
    ):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window <= 0:
            raise ValueError("window must be > 0")
        self.client = client
        self.limit = limit
        self.window = window
        self.prefix = prefix
        self._clock = clock
        self._script = client.register_script(CHECK_SCRIPT)

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def check(self, key: str) -> Decision:
        now_arg = "" if self._clock is None else repr(self._clock())
        allowed, remaining, reset_after, retry_after = await self._script(
            keys=[self._key(key)],
            args=[self.limit, self.window, now_arg],
        )

        if not int(allowed):
            wait = float(retry_after)
            return Decision(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_after=wait,
                retry_after=wait,
            )
        return Decision(
            allowed=True,
            limit=self.limit,
            remaining=int(remaining),
            reset_after=float(reset_after),
        )

    async def reset(self, key: str) -> None:
        await self.client.delete(self._key(key))
