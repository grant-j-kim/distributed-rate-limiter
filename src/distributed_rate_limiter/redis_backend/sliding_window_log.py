"""Sliding window log backed by Redis sorted sets."""

from __future__ import annotations

import uuid
from typing import Any

from distributed_rate_limiter.base import Clock, Decision, RateLimiter

# KEYS[1] = sorted set holding this client's request timestamps
# ARGV[1] = limit
# ARGV[2] = window length in seconds
# ARGV[3] = caller-supplied time, or "" to use the server's clock
# ARGV[4] = unique member id for this request
#
# Prune, count, and insert have to happen with nothing in between. Split
# across three round trips, any number of clients could each ZCARD the same
# under-limit count before any of them ZADDs, and all of them would be
# admitted -- the same race the fixed window's INCR avoids by being a single
# command. A sorted set has no equivalent single command, so this is a script.
CHECK_SCRIPT = """
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local member = ARGV[4]

local now
if ARGV[3] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)
else
  now = tonumber(ARGV[3])
end

local key = KEYS[1]

-- The window is half-open, (now - window, now]: an entry exactly `window`
-- old has expired. ZREMRANGEBYSCORE's max is inclusive, which matches the
-- in-memory implementation's `<= cutoff` eviction exactly.
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

local count = redis.call('ZCARD', key)

if count >= limit then
  -- The next slot frees when the oldest surviving entry ages out.
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local wait = window
  if oldest[2] then
    wait = tonumber(oldest[2]) + window - now
  end
  return {0, 0, tostring(wait)}
end

-- The member must be unique: ZADD on an existing member updates its score
-- instead of inserting, so two requests sharing a member would collapse into
-- one entry and the limiter would undercount.
redis.call('ZADD', key, now, member)

-- Unlike the fixed window, refreshing this TTL is correct. Here the key's
-- expiry is only garbage collection for an idle client; the window itself is
-- enforced by ZREMRANGEBYSCORE on the scores. In the fixed window the TTL
-- *is* the window, which is why refreshing it there locks clients out.
redis.call('PEXPIRE', key, math.ceil(window * 1000))

local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local reset_after = tonumber(oldest[2]) + window - now

return {1, limit - (count + 1), tostring(reset_after)}
"""


class RedisSlidingWindowLog(RateLimiter):
    """Exact trailing-window enforcement with state shared across instances.

    Behaviourally identical to InMemorySlidingWindowLog -- it passes the same
    shared scenario suite -- with the timestamp log held in a Redis sorted set
    instead of a deque.

    Cost profile carries over too: O(limit) entries per key rather than the
    fixed window's single integer, and every check pays a prune. What it buys
    is exactness -- no boundary burst, ever.
    """

    def __init__(
        self,
        client: Any,
        limit: int,
        window: float,
        *,
        prefix: str = "drl:swl",
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
        allowed, remaining, reset_after = await self._script(
            keys=[self._key(key)],
            args=[self.limit, self.window, now_arg, uuid.uuid4().hex],
        )

        reset_after = float(reset_after)
        if not int(allowed):
            return Decision(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_after=reset_after,
                retry_after=reset_after,
            )
        return Decision(
            allowed=True,
            limit=self.limit,
            remaining=int(remaining),
            reset_after=reset_after,
        )

    async def reset(self, key: str) -> None:
        await self.client.delete(self._key(key))
