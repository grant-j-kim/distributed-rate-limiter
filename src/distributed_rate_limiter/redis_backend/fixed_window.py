"""Fixed window counter backed by Redis."""

from __future__ import annotations

from typing import Any

from distributed_rate_limiter.base import Clock, Decision, RateLimiter

# KEYS[1] = base key for this client
# ARGV[1] = limit
# ARGV[2] = window length in seconds
# ARGV[3] = caller-supplied time, or "" to use the server's clock
#
# One script rather than INCR followed by EXPIRE, because those two commands
# are not atomic as a pair:
#
#   * calling EXPIRE on every request refreshes the TTL continuously, so under
#     sustained load the window never ends and the client is locked out for
#     good (the same bug the in-memory version is tested against);
#   * calling EXPIRE only when the counter comes back as 1 fixes that, but if
#     the client dies in between, the key is left with no TTL and leaks for
#     ever -- once per client, so the leak grows with your user base.
#
# Inside a script both happen or neither does, in a single round trip.
CHECK_SCRIPT = """
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local now
if ARGV[3] == '' then
  -- Read the clock from Redis so every application instance computes the
  -- same window boundary. Using each instance's own wall clock would let
  -- clock skew put two servers in different windows for the same request.
  local t = redis.call('TIME')
  now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)
else
  now = tonumber(ARGV[3])
end

local index = math.floor(now / window)
local key = KEYS[1] .. ':' .. index
local count = redis.call('INCR', key)

if count == 1 then
  -- Set the TTL only when the key is created. The window is a property of
  -- the clock, not of the last request, so it must not be extended.
  redis.call('PEXPIRE', key, math.ceil(window * 1000))
end

local reset_after = ((index + 1) * window) - now

if count > limit then
  return {0, 0, tostring(reset_after)}
end

return {1, limit - count, tostring(reset_after)}
"""


class RedisFixedWindow(RateLimiter):
    """Fixed window counter whose state lives in Redis.

    Behaviourally identical to InMemoryFixedWindow -- it passes the same
    shared scenario suite -- but the counter is shared, so N application
    instances enforce one limit between them rather than N limits.

    Windows are epoch-aligned and the counter key carries the window index,
    so each window is a distinct key that expires on its own. There is no
    sweeping and no stale state to clean up.
    """

    def __init__(
        self,
        client: Any,
        limit: int,
        window: float,
        *,
        prefix: str = "drl:fw",
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
        # None means "let Redis decide the time", which is what production
        # should do. Tests inject a clock to step over window boundaries
        # deterministically instead of sleeping.
        self._clock = clock
        self._script = client.register_script(CHECK_SCRIPT)

    def _base_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def check(self, key: str) -> Decision:
        now_arg = "" if self._clock is None else repr(self._clock())
        allowed, remaining, reset_after = await self._script(
            keys=[self._base_key(key)],
            args=[self.limit, self.window, now_arg],
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
        """Delete every window's counter for `key`.

        Scans rather than computing the current window index, so it clears
        state regardless of which window it was written in. Intended for
        tests and administrative use, not the request path.
        """
        pattern = f"{self._base_key(key)}:*"
        keys = [k async for k in self.client.scan_iter(match=pattern)]
        if keys:
            await self.client.delete(*keys)
