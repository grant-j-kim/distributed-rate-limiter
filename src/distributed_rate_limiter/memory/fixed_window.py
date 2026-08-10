"""Fixed window counter, in-memory reference implementation."""

from __future__ import annotations

from distributed_rate_limiter.base import Clock, Decision, RateLimiter, default_clock


class InMemoryFixedWindow(RateLimiter):
    """Counts requests per fixed, epoch-aligned window.

    Windows are aligned to the UNIX epoch (``floor(now / window)``) rather than
    started lazily on a key's first request. Lazy windows would drift apart on
    different server instances for the same client; epoch alignment means every
    instance computes the same boundary from the same timestamp, which is what
    makes the Redis version of this a drop-in replacement.

    The well-known flaw of this algorithm is the boundary burst: a client can
    send `limit` requests just before a boundary and `limit` more just after,
    passing 2x the limit in a span shorter than one window. That is inherent to
    fixed windows, not a bug here, and it is the behaviour the sliding window
    algorithms exist to fix. The test suite asserts it explicitly so that the
    comparison plots in Milestone 4 have a documented baseline.
    """

    def __init__(self, limit: int, window: float, *, clock: Clock = default_clock):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window <= 0:
            raise ValueError("window must be > 0")
        self.limit = limit
        self.window = window
        self._clock = clock
        # key -> (window index, count within that window)
        self._counters: dict[str, tuple[int, int]] = {}

    async def check(self, key: str) -> Decision:
        now = self._clock()
        index = int(now // self.window)
        reset_after = (index + 1) * self.window - now

        # No `await` between reading and writing _counters, so the event loop
        # cannot preempt this block. That is what makes the in-memory version
        # atomic; the Redis version has no such guarantee and must earn its
        # atomicity from the server (see Milestone 3).
        stored_index, count = self._counters.get(key, (index, 0))
        if stored_index != index:
            # Stale window: this is the first request of a new one.
            count = 0

        if count >= self.limit:
            self._counters[key] = (index, count)
            return Decision(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_after=reset_after,
                retry_after=reset_after,
            )

        count += 1
        self._counters[key] = (index, count)
        return Decision(
            allowed=True,
            limit=self.limit,
            remaining=self.limit - count,
            reset_after=reset_after,
        )

    async def reset(self, key: str) -> None:
        self._counters.pop(key, None)
