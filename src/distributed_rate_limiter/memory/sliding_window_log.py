"""Sliding window log, in-memory reference implementation."""

from __future__ import annotations

from collections import deque

from distributed_rate_limiter.base import Clock, Decision, RateLimiter, default_clock


class InMemorySlidingWindowLog(RateLimiter):
    """Keeps a timestamp per allowed request and counts those still in window.

    This is the exact algorithm: the window is a true trailing interval
    ``(now - window, now]`` rather than a fixed bucket, so the boundary burst
    that InMemoryFixedWindow permits is impossible here. No span of `window`
    seconds can ever contain more than `limit` allowed requests.

    The cost is memory: state is O(limit) timestamps per key instead of a
    single integer, and every check pays an eviction scan. That is what the
    sliding window *counter* approximates away.

    Only allowed requests are logged. Logging rejected ones too would let a
    client hammering while blocked keep refilling its own window and lock
    itself out indefinitely, and would make per-key memory unbounded under
    exactly the traffic where it matters.
    """

    def __init__(self, limit: int, window: float, *, clock: Clock = default_clock):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window <= 0:
            raise ValueError("window must be > 0")
        self.limit = limit
        self.window = window
        self._clock = clock
        # key -> timestamps of allowed requests, oldest first
        self._logs: dict[str, deque[float]] = {}

    def _evict(self, log: deque[float], now: float) -> None:
        """Drop entries that have aged out of the trailing window.

        The cutoff is half-open: an entry exactly `window` old has expired.
        This matches the fixed window's reset semantics, so both algorithms
        agree that waiting one full window restores the full quota.
        """
        cutoff = now - self.window
        while log and log[0] <= cutoff:
            log.popleft()

    async def check(self, key: str) -> Decision:
        now = self._clock()

        # As in the fixed window: no `await` between reading and mutating the
        # log, so the event loop cannot interleave another check here.
        log = self._logs.get(key)
        if log is None:
            log = self._logs[key] = deque()
        self._evict(log, now)

        if len(log) >= self.limit:
            # The next slot frees when the oldest live entry expires -- more
            # precise than the fixed window's "wait for the whole bucket".
            wait = log[0] + self.window - now
            return Decision(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_after=wait,
                retry_after=wait,
            )

        log.append(now)
        return Decision(
            allowed=True,
            limit=self.limit,
            remaining=self.limit - len(log),
            reset_after=log[0] + self.window - now,
        )

    async def reset(self, key: str) -> None:
        self._logs.pop(key, None)
