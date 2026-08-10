"""Sliding window counter, in-memory reference implementation."""

from __future__ import annotations

import math

from distributed_rate_limiter.base import Clock, Decision, RateLimiter, default_clock


class InMemorySlidingWindowCounter(RateLimiter):
    """Approximates the sliding window log with two counters per key.

    Keeps the current epoch-aligned window's count and the previous window's,
    then estimates the trailing-window total by weighting the previous count
    by how much of it still overlaps:

        estimate = prev * (window - elapsed) / window + current

    State is O(1) per key rather than the log's O(limit) timestamps, and the
    quota slides continuously instead of resetting at a cliff -- which is why
    this is the version usually deployed at scale.

    The approximation assumes the previous window's requests were spread
    uniformly across it. When they were not, the estimate is wrong in a
    direction worth knowing:

    * clustered at the *start* of the previous window -> over-counts, so the
      limiter is stricter than necessary. Safe, just lossy.
    * clustered at the *end* -> under-counts, and the limiter can admit more
      than `limit` requests in a true trailing window.

    The second case is a genuine over-admission, bounded but real, and is
    demonstrated directly in test_sliding_window_counter.py rather than left
    as a footnote. Use the log when the ceiling must be exact; use this when
    O(1) memory per key matters more than the last few percent.
    """

    def __init__(self, limit: int, window: float, *, clock: Clock = default_clock):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window <= 0:
            raise ValueError("window must be > 0")
        self.limit = limit
        self.window = window
        self._clock = clock
        # key -> (window index, count in that window, count in the one before)
        self._counters: dict[str, tuple[int, int, int]] = {}

    def _load(self, key: str, index: int) -> tuple[int, int]:
        """Return (current, previous) counts rolled forward to `index`."""
        stored = self._counters.get(key)
        if stored is None:
            return 0, 0

        stored_index, current, previous = stored
        delta = index - stored_index
        if delta == 0:
            return current, previous
        if delta == 1:
            # Exactly one window has elapsed: what was current is now previous.
            return 0, current
        # Two or more windows of silence: nothing left to carry.
        return 0, 0

    async def check(self, key: str) -> Decision:
        now = self._clock()
        index = int(now // self.window)
        start = index * self.window
        elapsed = now - start
        # Fraction of the previous window still inside the trailing window.
        overlap = (self.window - elapsed) / self.window

        # No `await` between load and store, so this block cannot be
        # interleaved by the event loop.
        current, previous = self._load(key, index)
        estimate = previous * overlap + current

        if estimate >= self.limit:
            wait = self._retry_after(previous, current, start, now)
            return Decision(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_after=wait,
                retry_after=wait,
            )

        current += 1
        self._counters[key] = (index, current, previous)
        return Decision(
            allowed=True,
            limit=self.limit,
            remaining=max(0, self.limit - math.ceil(previous * overlap + current)),
            reset_after=start + self.window - now,
        )

    def _retry_after(self, previous: int, current: int, start: float, now: float) -> float:
        """Seconds until the estimate is expected to fall below the limit.

        Within the current window the only thing that decreases is the
        previous window's contribution, which decays linearly. Solving
        `previous * (window - e) / window + current < limit` for the elapsed
        time `e` gives the instant the estimate clears.

        If the current window alone has already reached the limit, no amount
        of decay helps and the client must wait for the next window. That is
        a lower bound -- the next window inherits this one's count as its
        previous -- so the client may get one more 429 on retry. Reporting
        the true wait would require predicting traffic that has not happened
        yet; under-promising is the conventional trade, and the client's next
        429 carries a fresh estimate.
        """
        boundary = start + self.window - now

        headroom = self.limit - current
        if headroom <= 0 or previous <= 0:
            return boundary

        # e > window * (1 - headroom / previous). Nudge past the crossing:
        # the check denies on `>=`, so landing exactly on the solution earns
        # another 429 -- the precise thing a Retry-After must not do.
        clears_at = start + self.window * (1 - headroom / previous) + self.window * 1e-9
        return max(0.0, min(clears_at - now, boundary))

    async def reset(self, key: str) -> None:
        self._counters.pop(key, None)
