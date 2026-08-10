"""Token bucket, in-memory reference implementation."""

from __future__ import annotations

import math

from distributed_rate_limiter.base import Clock, Decision, RateLimiter, default_clock


class InMemoryTokenBucket(RateLimiter):
    """Tokens accrue continuously; each request spends one.

    Unlike the window algorithms, this one has no window and no eviction.
    State is a token count plus the time it was last updated, and refill is
    computed lazily on read as ``elapsed * refill_rate``, capped at capacity.
    An idle bucket therefore costs nothing -- no timer, no background sweep.

    Its distinguishing property is that burst size and sustained rate are
    separate knobs. `capacity` sets how large a burst is tolerated after a
    quiet period; `refill_rate` sets what is sustained indefinitely. The
    window algorithms conflate the two: their burst allowance *is* their rate.

    Tokens are deliberately a float. Truncating to int would mean any refill
    rate below one token per second never accumulates anything and the bucket
    deadlocks -- covered by test_fractional_tokens_accumulate.
    """

    # Refill is applied incrementally, so binary float error accumulates: ten
    # successive 0.1s refills sum to 0.9999999999999999, not 1.0. Without a
    # tolerance, a client that waited exactly the advertised retry_after is
    # denied for being ~2e-10 tokens short. The alternative -- inflating the
    # advertised wait instead -- puts the correction in the wrong place and
    # makes retry_after exceed its own window.
    _EPSILON = 1e-9

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        *,
        clock: Clock = default_clock,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")
        self.capacity = capacity
        self.refill_rate = refill_rate
        # Satisfy the RateLimiter interface: `window` is the time to refill an
        # empty bucket completely.
        self.limit = capacity
        self.window = capacity / refill_rate
        self._clock = clock
        # key -> (tokens available, time those tokens were computed)
        self._buckets: dict[str, tuple[float, float]] = {}

    @classmethod
    def from_limit_window(
        cls,
        limit: int,
        window: float,
        *,
        clock: Clock = default_clock,
    ) -> "InMemoryTokenBucket":
        """Build a bucket equivalent to `limit` requests per `window` seconds.

        Exists so the shared scenario suite, which speaks (limit, window), can
        exercise this algorithm without the algorithm giving up its real
        signature. Collapsing the two knobs like this is exactly the
        expressiveness the token bucket normally buys you.
        """
        if window <= 0:
            raise ValueError("window must be > 0")
        return cls(capacity=limit, refill_rate=limit / window, clock=clock)

    async def check(self, key: str) -> Decision:
        now = self._clock()

        # A new key starts full, so a first-time client gets its burst.
        tokens, last = self._buckets.get(key, (float(self.capacity), now))

        # No `await` between read and write: the event loop cannot interleave
        # another check here. The Redis version cannot rely on that and needs
        # a Lua script, because this refill is a read-modify-write over *two*
        # values (tokens and timestamp) that INCR cannot express atomically.
        elapsed = max(0.0, now - last)
        tokens = min(float(self.capacity), tokens + elapsed * self.refill_rate)

        if tokens + self._EPSILON < 1.0:
            wait = self._retry_after(tokens)
            # Persist the refill even on rejection: dropping it would discard
            # accrued tokens and stall the client indefinitely under load.
            self._buckets[key] = (tokens, now)
            return Decision(
                allowed=False,
                limit=self.capacity,
                remaining=0,
                reset_after=wait,
                retry_after=wait,
            )

        tokens -= 1.0
        self._buckets[key] = (tokens, now)
        return Decision(
            allowed=True,
            limit=self.capacity,
            remaining=int(tokens),
            reset_after=(self.capacity - tokens) / self.refill_rate,
        )

    def _retry_after(self, tokens: float) -> float:
        """Seconds until one whole token is available.

        Reported exactly, with no padding: a client waiting this long lands on
        1.0 tokens and the _EPSILON tolerance in check() absorbs the float
        drift. Padding here instead would push retry_after past the bucket's
        own refill window, which is a lie in the other direction.
        """
        return (1.0 - tokens) / self.refill_rate

    async def reset(self, key: str) -> None:
        self._buckets.pop(key, None)
