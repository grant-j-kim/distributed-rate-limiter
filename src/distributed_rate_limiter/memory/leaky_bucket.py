"""Leaky bucket, in-memory reference implementation."""

from __future__ import annotations

from distributed_rate_limiter.base import Clock, Decision, RateLimiter, default_clock


class InMemoryLeakyBucket(RateLimiter):
    """A queue draining at a constant rate: shapes traffic instead of dropping it.

    The other four algorithms are meters -- they answer allow or deny and the
    caller proceeds immediately either way. This one is a shaper. An admitted
    request carries a `delay`: how long it waits for the requests already
    queued ahead of it, which drain at exactly `leak_rate` per second. Output
    therefore leaves at a constant rate no matter how bursty the input, which
    is the property none of the others provide.

    Callers must honour ``Decision.delay``. Ignoring it turns this back into a
    meter: requests still get through, just not at the shaped rate.

    Shaping happens strictly within capacity. When the queue is full the
    request is rejected outright rather than queued, because admitting it
    would mean unbounded memory and unbounded latency -- a request "accepted"
    and then left sitting for minutes is worse than an honest 429.

    `level` is a float for the same reason the token bucket's count is: drain
    is applied incrementally, so binary float error accumulates and a strict
    comparison would reject a client that waited exactly as instructed.
    """

    _EPSILON = 1e-9

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
        *,
        clock: Clock = default_clock,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if leak_rate <= 0:
            raise ValueError("leak_rate must be > 0")
        self.capacity = capacity
        self.leak_rate = leak_rate
        # Satisfy the RateLimiter interface: `window` is the time for a full
        # queue to drain completely.
        self.limit = capacity
        self.window = capacity / leak_rate
        self._clock = clock
        # key -> (queue level, time that level was computed)
        self._buckets: dict[str, tuple[float, float]] = {}

    @classmethod
    def from_limit_window(
        cls,
        limit: int,
        window: float,
        *,
        clock: Clock = default_clock,
    ) -> "InMemoryLeakyBucket":
        """Build a bucket equivalent to `limit` requests per `window` seconds."""
        if window <= 0:
            raise ValueError("window must be > 0")
        return cls(capacity=limit, leak_rate=limit / window, clock=clock)

    async def check(self, key: str) -> Decision:
        now = self._clock()

        # No `await` between read and write, so this cannot be interleaved.
        level, last = self._buckets.get(key, (0.0, now))
        elapsed = max(0.0, now - last)
        level = max(0.0, level - elapsed * self.leak_rate)

        if level + 1.0 - self._EPSILON > self.capacity:
            # Queue full. Wait for enough drainage to leave room for one.
            wait = (level + 1.0 - self.capacity) / self.leak_rate
            self._buckets[key] = (level, now)
            return Decision(
                allowed=False,
                limit=self.capacity,
                remaining=0,
                reset_after=wait,
                retry_after=wait,
            )

        # This request joins the queue behind `level` others; it reaches the
        # front once they have drained.
        delay = level / self.leak_rate
        level += 1.0
        self._buckets[key] = (level, now)

        return Decision(
            allowed=True,
            limit=self.capacity,
            remaining=max(0, int(self.capacity - level)),
            reset_after=level / self.leak_rate,
            delay=delay,
        )

    async def reset(self, key: str) -> None:
        self._buckets.pop(key, None)
