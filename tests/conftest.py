from __future__ import annotations

from typing import Callable, Protocol

import pytest

from distributed_rate_limiter.base import Clock, RateLimiter
from distributed_rate_limiter.memory.fixed_window import InMemoryFixedWindow
from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog


class FakeClock:
    """A hand-cranked clock, so boundary tests are deterministic.

    Testing window boundaries with real time.sleep() makes the suite slow and
    flaky under CI load. Every algorithm takes an injected clock precisely so
    these tests can step over a boundary exactly.
    """

    def __init__(self, now: float = 1_000_000.0):
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def set(self, seconds: float) -> None:
        self._now = seconds


class LimiterFactory(Protocol):
    def __call__(self, limit: int, window: float, clock: Clock) -> RateLimiter: ...


# Every (algorithm, backend) pair registers here and inherits the shared
# scenario suite in test_common_scenarios.py. Redis-backed implementations
# will be appended to this list in Milestone 3, which is how we prove they
# behave identically to the in-memory references.
ALL_LIMITERS: list[tuple[str, LimiterFactory]] = [
    ("fixed_window", InMemoryFixedWindow),
    ("sliding_window_log", InMemorySlidingWindowLog),
]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(params=[f for _, f in ALL_LIMITERS], ids=[n for n, _ in ALL_LIMITERS])
def make_limiter(request, clock: FakeClock) -> Callable[..., RateLimiter]:
    """Builds one limiter of whichever implementation is under test."""
    factory: LimiterFactory = request.param

    def _make(limit: int = 5, window: float = 60.0) -> RateLimiter:
        return factory(limit=limit, window=window, clock=clock)

    return _make
