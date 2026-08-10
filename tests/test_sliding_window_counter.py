"""Behaviour specific to the sliding window counter.

The shared scenarios live in test_common_scenarios.py. What is tested here is
the approximation itself: where it is stricter than exact, where it is more
permissive, and how far it diverges from the log over realistic traffic.
"""

from __future__ import annotations

import random

import pytest

from distributed_rate_limiter.base import RateLimiter
from distributed_rate_limiter.memory.sliding_window_counter import (
    InMemorySlidingWindowCounter,
)
from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from tests.conftest import FakeClock


async def run_traffic(limiter: RateLimiter, clock: FakeClock, timestamps: list[float]) -> int:
    """Replay timestamps through a limiter, returning how many were allowed."""
    allowed = 0
    for ts in timestamps:
        clock.set(ts)
        if (await limiter.check("client-a")).allowed:
            allowed += 1
    return allowed


async def count_allowed(limiter: RateLimiter, attempts: int, key: str = "client-a") -> int:
    allowed = 0
    for _ in range(attempts):
        if (await limiter.check(key)).allowed:
            allowed += 1
    return allowed


async def test_no_cliff_at_the_window_boundary(clock: FakeClock):
    """Unlike the fixed window, quota does not snap back at the boundary.

    One request does slip through immediately after the boundary: with
    previous == limit, the estimate is limit * overlap, which is strictly
    below the limit for any elapsed time above zero. That single request is
    the algorithm's known over-admission -- and it is still a long way from
    the fixed window, which hands back all 5 slots here.
    """
    window = 60.0
    limiter = InMemorySlidingWindowCounter(limit=5, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 1.0)
    assert await count_allowed(limiter, 5) == 5

    clock.set(boundary + 0.001)
    assert await count_allowed(limiter, 5) == 1


async def test_quota_returns_gradually_across_the_window(clock: FakeClock):
    """The previous window's weight decays linearly, freeing quota smoothly."""
    window = 60.0
    limiter = InMemorySlidingWindowCounter(limit=10, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 0.001)
    for _ in range(10):
        assert (await limiter.check("client-a")).allowed

    # 50% into the next window, half the old count still counts: estimate 5,
    # so 5 of the 10 slots are usable.
    clock.set(boundary + window * 0.5)
    assert await count_allowed(limiter, 10) == 5

    # 90% in, only 10% of the old count remains, but the 5 just spent are now
    # in the current window and count in full.
    clock.set(boundary + window * 0.9)
    assert (await limiter.check("client-a")).allowed


async def test_over_admits_when_previous_window_burst_at_the_end(clock: FakeClock):
    """The approximation's real failure mode, demonstrated not described.

    The estimate assumes the previous window's requests were spread evenly.
    When they all landed in its final second, the weighted estimate falls just
    short of the limit and one extra request slips through -- so a true
    trailing 60s window ends up holding limit + 1. The exact log rejects it.
    """
    window = 60.0
    counter = InMemorySlidingWindowCounter(limit=5, window=window, clock=clock)
    log = InMemorySlidingWindowLog(limit=5, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window

    # Five requests packed into the last second of the previous window.
    clock.set(boundary - 1.0)
    for _ in range(5):
        assert (await counter.check("client-a")).allowed
        assert (await log.check("client-a")).allowed

    # One second into the new window: 59/60 of the old count still weighs in,
    # giving an estimate of 4.917 -- just under the limit of 5.
    clock.set(boundary + 1.0)
    counter_decision = await counter.check("client-a")
    log_decision = await log.check("client-a")

    assert counter_decision.allowed, "counter under-counts the end-loaded burst"
    assert not log_decision.allowed, "the exact algorithm rejects it"

    # 6 requests inside a 2-second span against a limit of 5 per 60s.


async def test_is_stricter_when_previous_window_burst_at_the_start(clock: FakeClock):
    """The error runs both ways; this direction is the safe one.

    A burst at the *start* of the previous window has already aged out of a
    true trailing window, but the estimate still charges ~full weight for it,
    so the counter rejects requests the exact log allows.
    """
    window = 60.0
    counter = InMemorySlidingWindowCounter(limit=5, window=window, clock=clock)
    log = InMemorySlidingWindowLog(limit=5, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window

    # Five requests at the very start of the previous window.
    clock.set(boundary - window)
    for _ in range(5):
        assert (await counter.check("client-a")).allowed
        assert (await log.check("client-a")).allowed

    # One second into the new window, those requests are 61s old and have left
    # the log's trailing window entirely -- but still carry 59/60 weight here.
    # The counter lets exactly one through (estimate 4.917) and then clamps;
    # the log has forgotten them and grants the full quota.
    clock.set(boundary + 1.0)
    assert await count_allowed(counter, 5) == 1, "counter over-counts the aged-out burst"
    assert await count_allowed(log, 5) == 5, "exact algorithm has already forgotten them"


async def test_divergence_from_exact_algorithm_stays_small(clock: FakeClock):
    """Measure, don't assume, how far the approximation drifts.

    Poisson traffic at roughly 2x the limit over 100 windows (~20k requests),
    replayed through both algorithms.

    Measured across four seeds, the counter allowed 1.01%-1.15% more requests
    than the exact log -- and it was more permissive in every run, never
    stricter. With this seed: counter 10009, log 9909 (+100, 1.01%). The 2%
    bound is set from those runs, not guessed. The direction assertion is the
    more interesting half: it says the approximation's cost is paid in extra
    admissions, which is what to weigh when choosing between the two.
    """
    window = 60.0
    limit = 100
    rng = random.Random(1_618_033)

    start = clock()
    timestamps: list[float] = []
    t = start
    end = start + window * 100
    while t < end:
        t += rng.expovariate(200.0 / window)  # ~200 req/window vs a limit of 100
        timestamps.append(t)

    counter_allowed = await run_traffic(
        InMemorySlidingWindowCounter(limit=limit, window=window, clock=clock),
        clock,
        timestamps,
    )
    clock.set(start)
    log_allowed = await run_traffic(
        InMemorySlidingWindowLog(limit=limit, window=window, clock=clock),
        clock,
        timestamps,
    )

    divergence = abs(counter_allowed - log_allowed) / log_allowed
    detail = (
        f"counter allowed {counter_allowed}, log allowed {log_allowed} "
        f"({divergence:.2%} divergence over {len(timestamps)} requests)"
    )
    assert divergence < 0.02, detail
    assert counter_allowed >= log_allowed, f"expected the approximation to err permissive: {detail}"


async def test_state_is_constant_size_per_key(clock: FakeClock):
    """Two integers per key regardless of traffic -- the point of the algorithm."""
    limiter = InMemorySlidingWindowCounter(limit=50, window=60.0, clock=clock)

    for _ in range(5000):
        await limiter.check("client-a")
        clock.advance(0.01)

    index, current, previous = limiter._counters["client-a"]
    assert isinstance(current, int) and isinstance(previous, int)
    assert len(limiter._counters) == 1


async def test_counts_are_dropped_after_two_idle_windows(clock: FakeClock):
    limiter = InMemorySlidingWindowCounter(limit=5, window=60.0, clock=clock)

    for _ in range(5):
        assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    clock.advance(120.0)
    for _ in range(5):
        assert (await limiter.check("client-a")).allowed


async def test_retry_after_clears_the_estimate(clock: FakeClock):
    """Waiting exactly retry_after must actually get the client through.

    A retry_after that is too short sends the client straight into another
    429, which is worse than useless under load.
    """
    window = 60.0
    limiter = InMemorySlidingWindowCounter(limit=10, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 0.001)
    for _ in range(10):
        assert (await limiter.check("client-a")).allowed

    # Consume the one request the estimate still leaves room for, then hit
    # the wall.
    clock.set(boundary + 1.0)
    assert (await limiter.check("client-a")).allowed
    denied = await limiter.check("client-a")
    assert not denied.allowed
    assert denied.retry_after is not None and denied.retry_after > 0

    clock.advance(denied.retry_after)
    assert (await limiter.check("client-a")).allowed


@pytest.mark.parametrize(
    "limit,window",
    [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)],
)
async def test_invalid_configuration_is_rejected(limit, window):
    with pytest.raises(ValueError):
        InMemorySlidingWindowCounter(limit=limit, window=window)
