"""Behaviour specific to the leaky bucket.

The shared scenarios live in test_common_scenarios.py. What is tested here is
the shaping itself: that a burst is spread out at a constant output rate
rather than rejected, and that shaping stops at capacity.
"""

from __future__ import annotations

import pytest

from distributed_rate_limiter.memory.leaky_bucket import InMemoryLeakyBucket
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from tests.conftest import FakeClock


async def test_burst_is_spread_at_a_constant_output_rate(clock: FakeClock):
    """The defining property: input is bursty, output is not.

    Ten simultaneous requests against a bucket draining at 2/s are all
    admitted, but scheduled 0.5s apart. No metering algorithm can do this --
    they would admit them all instantly or reject the excess.
    """
    limiter = InMemoryLeakyBucket(capacity=10, leak_rate=2.0, clock=clock)

    delays = []
    for _ in range(10):
        decision = await limiter.check("client-a")
        assert decision.allowed
        delays.append(decision.delay)

    assert delays == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

    # Every consecutive pair is exactly one drain interval apart.
    gaps = [b - a for a, b in zip(delays, delays[1:])]
    assert gaps == pytest.approx([0.5] * 9)


async def test_arrival_at_an_empty_bucket_is_not_delayed(clock: FakeClock):
    """Shaping must not add latency to traffic that is already well behaved."""
    limiter = InMemoryLeakyBucket(capacity=5, leak_rate=1.0, clock=clock)

    for _ in range(10):
        decision = await limiter.check("client-a")
        assert decision.allowed
        assert decision.delay == 0.0, "paced traffic should pass through untouched"
        clock.advance(1.0)  # exactly the drain rate


async def test_queue_full_rejects_rather_than_queueing_forever(clock: FakeClock):
    """Shaping stops at capacity; past it, an honest 429.

    Admitting beyond capacity would mean unbounded queue depth and a request
    'accepted' then left waiting for minutes.
    """
    limiter = InMemoryLeakyBucket(capacity=5, leak_rate=1.0, clock=clock)

    for _ in range(5):
        assert (await limiter.check("client-a")).allowed

    denied = await limiter.check("client-a")
    assert not denied.allowed
    assert denied.delay == 0.0
    assert denied.retry_after == pytest.approx(1.0), "one drain interval frees one slot"


async def test_retry_after_actually_clears(clock: FakeClock):
    """Waiting the advertised time must get the client in, at awkward rates."""
    for rate in (0.3, 1.0 / 3.0, 7.0 / 11.0, 0.07):
        clock.set(1_000_000.0)
        limiter = InMemoryLeakyBucket(capacity=2, leak_rate=rate, clock=clock)

        assert (await limiter.check("client-a")).allowed
        assert (await limiter.check("client-a")).allowed
        denied = await limiter.check("client-a")
        assert not denied.allowed

        clock.advance(denied.retry_after)
        assert (await limiter.check("client-a")).allowed, f"retry_after too short at {rate}"


async def test_queue_drains_over_time(clock: FakeClock):
    """Capacity returns continuously as the queue leaks."""
    limiter = InMemoryLeakyBucket(capacity=4, leak_rate=1.0, clock=clock)

    for _ in range(4):
        assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    clock.advance(2.0)  # two slots have drained
    assert (await limiter.check("client-a")).allowed
    assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed


async def test_delay_never_exceeds_a_full_drain(clock: FakeClock):
    """Queue depth bounds worst-case added latency.

    capacity / leak_rate is the longest any admitted request can wait, which
    is what makes this safe to await in middleware.
    """
    capacity, rate = 20, 4.0
    limiter = InMemoryLeakyBucket(capacity=capacity, leak_rate=rate, clock=clock)

    worst_case = capacity / rate
    for _ in range(100):
        decision = await limiter.check("client-a")
        if decision.allowed:
            assert decision.delay < worst_case


async def test_shapes_where_the_token_bucket_meters(clock: FakeClock):
    """Direct contrast: same burst, same admissions, different output.

    Configured equivalently, both admit the full burst. The token bucket lets
    all of it through instantly; the leaky bucket admits the same requests but
    paces them out.
    """
    tokens = InMemoryTokenBucket(capacity=10, refill_rate=2.0, clock=clock)
    leaky = InMemoryLeakyBucket(capacity=10, leak_rate=2.0, clock=clock)

    token_decisions = [await tokens.check("client-a") for _ in range(10)]
    leaky_decisions = [await leaky.check("client-a") for _ in range(10)]

    assert all(d.allowed for d in token_decisions)
    assert all(d.allowed for d in leaky_decisions)

    assert all(d.delay == 0.0 for d in token_decisions), "token bucket does not shape"
    assert sum(d.delay for d in leaky_decisions) > 0.0, "leaky bucket shapes"
    assert max(d.delay for d in leaky_decisions) == pytest.approx(4.5)


async def test_from_limit_window_maps_onto_capacity_and_rate(clock: FakeClock):
    limiter = InMemoryLeakyBucket.from_limit_window(limit=120, window=60.0, clock=clock)

    assert limiter.capacity == 120
    assert limiter.leak_rate == pytest.approx(2.0)
    assert limiter.limit == 120
    assert limiter.window == pytest.approx(60.0)


@pytest.mark.parametrize(
    "capacity,leak_rate",
    [(0, 1.0), (-1, 1.0), (5, 0.0), (5, -1.0)],
)
async def test_invalid_configuration_is_rejected(capacity, leak_rate):
    with pytest.raises(ValueError):
        InMemoryLeakyBucket(capacity=capacity, leak_rate=leak_rate)


@pytest.mark.parametrize("window", [0.0, -1.0])
async def test_from_limit_window_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        InMemoryLeakyBucket.from_limit_window(limit=5, window=window)
