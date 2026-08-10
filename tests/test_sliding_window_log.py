"""Behaviour specific to the sliding window log.

The shared scenarios live in test_common_scenarios.py. What is tested here is
the property that distinguishes this algorithm from the fixed window -- no
boundary burst -- plus the eviction edge cases and the memory bound.
"""

from __future__ import annotations

import pytest

from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from tests.conftest import FakeClock


async def test_boundary_burst_is_rejected(clock: FakeClock):
    """The direct contrast with test_fixed_window's boundary burst.

    Identical traffic -- `limit` requests either side of a fixed-window
    boundary -- but here the trailing window sees all of them, so the second
    half is rejected. This pair of tests is the whole reason the sliding
    window exists.
    """
    window = 60.0
    limiter = InMemorySlidingWindowLog(limit=5, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 1.0)

    first_half = [await limiter.check("client-a") for _ in range(5)]
    assert all(d.allowed for d in first_half)

    clock.set(boundary + 0.001)

    second_half = [await limiter.check("client-a") for _ in range(5)]
    assert not any(d.allowed for d in second_half), (
        "sliding window must not permit the fixed window's 2x boundary burst"
    )


async def test_capacity_returns_gradually_not_all_at_once(clock: FakeClock):
    """Slots free one at a time as individual requests age out.

    The fixed window restores the entire quota at a boundary; the log restores
    exactly one slot per expiring entry.
    """
    limiter = InMemorySlidingWindowLog(limit=3, window=60.0, clock=clock)

    # Three requests spaced 10s apart, then exhausted.
    for _ in range(3):
        assert (await limiter.check("client-a")).allowed
        clock.advance(10.0)
    assert not (await limiter.check("client-a")).allowed

    # Requests landed at t=0/10/20 and the clock now sits at t=30. Step to
    # t=60 exactly, where only the first request has aged out.
    clock.advance(30.0)
    assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed, "only one slot should have freed"

    # Ten seconds later the second entry expires, freeing exactly one more.
    clock.advance(10.0)
    assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed


async def test_entry_exactly_one_window_old_has_expired(clock: FakeClock):
    """The window is half-open: (now - window, now].

    An inclusive cutoff would keep a stale entry one tick too long and make
    this algorithm disagree with the fixed window about what "one full window
    has passed" means.
    """
    limiter = InMemorySlidingWindowLog(limit=1, window=60.0, clock=clock)

    assert (await limiter.check("client-a")).allowed

    clock.advance(59.999)
    assert not (await limiter.check("client-a")).allowed

    clock.advance(0.001)  # now exactly 60.0s old
    assert (await limiter.check("client-a")).allowed


async def test_retry_after_points_at_the_oldest_entry(clock: FakeClock):
    """Wait time is until the next slot frees, not a full window."""
    limiter = InMemorySlidingWindowLog(limit=2, window=60.0, clock=clock)

    await limiter.check("client-a")
    clock.advance(25.0)
    await limiter.check("client-a")

    denied = await limiter.check("client-a")
    assert not denied.allowed
    # The oldest entry is 25s old, so it expires in 35s.
    assert denied.retry_after == pytest.approx(35.0)

    clock.advance(35.0)
    assert (await limiter.check("client-a")).allowed


async def test_rejected_requests_are_not_logged(clock: FakeClock):
    """Hammering while blocked must not refill the window.

    If rejected requests were logged, each one would push the expiry of the
    oldest live entry further out and the client would never recover. This is
    the sliding-window analogue of refreshing a TTL on every request.
    """
    limiter = InMemorySlidingWindowLog(limit=1, window=60.0, clock=clock)

    assert (await limiter.check("client-a")).allowed

    for _ in range(50):
        clock.advance(1.0)
        assert not (await limiter.check("client-a")).allowed

    # 50s of sustained rejected traffic; the original entry still expires on
    # schedule at t=60, not pushed out by the hammering.
    clock.advance(10.0)
    assert (await limiter.check("client-a")).allowed


async def test_log_never_grows_beyond_the_limit(clock: FakeClock):
    """Per-key memory is bounded by `limit`, even under heavy abuse."""
    limiter = InMemorySlidingWindowLog(limit=5, window=60.0, clock=clock)

    for _ in range(1000):
        await limiter.check("client-a")
        clock.advance(0.001)

    assert len(limiter._logs["client-a"]) <= 5


async def test_evicted_keys_do_not_leak_between_clients(clock: FakeClock):
    limiter = InMemorySlidingWindowLog(limit=2, window=60.0, clock=clock)

    for _ in range(2):
        assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    for _ in range(2):
        assert (await limiter.check("client-b")).allowed


@pytest.mark.parametrize(
    "limit,window",
    [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)],
)
async def test_invalid_configuration_is_rejected(limit, window):
    with pytest.raises(ValueError):
        InMemorySlidingWindowLog(limit=limit, window=window)
