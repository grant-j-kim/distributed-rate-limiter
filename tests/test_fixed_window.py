"""Behaviour specific to the fixed window counter.

The shared scenarios live in test_common_scenarios.py. What is tested here is
the algorithm's characteristic *flaw* -- the boundary burst -- plus its window
alignment, which is what makes the Redis version safe to run on many instances.
"""

from __future__ import annotations

import pytest

from distributed_rate_limiter.memory.fixed_window import InMemoryFixedWindow
from tests.conftest import FakeClock


async def test_boundary_burst_allows_double_the_limit(clock: FakeClock):
    """The known fixed-window flaw, asserted rather than hidden.

    A client can push `limit` requests at the end of one window and `limit`
    more at the start of the next -- 2x the limit within a span shorter than
    a single window. This is the baseline the sliding window algorithms are
    measured against in Milestone 4.
    """
    window = 60.0
    limiter = InMemoryFixedWindow(limit=5, window=window, clock=clock)

    # Park the clock 1 second before a window boundary.
    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 1.0)

    first_half = [await limiter.check("client-a") for _ in range(5)]
    assert all(d.allowed for d in first_half)

    # Step just past the boundary: a brand new window, full quota.
    clock.set(boundary + 0.001)

    second_half = [await limiter.check("client-a") for _ in range(5)]
    assert all(d.allowed for d in second_half)

    # 10 requests allowed within ~1 second, against a limit of 5 per 60s.
    assert len(first_half) + len(second_half) == 10


async def test_windows_are_epoch_aligned_not_first_request_aligned(clock: FakeClock):
    """Two instances started at different times must agree on the boundary.

    Lazily starting a window on a key's first request would give each server
    instance its own boundary for the same client. Epoch alignment means the
    boundary is a pure function of the timestamp.
    """
    window = 60.0
    boundary = (int(clock() // window) + 1) * window

    # One limiter first sees this key well before the boundary...
    early = InMemoryFixedWindow(limit=5, window=window, clock=clock)
    clock.set(boundary - 30.0)
    await early.check("client-a")

    # ...another only sees it just before. Both must reset at the same instant.
    late = InMemoryFixedWindow(limit=5, window=window, clock=clock)
    clock.set(boundary - 0.5)
    early_decision = await early.check("client-a")
    late_decision = await late.check("client-a")

    assert early_decision.reset_after == pytest.approx(0.5)
    assert late_decision.reset_after == pytest.approx(0.5)


async def test_reset_after_counts_down_within_a_window(clock: FakeClock):
    window = 60.0
    limiter = InMemoryFixedWindow(limit=100, window=window, clock=clock)
    clock.set((int(clock() // window)) * window)  # sit exactly on a boundary

    first = await limiter.check("client-a")
    assert first.reset_after == pytest.approx(60.0)

    clock.advance(45.0)
    later = await limiter.check("client-a")
    assert later.reset_after == pytest.approx(15.0)


async def test_rejected_requests_do_not_extend_the_window(clock: FakeClock):
    """Hammering while blocked must not push the reset further out.

    A naive implementation that refreshes the key's TTL on every request --
    including rejected ones -- locks the client out forever under sustained
    load. Guarding against that here matters more for the Redis version, where
    it is an easy EXPIRE mistake to make.
    """
    window = 60.0
    limiter = InMemoryFixedWindow(limit=1, window=window, clock=clock)
    clock.set((int(clock() // window)) * window)

    assert (await limiter.check("client-a")).allowed

    for _ in range(20):
        clock.advance(1.0)
        denied = await limiter.check("client-a")
        assert not denied.allowed

    # 20s in, the window should be 40s from resetting -- not 60s.
    assert denied.retry_after == pytest.approx(40.0)

    clock.advance(40.0)
    assert (await limiter.check("client-a")).allowed


@pytest.mark.parametrize(
    "limit,window",
    [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)],
)
async def test_invalid_configuration_is_rejected(limit, window):
    with pytest.raises(ValueError):
        InMemoryFixedWindow(limit=limit, window=window)
