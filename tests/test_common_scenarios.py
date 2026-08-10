"""Scenarios every rate limiter must satisfy, regardless of algorithm.

This suite runs against every implementation registered in conftest.ALL_LIMITERS.
When the Redis backends land, they join this list and must pass unchanged --
that equivalence is the point.
"""

from __future__ import annotations

from tests.conftest import FakeClock


async def test_steady_traffic_under_limit_is_always_allowed(make_limiter, clock: FakeClock):
    """Traffic paced below the limit never gets rejected, however long it runs."""
    limiter = make_limiter(limit=10, window=60.0)

    # One request every 10s for 10 minutes: 6/min against a limit of 10/min.
    for _ in range(60):
        decision = await limiter.check("client-a")
        assert decision.allowed
        clock.advance(10.0)


async def test_exactly_at_limit_allows_then_rejects(make_limiter):
    """The limit-th request is allowed; the one after it is not."""
    limiter = make_limiter(limit=5, window=60.0)

    for i in range(5):
        decision = await limiter.check("client-a")
        assert decision.allowed, f"request {i + 1} of 5 should be allowed"
        assert decision.remaining == 5 - (i + 1)

    decision = await limiter.check("client-a")
    assert not decision.allowed
    assert decision.remaining == 0


async def test_burst_exceeding_limit_rejects_the_excess(make_limiter):
    """A 50-request burst against a limit of 5 allows exactly 5."""
    limiter = make_limiter(limit=5, window=60.0)

    decisions = [await limiter.check("client-a") for _ in range(50)]
    allowed = [d for d in decisions if d.allowed]

    assert len(allowed) == 5
    assert all(not d.allowed for d in decisions[5:])


async def test_rejection_reports_retry_after(make_limiter):
    """A rejected request tells the client how long to wait."""
    limiter = make_limiter(limit=1, window=60.0)

    assert (await limiter.check("client-a")).allowed
    denied = await limiter.check("client-a")

    assert not denied.allowed
    assert denied.retry_after is not None
    assert 0 < denied.retry_after <= 60.0


async def test_allowed_request_has_no_retry_after(make_limiter):
    limiter = make_limiter(limit=5, window=60.0)
    decision = await limiter.check("client-a")
    assert decision.allowed
    assert decision.retry_after is None


async def test_capacity_returns_after_a_full_window(make_limiter, clock: FakeClock):
    """Waiting out a whole window restores the full quota for any algorithm."""
    limiter = make_limiter(limit=5, window=60.0)

    for _ in range(5):
        assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    clock.advance(60.0)

    for i in range(5):
        assert (await limiter.check("client-a")).allowed, f"request {i + 1} after reset"


async def test_keys_are_independent(make_limiter):
    """Exhausting one client's quota must not affect another's."""
    limiter = make_limiter(limit=3, window=60.0)

    for _ in range(3):
        assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    for _ in range(3):
        assert (await limiter.check("client-b")).allowed


async def test_remaining_never_goes_negative(make_limiter):
    limiter = make_limiter(limit=2, window=60.0)
    decisions = [await limiter.check("client-a") for _ in range(10)]
    assert all(d.remaining >= 0 for d in decisions)


async def test_reset_clears_state(make_limiter):
    limiter = make_limiter(limit=2, window=60.0)

    for _ in range(2):
        await limiter.check("client-a")
    assert not (await limiter.check("client-a")).allowed

    await limiter.reset("client-a")
    assert (await limiter.check("client-a")).allowed
