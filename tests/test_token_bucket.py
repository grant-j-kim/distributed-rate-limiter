"""Behaviour specific to the token bucket.

The shared scenarios live in test_common_scenarios.py. What is tested here is
what the window algorithms cannot express: burst capacity decoupled from
sustained rate, and continuous rather than quantised refill.
"""

from __future__ import annotations

import pytest

from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from tests.conftest import FakeClock


async def count_allowed(limiter, attempts: int, key: str = "client-a") -> int:
    allowed = 0
    for _ in range(attempts):
        if (await limiter.check(key)).allowed:
            allowed += 1
    return allowed


async def test_burst_capacity_is_independent_of_sustained_rate(clock: FakeClock):
    """The property no window algorithm can express.

    A bucket holding 100 tokens that refills at 1/s tolerates a 100-request
    burst but sustains only 1/s. Under a window algorithm those two numbers
    are the same knob.
    """
    limiter = InMemoryTokenBucket(capacity=100, refill_rate=1.0, clock=clock)

    # Full bucket: the whole burst goes through at once.
    assert await count_allowed(limiter, 100) == 100
    assert await count_allowed(limiter, 50) == 0

    # Sustained rate is 1/s regardless of that burst allowance.
    clock.advance(1.0)
    assert await count_allowed(limiter, 10) == 1
    clock.advance(1.0)
    assert await count_allowed(limiter, 10) == 1


async def test_fractional_tokens_accumulate(clock: FakeClock):
    """Refill below one token per second must still make progress.

    Storing tokens as an int would floor each partial refill to zero and the
    bucket would never recover -- a deadlock that only shows up at low rates.
    """
    limiter = InMemoryTokenBucket(capacity=1, refill_rate=0.5, clock=clock)

    assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    clock.advance(1.0)  # 0.5 tokens: still not enough for a whole request
    assert not (await limiter.check("client-a")).allowed

    clock.advance(1.0)  # now a full token has accrued
    assert (await limiter.check("client-a")).allowed


async def test_tokens_are_capped_at_capacity(clock: FakeClock):
    """Idling for an hour does not bank an hour's worth of burst."""
    limiter = InMemoryTokenBucket(capacity=5, refill_rate=1.0, clock=clock)

    assert await count_allowed(limiter, 5) == 5
    clock.advance(3600.0)

    assert await count_allowed(limiter, 100) == 5, "burst must stay bounded by capacity"


async def test_refill_is_continuous_not_quantised(clock: FakeClock):
    """Capacity returns smoothly, with no boundary and no cliff."""
    limiter = InMemoryTokenBucket(capacity=10, refill_rate=1.0, clock=clock)
    assert await count_allowed(limiter, 10) == 10

    # One token per second, available the moment it accrues.
    for _ in range(5):
        clock.advance(1.0)
        assert (await limiter.check("client-a")).allowed
        assert not (await limiter.check("client-a")).allowed

    # Half a second buys nothing; the other half completes the token.
    clock.advance(0.5)
    assert not (await limiter.check("client-a")).allowed
    clock.advance(0.5)
    assert (await limiter.check("client-a")).allowed


async def test_retry_after_actually_clears(clock: FakeClock):
    """Waiting exactly retry_after must succeed, at awkward rates too.

    The exact solution lands on 1.0 tokens and the check requires >= 1.0, so
    float error can leave a client that waited precisely as instructed still
    one hair short. Rates chosen to be unrepresentable in binary floating
    point.
    """
    for rate in (0.3, 1.0 / 3.0, 7.0 / 11.0, 0.07):
        clock.set(1_000_000.0)
        limiter = InMemoryTokenBucket(capacity=1, refill_rate=rate, clock=clock)

        assert (await limiter.check("client-a")).allowed
        denied = await limiter.check("client-a")
        assert not denied.allowed
        assert denied.retry_after is not None

        clock.advance(denied.retry_after)
        assert (await limiter.check("client-a")).allowed, f"retry_after too short at rate {rate}"


async def test_refill_is_preserved_when_a_request_is_rejected(clock: FakeClock):
    """Rejected requests must not discard accrued tokens.

    If the refill were only persisted on the allowed path, a client polling
    faster than the refill rate would keep resetting its own progress and
    never accumulate a token -- starvation that appears only under load.
    """
    limiter = InMemoryTokenBucket(capacity=1, refill_rate=1.0, clock=clock)
    assert (await limiter.check("client-a")).allowed

    # Poll ten times a second for a second: every one rejected until the
    # token completes, and the accrual must survive all that polling.
    for _ in range(9):
        clock.advance(0.1)
        assert not (await limiter.check("client-a")).allowed

    clock.advance(0.1)
    assert (await limiter.check("client-a")).allowed


async def test_smooths_the_burst_a_fixed_window_would_allow(clock: FakeClock):
    """Contrast with the fixed window's boundary behaviour.

    Configured equivalently to 5 per 60s, the bucket refills at 1/12s, so
    straddling a 60s boundary yields one extra token -- not a second full
    allowance of 5.
    """
    window = 60.0
    limiter = InMemoryTokenBucket.from_limit_window(limit=5, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 1.0)
    assert await count_allowed(limiter, 5) == 5

    clock.set(boundary + 1.0)
    assert await count_allowed(limiter, 5) == 0, "2s of refill is under one token"

    clock.set(boundary + 11.0)  # 12s after draining: exactly one token
    assert await count_allowed(limiter, 5) == 1


async def test_from_limit_window_maps_onto_capacity_and_rate(clock: FakeClock):
    limiter = InMemoryTokenBucket.from_limit_window(limit=120, window=60.0, clock=clock)

    assert limiter.capacity == 120
    assert limiter.refill_rate == pytest.approx(2.0)
    assert limiter.limit == 120
    assert limiter.window == pytest.approx(60.0)


async def test_new_keys_start_full(clock: FakeClock):
    """A first-time client gets its burst rather than waiting to fill."""
    limiter = InMemoryTokenBucket(capacity=3, refill_rate=1.0, clock=clock)
    assert await count_allowed(limiter, 3) == 3


@pytest.mark.parametrize(
    "capacity,refill_rate",
    [(0, 1.0), (-1, 1.0), (5, 0.0), (5, -1.0)],
)
async def test_invalid_configuration_is_rejected(capacity, refill_rate):
    with pytest.raises(ValueError):
        InMemoryTokenBucket(capacity=capacity, refill_rate=refill_rate)


@pytest.mark.parametrize("window", [0.0, -1.0])
async def test_from_limit_window_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        InMemoryTokenBucket.from_limit_window(limit=5, window=window)
