"""Behaviour specific to the Redis sliding window counter."""

from __future__ import annotations

import uuid

import pytest

from distributed_rate_limiter.memory.sliding_window_counter import (
    InMemorySlidingWindowCounter,
)
from distributed_rate_limiter.redis_backend.sliding_window_counter import (
    RedisSlidingWindowCounter,
)
from tests.conftest import FakeClock


@pytest.fixture
def redis_or_skip(redis_client):
    if redis_client is None:
        pytest.skip("no Redis server available")
    return redis_client


@pytest.fixture
def prefix() -> str:
    return f"drltest:{uuid.uuid4().hex}"


async def test_matches_the_in_memory_reference_exactly(redis_or_skip, prefix):
    """Same weighting maths, same answers, across the language boundary.

    The weighting is floating point arithmetic performed in Lua rather than
    Python, so agreeing on every decision -- including the ones that hinge on
    an estimate sitting a hair under the limit -- is the real check.
    """
    window, limit = 60.0, 10
    clock = FakeClock()
    redis_limiter = RedisSlidingWindowCounter(
        redis_or_skip, limit=limit, window=window, prefix=prefix, clock=clock
    )
    memory_limiter = InMemorySlidingWindowCounter(limit=limit, window=window, clock=clock)

    boundary = (int(clock() // window) + 1) * window
    # Straddle a boundary, then probe the decay curve at several points.
    offsets = (
        [boundary - 1.0] * 12
        + [boundary + 0.001] * 4
        + [boundary + window * 0.25] * 4
        + [boundary + window * 0.5] * 6
        + [boundary + window * 0.9] * 6
        + [boundary + window * 2.5] * 12
    )

    redis_answers, memory_answers = [], []
    for at in offsets:
        clock.set(at)
        redis_answers.append((await redis_limiter.check("client-a")).allowed)
        memory_answers.append((await memory_limiter.check("client-a")).allowed)

    assert redis_answers == memory_answers, (
        f"disagreement:\n  redis:  {redis_answers}\n  memory: {memory_answers}"
    )


async def test_admits_exactly_one_extra_past_a_boundary(redis_or_skip, prefix):
    """The known over-admission, reproduced server-side.

    With previous == limit the estimate is limit * overlap, strictly under the
    limit for any elapsed time above zero, so exactly one request slips
    through -- far from the fixed window's full second allowance.
    """
    window = 60.0
    clock = FakeClock()
    limiter = RedisSlidingWindowCounter(
        redis_or_skip, limit=5, window=window, prefix=prefix, clock=clock
    )

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 1.0)
    for _ in range(5):
        assert (await limiter.check("client-a")).allowed

    clock.set(boundary + 0.001)
    allowed = 0
    for _ in range(5):
        if (await limiter.check("client-a")).allowed:
            allowed += 1
    assert allowed == 1


async def test_key_survives_into_the_next_window(redis_or_skip, prefix):
    """The TTL rule unique to this algorithm.

    This window's count is read as `prev` during the next one. A TTL of a
    single window would drop that history and hand the client a clean slate
    at every boundary -- reintroducing the fixed window's cliff, which is the
    whole thing this algorithm avoids.
    """
    window = 30.0
    limiter = RedisSlidingWindowCounter(
        redis_or_skip, limit=5, window=window, prefix=prefix
    )
    await limiter.check("client-a")

    ttl_ms = await redis_or_skip.pttl(f"{prefix}:client-a")
    assert ttl_ms > window * 1000, (
        f"TTL {ttl_ms}ms does not reach the next window; history would be lost"
    )


async def test_state_is_three_fields(redis_or_skip, prefix):
    """O(1) per key, and single-key so it stays Cluster-safe."""
    clock = FakeClock()
    limiter = RedisSlidingWindowCounter(
        redis_or_skip, limit=5, window=60.0, prefix=prefix, clock=clock
    )
    for _ in range(50):
        await limiter.check("client-a")

    stored = await redis_or_skip.hgetall(f"{prefix}:client-a")
    assert set(stored) == {"idx", "cur", "prev"}

    keys = [k async for k in redis_or_skip.scan_iter(match=f"{prefix}:*")]
    assert len(keys) == 1, f"expected a single key per client, found {keys}"


async def test_history_is_dropped_after_two_idle_windows(redis_or_skip, prefix):
    clock = FakeClock()
    limiter = RedisSlidingWindowCounter(
        redis_or_skip, limit=5, window=60.0, prefix=prefix, clock=clock
    )

    for _ in range(5):
        assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    clock.advance(120.0)
    for _ in range(5):
        assert (await limiter.check("client-a")).allowed


async def test_retry_after_clears_the_estimate(redis_or_skip, prefix):
    """Waiting the advertised time must actually get through."""
    window = 60.0
    clock = FakeClock()
    limiter = RedisSlidingWindowCounter(
        redis_or_skip, limit=10, window=window, prefix=prefix, clock=clock
    )

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 0.001)
    for _ in range(10):
        assert (await limiter.check("client-a")).allowed

    clock.set(boundary + 1.0)
    assert (await limiter.check("client-a")).allowed  # the one over-admission
    denied = await limiter.check("client-a")
    assert not denied.allowed

    clock.advance(denied.retry_after)
    assert (await limiter.check("client-a")).allowed
