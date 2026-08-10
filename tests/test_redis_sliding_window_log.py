"""Behaviour specific to the Redis sliding window log.

Shared scenarios run from test_common_scenarios.py and concurrency from
test_redis_concurrency.py. What is tested here is the sorted-set encoding:
member uniqueness, the bound on stored entries, and exact agreement with the
in-memory reference on the same traffic.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from distributed_rate_limiter.redis_backend.sliding_window_log import (
    RedisSlidingWindowLog,
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


async def test_identical_timestamps_do_not_collide(redis_or_skip, prefix, clock: FakeClock):
    """Every request needs its own sorted-set member.

    ZADD on an existing member updates that member's score rather than
    inserting a new entry. If requests sharing a timestamp also shared a
    member, the set would hold a single entry no matter how many arrived and
    the limiter would admit far more than the limit. With a frozen clock,
    every request here carries the same score -- so only distinct members
    keep the count honest.
    """
    limiter = RedisSlidingWindowLog(
        redis_or_skip, limit=5, window=60.0, prefix=prefix, clock=clock
    )

    allowed = 0
    for _ in range(20):
        if (await limiter.check("client-a")).allowed:
            allowed += 1

    assert allowed == 5, f"identical timestamps collapsed into one entry (allowed {allowed})"
    assert await redis_or_skip.zcard(f"{prefix}:client-a") == 5


async def test_stored_entries_never_exceed_the_limit(redis_or_skip, prefix, clock: FakeClock):
    """Per-key memory stays O(limit): rejected requests are not stored."""
    limiter = RedisSlidingWindowLog(
        redis_or_skip, limit=5, window=60.0, prefix=prefix, clock=clock
    )

    for _ in range(500):
        await limiter.check("client-a")
        clock.advance(0.001)

    assert await redis_or_skip.zcard(f"{prefix}:client-a") <= 5


async def test_no_boundary_burst(redis_or_skip, prefix, clock: FakeClock):
    """The exactness guarantee, over Redis.

    The Redis fixed window would allow 5 more here; this must allow none.
    """
    window = 60.0
    limiter = RedisSlidingWindowLog(
        redis_or_skip, limit=5, window=window, prefix=prefix, clock=clock
    )

    boundary = (int(clock() // window) + 1) * window
    clock.set(boundary - 1.0)
    for _ in range(5):
        assert (await limiter.check("client-a")).allowed

    clock.set(boundary + 0.001)
    for _ in range(5):
        assert not (await limiter.check("client-a")).allowed


async def test_old_entries_are_pruned_from_the_sorted_set(redis_or_skip, prefix, clock: FakeClock):
    """Entries must leave the set, not just stop counting."""
    limiter = RedisSlidingWindowLog(
        redis_or_skip, limit=5, window=60.0, prefix=prefix, clock=clock
    )

    for _ in range(5):
        assert (await limiter.check("client-a")).allowed
    assert await redis_or_skip.zcard(f"{prefix}:client-a") == 5

    clock.advance(60.0)
    assert (await limiter.check("client-a")).allowed

    # The five originals aged out and only the new one remains.
    assert await redis_or_skip.zcard(f"{prefix}:client-a") == 1


async def test_key_carries_a_ttl(redis_or_skip, prefix, clock: FakeClock):
    """An abandoned key must not live for ever."""
    limiter = RedisSlidingWindowLog(
        redis_or_skip, limit=5, window=30.0, prefix=prefix, clock=clock
    )
    await limiter.check("client-a")

    ttl = await redis_or_skip.pttl(f"{prefix}:client-a")
    assert 0 < ttl <= 30_000


async def test_matches_the_in_memory_reference_exactly(redis_or_skip, prefix):
    """Same traffic, same decisions -- asserted request by request.

    The shared scenario suite proves both implementations satisfy the same
    contract. This goes further and pins them to the identical sequence of
    allow/deny answers on a mixed trace, which is what 'the Redis version is
    a drop-in replacement' actually has to mean.
    """
    window = 10.0
    limit = 5
    clock = FakeClock()

    redis_limiter = RedisSlidingWindowLog(
        redis_or_skip, limit=limit, window=window, prefix=prefix, clock=clock
    )
    memory_limiter = InMemorySlidingWindowLog(limit=limit, window=window, clock=clock)

    start = clock()
    # A trace with bursts, gaps that partially drain, and a full drain.
    offsets = (
        [0.0] * 8
        + [1.0, 1.5, 2.0]
        + [5.0] * 4
        + [9.99, 10.0, 10.01]
        + [11.0] * 6
        + [30.0] * 7
    )

    redis_answers, memory_answers = [], []
    for offset in offsets:
        clock.set(start + offset)
        redis_answers.append((await redis_limiter.check("client-a")).allowed)
        memory_answers.append((await memory_limiter.check("client-a")).allowed)

    assert redis_answers == memory_answers, (
        "Redis and in-memory implementations disagreed:\n"
        f"  redis:  {redis_answers}\n"
        f"  memory: {memory_answers}"
    )


async def test_concurrent_requests_at_one_timestamp_hold_the_limit(redis_or_skip, prefix):
    """Concurrency and identical scores at the same time -- the worst case."""
    limiter = RedisSlidingWindowLog(redis_or_skip, limit=10, window=60.0, prefix=prefix)

    decisions = await asyncio.gather(*(limiter.check("client-a") for _ in range(200)))
    allowed = sum(1 for d in decisions if d.allowed)

    assert allowed == 10, f"expected exactly 10 under concurrency, got {allowed}"
