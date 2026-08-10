"""Behaviour specific to the Redis token and leaky buckets.

Shared scenarios and concurrency come from the common suites. What is tested
here is what moved into Lua: float arithmetic on the server, the TTL rule
that has no window to borrow from, and shaping state surviving a round trip.
"""

from __future__ import annotations

import uuid

import pytest

from distributed_rate_limiter.memory.leaky_bucket import InMemoryLeakyBucket
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from distributed_rate_limiter.redis_backend.leaky_bucket import RedisLeakyBucket
from distributed_rate_limiter.redis_backend.token_bucket import RedisTokenBucket
from tests.conftest import FakeClock


@pytest.fixture
def redis_or_skip(redis_client):
    if redis_client is None:
        pytest.skip("no Redis server available")
    return redis_client


@pytest.fixture
def prefix() -> str:
    return f"drltest:{uuid.uuid4().hex}"


# --------------------------------------------------------------------------
# Token bucket
# --------------------------------------------------------------------------


async def test_token_bucket_matches_the_in_memory_reference(redis_or_skip, prefix):
    """Identical decisions on a trace with bursts, partial refills and idling."""
    clock = FakeClock()
    redis_limiter = RedisTokenBucket(
        redis_or_skip, capacity=5, refill_rate=1.0, prefix=prefix, clock=clock
    )
    memory_limiter = InMemoryTokenBucket(capacity=5, refill_rate=1.0, clock=clock)

    start = clock()
    offsets = (
        [0.0] * 8  # drain the initial burst
        + [0.5, 0.99, 1.0, 1.01]  # partial refill either side of one token
        + [1.5] * 3
        + [4.0] * 4  # several tokens accrued
        + [100.0] * 7  # long idle: capped at capacity, not 100 tokens
    )

    redis_answers, memory_answers = [], []
    for offset in offsets:
        clock.set(start + offset)
        redis_answers.append((await redis_limiter.check("client-a")).allowed)
        memory_answers.append((await memory_limiter.check("client-a")).allowed)

    assert redis_answers == memory_answers, (
        f"disagreement:\n  redis:  {redis_answers}\n  memory: {memory_answers}"
    )


@pytest.mark.parametrize("rate", [0.3, 1.0 / 3.0, 7.0 / 11.0, 0.07])
async def test_token_bucket_retry_after_clears_in_lua(redis_or_skip, prefix, rate):
    """Waiting exactly retry_after must succeed, with the maths done in Lua.

    Lua numbers are IEEE doubles like Python floats, so the same accumulated
    drift applies. Rates chosen to be unrepresentable in binary.
    """
    clock = FakeClock()
    limiter = RedisTokenBucket(
        redis_or_skip, capacity=1, refill_rate=rate, prefix=f"{prefix}:{rate}", clock=clock
    )

    assert (await limiter.check("client-a")).allowed
    denied = await limiter.check("client-a")
    assert not denied.allowed

    clock.advance(denied.retry_after)
    assert (await limiter.check("client-a")).allowed, f"retry_after too short at rate {rate}"


async def test_token_bucket_fractional_refill_accumulates(redis_or_skip, prefix):
    """Sub-one-per-second refill must still make progress.

    Tokens are stored in a Redis hash as strings and parsed back with
    tonumber; if that round trip truncated to an integer, a rate below 1/s
    would never accrue anything and the bucket would deadlock.
    """
    clock = FakeClock()
    limiter = RedisTokenBucket(
        redis_or_skip, capacity=1, refill_rate=0.5, prefix=prefix, clock=clock
    )

    assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed

    clock.advance(1.0)  # 0.5 tokens: not yet a whole one
    assert not (await limiter.check("client-a")).allowed

    clock.advance(1.0)
    assert (await limiter.check("client-a")).allowed


async def test_token_bucket_key_outlives_a_full_refill(redis_or_skip, prefix):
    """The TTL rule the window algorithms do not have to think about.

    A token bucket has no window, so the key's TTL must exceed the time to
    refill from empty to full. Expiring sooner would delete a throttled
    client's state, and its next request would find no key and start again
    with a full bucket -- a free reset for anyone willing to wait.
    """
    capacity, rate = 10, 2.0  # 5s to refill completely
    limiter = RedisTokenBucket(
        redis_or_skip, capacity=capacity, refill_rate=rate, prefix=prefix
    )
    await limiter.check("client-a")

    ttl_ms = await redis_or_skip.pttl(f"{prefix}:client-a")
    full_refill_ms = (capacity / rate) * 1000

    assert ttl_ms > full_refill_ms, (
        f"TTL {ttl_ms}ms does not outlive a full refill ({full_refill_ms}ms); "
        "a throttled client could reset its bucket by waiting"
    )


async def test_token_bucket_burst_is_independent_of_rate(redis_or_skip, prefix):
    """The property no window algorithm has, preserved over Redis."""
    clock = FakeClock()
    limiter = RedisTokenBucket(
        redis_or_skip, capacity=50, refill_rate=1.0, prefix=prefix, clock=clock
    )

    allowed = 0
    for _ in range(50):
        if (await limiter.check("client-a")).allowed:
            allowed += 1
    assert allowed == 50, "full burst should pass immediately"
    assert not (await limiter.check("client-a")).allowed

    clock.advance(1.0)
    assert (await limiter.check("client-a")).allowed
    assert not (await limiter.check("client-a")).allowed, "sustained rate is 1/s"


async def test_token_bucket_state_is_two_fields(redis_or_skip, prefix):
    """O(1) state per key: a hash of tokens and a timestamp."""
    limiter = RedisTokenBucket(redis_or_skip, capacity=5, refill_rate=1.0, prefix=prefix)
    for _ in range(20):
        await limiter.check("client-a")

    stored = await redis_or_skip.hgetall(f"{prefix}:client-a")
    assert set(stored) == {"tokens", "ts"}


# --------------------------------------------------------------------------
# Leaky bucket
# --------------------------------------------------------------------------


async def test_leaky_bucket_matches_the_in_memory_reference(redis_or_skip, prefix):
    """Identical admissions *and* identical shaping delays."""
    clock = FakeClock()
    redis_limiter = RedisLeakyBucket(
        redis_or_skip, capacity=5, leak_rate=2.0, prefix=prefix, clock=clock
    )
    memory_limiter = InMemoryLeakyBucket(capacity=5, leak_rate=2.0, clock=clock)

    start = clock()
    offsets = [0.0] * 8 + [0.5, 1.0, 1.5] + [2.0] * 4 + [10.0] * 6

    redis_answers, memory_answers = [], []
    for offset in offsets:
        clock.set(start + offset)
        r = await redis_limiter.check("client-a")
        m = await memory_limiter.check("client-a")
        redis_answers.append((r.allowed, round(r.delay, 9)))
        memory_answers.append((m.allowed, round(m.delay, 9)))

    assert redis_answers == memory_answers, (
        f"disagreement:\n  redis:  {redis_answers}\n  memory: {memory_answers}"
    )


async def test_leaky_bucket_paces_admitted_requests(redis_or_skip, prefix):
    """Delay must survive the round trip, or the shaper becomes a meter."""
    clock = FakeClock()
    limiter = RedisLeakyBucket(
        redis_or_skip, capacity=10, leak_rate=2.0, prefix=prefix, clock=clock
    )

    delays = []
    for _ in range(10):
        decision = await limiter.check("client-a")
        assert decision.allowed
        delays.append(decision.delay)

    assert delays == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])


async def test_leaky_bucket_max_delay_rejects_instead_of_holding(redis_or_skip, prefix):
    """Queue timeout, enforced inside the script."""
    clock = FakeClock()
    limiter = RedisLeakyBucket.from_limit_window(
        redis_or_skip, limit=2, window=60.0, max_delay=5.0, prefix=prefix, clock=clock
    )

    first = await limiter.check("client-a")
    assert first.allowed and first.delay == 0.0

    denied = await limiter.check("client-a")
    assert not denied.allowed
    assert denied.retry_after == pytest.approx(25.0)


async def test_leaky_bucket_max_delay_rejection_does_not_consume_a_slot(redis_or_skip, prefix):
    """A timed-out request must not be charged for queue space."""
    clock = FakeClock()
    limiter = RedisLeakyBucket(
        redis_or_skip, capacity=10, leak_rate=1.0, max_delay=1.0, prefix=prefix, clock=clock
    )

    assert (await limiter.check("client-a")).allowed
    assert (await limiter.check("client-a")).allowed

    for _ in range(20):
        assert not (await limiter.check("client-a")).allowed

    clock.advance(2.0)
    admitted = await limiter.check("client-a")
    assert admitted.allowed and admitted.delay == pytest.approx(0.0)


async def test_leaky_bucket_key_outlives_a_full_drain(redis_or_skip, prefix):
    """Same TTL reasoning as the token bucket, mirrored."""
    capacity, rate = 10, 2.0
    limiter = RedisLeakyBucket(
        redis_or_skip, capacity=capacity, leak_rate=rate, prefix=prefix
    )
    await limiter.check("client-a")

    ttl_ms = await redis_or_skip.pttl(f"{prefix}:client-a")
    assert ttl_ms > (capacity / rate) * 1000


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (RedisTokenBucket, {"capacity": 0, "refill_rate": 1.0}),
        (RedisTokenBucket, {"capacity": 5, "refill_rate": 0.0}),
        (RedisLeakyBucket, {"capacity": 0, "leak_rate": 1.0}),
        (RedisLeakyBucket, {"capacity": 5, "leak_rate": -1.0}),
        (RedisLeakyBucket, {"capacity": 5, "leak_rate": 1.0, "max_delay": -1.0}),
    ],
)
async def test_invalid_configuration_is_rejected(redis_or_skip, cls, kwargs):
    with pytest.raises(ValueError):
        cls(redis_or_skip, **kwargs)
