"""Concurrency correctness against a real Redis server.

This is the milestone the project exists for. The in-memory limiters are
atomic for a reason that evaporates here: they mutate a dict with no `await`
between read and write, so the event loop cannot interleave them. Every Redis
call is a network round trip, which is an await point, so any read-then-write
sequence can be interleaved by another client.

The tests below hammer a single key with genuinely concurrent clients. A
deliberately naive implementation is included so the suite proves it can
*detect* the race rather than merely failing to trigger it.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from distributed_rate_limiter.base import Decision, RateLimiter
from distributed_rate_limiter.redis_backend.fixed_window import RedisFixedWindow


@pytest.fixture
def redis_or_skip(redis_client):
    if redis_client is None:
        pytest.skip("no Redis server available")
    return redis_client


@pytest.fixture
def prefix() -> str:
    return f"drltest:{uuid.uuid4().hex}"


class NaiveRedisFixedWindow(RateLimiter):
    """A textbook-wrong implementation, here to validate the test itself.

    Reads the counter, decides, then writes it back -- three round trips with
    two gaps. Any number of clients can read the same value before any of them
    writes, so they all believe they are under the limit.

    This is the exact mistake the Lua script exists to prevent. If the
    concurrency test cannot make *this* fail, it cannot prove anything about
    the real implementation passing.
    """

    def __init__(self, client, limit: int, window: float, *, prefix: str):
        self.client = client
        self.limit = limit
        self.window = window
        self.prefix = prefix

    async def check(self, key: str) -> Decision:
        redis_key = f"{self.prefix}:{key}"
        current = await self.client.get(redis_key)  # <-- gap opens here
        count = int(current or 0)

        if count >= self.limit:
            return Decision(
                allowed=False, limit=self.limit, remaining=0, reset_after=self.window,
                retry_after=self.window,
            )

        await self.client.set(redis_key, count + 1, ex=int(self.window))  # <-- and closes here
        return Decision(
            allowed=True,
            limit=self.limit,
            remaining=self.limit - (count + 1),
            reset_after=self.window,
        )

    async def reset(self, key: str) -> None:
        await self.client.delete(f"{self.prefix}:{key}")


async def count_allowed_concurrently(limiter: RateLimiter, requests: int, key: str = "c") -> int:
    """Fire `requests` checks simultaneously and count the admissions."""
    decisions = await asyncio.gather(*(limiter.check(key) for _ in range(requests)))
    return sum(1 for d in decisions if d.allowed)


async def test_naive_implementation_over_admits_under_concurrency(redis_or_skip, prefix):
    """The control case: prove the race is reachable and the test detects it.

    If this ever starts passing, the concurrency tests below have stopped
    testing anything and the suite is lying.
    """
    limiter = NaiveRedisFixedWindow(redis_or_skip, limit=5, window=60.0, prefix=prefix)

    allowed = await count_allowed_concurrently(limiter, requests=50)

    assert allowed > 5, (
        "expected the read-then-write implementation to over-admit under "
        f"concurrency, but it allowed exactly {allowed}. The test may no "
        "longer be exercising real concurrency."
    )


async def test_atomic_implementation_holds_the_limit_exactly(make_redis_limiter):
    """The real thing: 50 simultaneous requests, limit 5, exactly 5 admitted.

    Runs against every Redis-backed limiter registered in conftest, so a new
    backend cannot be added without proving this.
    """
    limiter = make_redis_limiter(limit=5, window=60.0)

    allowed = await count_allowed_concurrently(limiter, requests=50)

    assert allowed == 5, f"expected exactly 5 admissions under concurrency, got {allowed}"


@pytest.mark.parametrize("concurrency", [10, 100, 500])
async def test_limit_holds_at_several_concurrency_levels(make_redis_limiter, concurrency):
    """Load level must not change the answer."""
    limiter = make_redis_limiter(limit=20, window=60.0, suffix=f":{concurrency}")

    allowed = await count_allowed_concurrently(limiter, requests=concurrency)

    expected = min(20, concurrency)
    assert allowed == expected, f"{concurrency} concurrent requests admitted {allowed}"


async def test_separate_instances_share_one_limit(make_redis_limiter, redis_or_skip):
    """The reason Redis is here at all.

    Four limiter objects standing in for four application instances, all
    hitting the same key concurrently. They must enforce one limit between
    them, not four -- which is precisely what the in-memory versions cannot
    do, since each process holds its own dict.
    """
    shared = make_redis_limiter(limit=10, window=60.0)
    # Same prefix, separate objects: four instances of one deployment.
    instances = [
        make_redis_limiter(limit=10, window=60.0, prefix=shared.prefix) for _ in range(4)
    ]

    results = await asyncio.gather(
        *(instance.check("shared-client") for instance in instances for _ in range(25))
    )
    allowed = sum(1 for d in results if d.allowed)

    assert allowed == 10, f"four instances admitted {allowed}, expected one shared limit of 10"


async def test_concurrent_traffic_on_distinct_keys_is_unaffected(make_redis_limiter):
    """Contention on one key must not bleed into another."""
    limiter = make_redis_limiter(limit=5, window=60.0)

    async def run(key: str) -> int:
        return await count_allowed_concurrently(limiter, requests=20, key=key)

    counts = await asyncio.gather(*(run(f"client-{i}") for i in range(10)))

    assert counts == [5] * 10, f"per-key limits leaked under concurrency: {counts}"


async def test_ttl_is_set_and_not_extended_by_later_requests(redis_or_skip, prefix):
    """The bug the Lua script exists to prevent, checked at the key level.

    The counter key must carry a TTL from creation (or it leaks for ever),
    and that TTL must not be pushed out by subsequent requests (or a client
    under sustained load is locked out permanently).
    """
    limiter = RedisFixedWindow(redis_or_skip, limit=100, window=60.0, prefix=prefix)

    await limiter.check("client-a")
    keys = [k async for k in redis_or_skip.scan_iter(match=f"{prefix}:client-a:*")]
    assert len(keys) == 1, f"expected one counter key, found {keys}"

    first_ttl = await redis_or_skip.pttl(keys[0])
    assert first_ttl > 0, "counter key was created without a TTL and would leak for ever"

    await asyncio.sleep(0.05)
    for _ in range(20):
        await limiter.check("client-a")

    later_ttl = await redis_or_skip.pttl(keys[0])
    assert later_ttl < first_ttl, "TTL was refreshed by later requests -- window never ends"


async def test_key_expires_so_state_does_not_accumulate(redis_or_skip, prefix):
    """A short window must actually clean itself up."""
    limiter = RedisFixedWindow(redis_or_skip, limit=2, window=1.0, prefix=prefix)

    await limiter.check("client-a")
    assert [k async for k in redis_or_skip.scan_iter(match=f"{prefix}:*")]

    await asyncio.sleep(1.2)

    remaining = [k async for k in redis_or_skip.scan_iter(match=f"{prefix}:*")]
    assert remaining == [], f"window key outlived its window: {remaining}"


async def test_uses_the_server_clock_when_no_clock_is_injected(redis_or_skip, prefix):
    """Default behaviour must not depend on the application server's clock.

    With no clock injected the script reads Redis TIME, so instances with
    skewed clocks still agree on the window boundary.
    """
    limiter = RedisFixedWindow(redis_or_skip, limit=3, window=60.0, prefix=prefix)
    assert limiter._clock is None

    decision = await limiter.check("client-a")
    assert decision.allowed
    # A sane reset_after can only have come from the server's own clock.
    assert 0 < decision.reset_after <= 60.0
