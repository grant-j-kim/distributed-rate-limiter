from __future__ import annotations

import functools
import os
import uuid
from typing import Callable, Protocol

import pytest

from distributed_rate_limiter.base import Clock, RateLimiter
from distributed_rate_limiter.memory.fixed_window import InMemoryFixedWindow
from distributed_rate_limiter.memory.leaky_bucket import InMemoryLeakyBucket
from distributed_rate_limiter.memory.sliding_window_counter import (
    InMemorySlidingWindowCounter,
)
from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from distributed_rate_limiter.redis_backend.fixed_window import RedisFixedWindow
from distributed_rate_limiter.redis_backend.sliding_window_log import (
    RedisSlidingWindowLog,
)

REDIS_URL = os.environ.get("DRL_TEST_REDIS_URL", "redis://localhost:6379/15")


class FakeClock:
    """A hand-cranked clock, so boundary tests are deterministic.

    Testing window boundaries with real time.sleep() makes the suite slow and
    flaky under CI load. Every algorithm takes an injected clock precisely so
    these tests can step over a boundary exactly.
    """

    def __init__(self, now: float = 1_000_000.0):
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def set(self, seconds: float) -> None:
        self._now = seconds


class LimiterFactory(Protocol):
    def __call__(self, limit: int, window: float, clock: Clock) -> RateLimiter: ...


# Every (algorithm, backend) pair registers here and inherits the shared
# scenario suite in test_common_scenarios.py. A Redis implementation joining
# this list and passing unchanged is what proves it behaves identically to
# its in-memory reference -- the equivalence is asserted, not assumed.
#
# Entries are (id, needs_redis, factory).
ALL_LIMITERS: list[tuple[str, bool, LimiterFactory]] = [
    ("fixed_window", False, InMemoryFixedWindow),
    ("sliding_window_log", False, InMemorySlidingWindowLog),
    ("sliding_window_counter", False, InMemorySlidingWindowCounter),
    # The token bucket's real signature is (capacity, refill_rate); the
    # classmethod adapts it to the (limit, window) the fixture speaks rather
    # than forcing the algorithm to give up its two independent knobs.
    ("token_bucket", False, InMemoryTokenBucket.from_limit_window),
    ("leaky_bucket", False, InMemoryLeakyBucket.from_limit_window),
    ("redis_fixed_window", True, RedisFixedWindow),
    ("redis_sliding_window_log", True, RedisSlidingWindowLog),
]

# Redis-backed limiters also inherit the concurrency suite in
# test_redis_concurrency.py. Every entry here must hold its limit exactly
# when hammered by simultaneous clients -- the property that distinguishes a
# correct distributed limiter from one that merely looks correct serially.
REDIS_LIMITERS: list[tuple[str, LimiterFactory]] = [
    ("fixed_window", RedisFixedWindow),
    ("sliding_window_log", RedisSlidingWindowLog),
]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def redis_client():
    """An async Redis client, or None when no server is reachable.

    Returning None rather than erroring lets the suite stay runnable without
    Redis; the Redis-backed parameters skip instead of failing.
    """
    try:
        import redis.asyncio as aioredis
    except ImportError:  # pragma: no cover - redis extra not installed
        yield None
        return

    # A blocking pool, so that firing hundreds of concurrent checks queues for
    # a connection instead of raising MaxConnectionsError. Real deployments
    # bound their pool the same way; the limiter has to be correct whether
    # requests reach Redis all at once or queue behind a connection.
    pool = aioredis.BlockingConnectionPool.from_url(
        REDIS_URL, decode_responses=True, max_connections=64, timeout=20
    )
    client = aioredis.Redis(connection_pool=pool)
    try:
        await client.ping()
    except Exception:  # pragma: no cover - no server running
        await client.aclose()
        yield None
        return

    yield client
    await client.aclose()


@pytest.fixture(
    params=[(needs_redis, f) for _, needs_redis, f in ALL_LIMITERS],
    ids=[name for name, _, _ in ALL_LIMITERS],
)
def make_limiter(request, clock: FakeClock, redis_client) -> Callable[..., RateLimiter]:
    """Builds one limiter of whichever implementation is under test."""
    needs_redis, factory = request.param

    if not needs_redis:
        return lambda limit=5, window=60.0: factory(limit=limit, window=window, clock=clock)

    if redis_client is None:
        pytest.skip(f"no Redis server at {REDIS_URL}")

    # A unique prefix per test keeps keys isolated without flushing the
    # database, so a stray FLUSHDB can never wipe something real.
    prefix = f"drltest:{uuid.uuid4().hex}"
    return functools.partial(
        factory, client=redis_client, clock=clock, prefix=prefix, limit=5, window=60.0
    )


@pytest.fixture(
    params=[f for _, f in REDIS_LIMITERS],
    ids=[name for name, _ in REDIS_LIMITERS],
)
def make_redis_limiter(request, redis_client) -> Callable[..., RateLimiter]:
    """Builds one Redis-backed limiter, for the concurrency suite.

    No clock is injected: these tests exercise the production path, where the
    script reads the time from Redis itself.
    """
    if redis_client is None:
        pytest.skip(f"no Redis server at {REDIS_URL}")

    factory: LimiterFactory = request.param

    def _make(limit: int = 5, window: float = 60.0, suffix: str = "") -> RateLimiter:
        prefix = f"drltest:{uuid.uuid4().hex}{suffix}"
        return factory(client=redis_client, limit=limit, window=window, prefix=prefix)

    return _make
