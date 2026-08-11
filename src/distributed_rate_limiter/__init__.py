"""Rate limiting algorithms that stay correct across multiple server instances.

Five algorithms, each in two implementations -- one in-memory, one backed by
Redis and atomic under real concurrency:

    from distributed_rate_limiter import create_limiter

    limiter = create_limiter("token_bucket", limit=100, window=60)          # local
    limiter = create_limiter("token_bucket", 100, 60,                       # shared
                             backend="redis", client=redis_client)

    decision = await limiter.check(client_id)
    if not decision.allowed:
        ...  # 429, Retry-After: decision.retry_after

The FastAPI integration (`RateLimitMiddleware`, `rate_limit`) lives in
`distributed_rate_limiter.middleware` and is imported from there, so the
algorithms stay usable without FastAPI installed. The Redis classes are safe
to import here unconditionally: they depend on nothing outside the standard
library and take the client as a duck-typed argument, so redis-py is only
needed by the application that supplies one.
"""

__version__ = "0.1.0"

from distributed_rate_limiter.base import Clock, Decision, RateLimiter
from distributed_rate_limiter.memory.fixed_window import InMemoryFixedWindow
from distributed_rate_limiter.memory.leaky_bucket import InMemoryLeakyBucket
from distributed_rate_limiter.memory.sliding_window_counter import (
    InMemorySlidingWindowCounter,
)
from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from distributed_rate_limiter.redis_backend.fixed_window import RedisFixedWindow
from distributed_rate_limiter.redis_backend.leaky_bucket import RedisLeakyBucket
from distributed_rate_limiter.redis_backend.sliding_window_counter import (
    RedisSlidingWindowCounter,
)
from distributed_rate_limiter.redis_backend.sliding_window_log import (
    RedisSlidingWindowLog,
)
from distributed_rate_limiter.redis_backend.token_bucket import RedisTokenBucket
from distributed_rate_limiter.registry import ALGORITHMS, BACKENDS, create_limiter

__all__ = [
    "__version__",
    "ALGORITHMS",
    "BACKENDS",
    "create_limiter",
    "Clock",
    "Decision",
    "RateLimiter",
    "InMemoryFixedWindow",
    "InMemoryLeakyBucket",
    "InMemorySlidingWindowCounter",
    "InMemorySlidingWindowLog",
    "InMemoryTokenBucket",
    "RedisFixedWindow",
    "RedisLeakyBucket",
    "RedisSlidingWindowCounter",
    "RedisSlidingWindowLog",
    "RedisTokenBucket",
]
