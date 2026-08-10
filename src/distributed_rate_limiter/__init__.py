"""Rate limiting algorithms that stay correct across multiple server instances.

The FastAPI integration (`RateLimitMiddleware`, `rate_limit`) lives in
`distributed_rate_limiter.middleware` and is imported from there, so the
algorithms stay usable without FastAPI installed.
"""

from distributed_rate_limiter.base import Clock, Decision, RateLimiter
from distributed_rate_limiter.memory.fixed_window import InMemoryFixedWindow
from distributed_rate_limiter.memory.leaky_bucket import InMemoryLeakyBucket
from distributed_rate_limiter.memory.sliding_window_counter import (
    InMemorySlidingWindowCounter,
)
from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from distributed_rate_limiter.registry import ALGORITHMS, create_limiter

__all__ = [
    "ALGORITHMS",
    "create_limiter",
    "Clock",
    "Decision",
    "RateLimiter",
    "InMemoryFixedWindow",
    "InMemoryLeakyBucket",
    "InMemorySlidingWindowCounter",
    "InMemorySlidingWindowLog",
    "InMemoryTokenBucket",
]
