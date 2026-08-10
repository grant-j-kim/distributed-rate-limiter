"""Name -> algorithm lookup, so limiters can be configured by string.

Lets `@rate_limit(algorithm="token_bucket", ...)` work without callers
importing implementation classes.
"""

from __future__ import annotations

from typing import Callable

from distributed_rate_limiter.base import Clock, RateLimiter, default_clock
from distributed_rate_limiter.memory.fixed_window import InMemoryFixedWindow
from distributed_rate_limiter.memory.leaky_bucket import InMemoryLeakyBucket
from distributed_rate_limiter.memory.sliding_window_counter import (
    InMemorySlidingWindowCounter,
)
from distributed_rate_limiter.memory.sliding_window_log import InMemorySlidingWindowLog
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket

# The bucket algorithms take (capacity, rate); their classmethods adapt them
# to the (limit, window) vocabulary the HTTP layer speaks.
ALGORITHMS: dict[str, Callable[..., RateLimiter]] = {
    "fixed_window": InMemoryFixedWindow,
    "sliding_window_log": InMemorySlidingWindowLog,
    "sliding_window_counter": InMemorySlidingWindowCounter,
    "token_bucket": InMemoryTokenBucket.from_limit_window,
    "leaky_bucket": InMemoryLeakyBucket.from_limit_window,
}


def create_limiter(
    algorithm: str,
    limit: int,
    window: float,
    *,
    clock: Clock = default_clock,
    **options: object,
) -> RateLimiter:
    """Build a limiter by name.

    Raises ValueError listing the valid names rather than KeyError, since a
    typo here is a configuration mistake and the caller deserves the options.

    `options` are forwarded to the algorithm, for settings only some of them
    have -- currently the leaky bucket's `max_delay` queue timeout.
    """
    try:
        factory = ALGORITHMS[algorithm]
    except KeyError:
        valid = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unknown algorithm {algorithm!r}; choose one of: {valid}") from None
    try:
        return factory(limit=limit, window=window, clock=clock, **options)
    except TypeError as exc:
        raise ValueError(f"{algorithm} does not accept {sorted(options)}: {exc}") from None
