"""Name -> implementation lookup, so limiters can be configured by string.

Lets `@rate_limit(algorithm="token_bucket", backend="redis", client=r)` work
without callers importing implementation classes -- which matters because the
whole point of the package is that an application picks a policy in config
and gets a correct distributed limiter, rather than wiring one by hand.

Two axes, not one: an algorithm and a backend. They are looked up together
because each (algorithm, backend) pair is its own class; there is no generic
store to swap underneath a shared algorithm, for the reason spelled out in
`base.RateLimiter`.
"""

from __future__ import annotations

from typing import Any, Callable

from distributed_rate_limiter.base import Clock, RateLimiter, default_clock
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

# The Redis limiters are imported unconditionally, and that is safe: they
# import nothing outside the standard library. They take the client as a
# duck-typed argument and only ever call `register_script` on it, so this
# module does not depend on redis-py being installed -- only on the caller
# having a client to hand in if they ask for the redis backend.

# The bucket algorithms take (capacity, rate); their classmethods adapt them
# to the (limit, window) vocabulary the HTTP layer speaks.
BACKENDS: dict[str, dict[str, Callable[..., RateLimiter]]] = {
    "memory": {
        "fixed_window": InMemoryFixedWindow,
        "sliding_window_log": InMemorySlidingWindowLog,
        "sliding_window_counter": InMemorySlidingWindowCounter,
        "token_bucket": InMemoryTokenBucket.from_limit_window,
        "leaky_bucket": InMemoryLeakyBucket.from_limit_window,
    },
    "redis": {
        "fixed_window": RedisFixedWindow,
        "sliding_window_log": RedisSlidingWindowLog,
        "sliding_window_counter": RedisSlidingWindowCounter,
        "token_bucket": RedisTokenBucket.from_limit_window,
        "leaky_bucket": RedisLeakyBucket.from_limit_window,
    },
}

# The in-memory table under its original name. Every backend offers the same
# five algorithms, so this doubles as the list of valid algorithm names.
ALGORITHMS = BACKENDS["memory"]


class _Unset:
    """Sentinel for 'the caller did not pass a clock'.

    None cannot serve here: for the Redis limiters None is a *meaningful*
    value that selects the server's own clock, so there would be no way to
    distinguish it from an unfilled default.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


def create_limiter(
    algorithm: str,
    limit: int,
    window: float,
    *,
    backend: str = "memory",
    clock: Clock | None | _Unset = _UNSET,
    **options: Any,
) -> RateLimiter:
    """Build a limiter by name.

        create_limiter("token_bucket", 100, 60)
        create_limiter("token_bucket", 100, 60, backend="redis", client=r)

    Raises ValueError listing the valid names rather than KeyError, since a
    typo here is a configuration mistake and the caller deserves the options.

    `options` are forwarded to the implementation: `client` and `prefix` for
    the Redis backend, and settings only some algorithms have -- currently the
    leaky bucket's `max_delay` queue timeout.

    **The clock default differs by backend, deliberately.** In-memory limiters
    fall back to the local wall clock. Redis limiters fall back to *no clock
    argument at all*, which makes them read Redis's own TIME. Defaulting this
    parameter to `default_clock` and forwarding it unconditionally would look
    harmless and would quietly hand every Redis limiter the local process
    clock instead -- so two application instances with a few seconds of skew
    would disagree about where a window boundary falls, while every test
    passed and nothing raised. The clock is therefore only forwarded when the
    caller actually supplied one.
    """
    try:
        table = BACKENDS[backend]
    except KeyError:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"unknown backend {backend!r}; choose one of: {valid}"
        ) from None

    try:
        factory = table[algorithm]
    except KeyError:
        valid = ", ".join(sorted(table))
        raise ValueError(
            f"unknown algorithm {algorithm!r}; choose one of: {valid}"
        ) from None

    if backend == "redis" and "client" not in options:
        raise ValueError(
            "the redis backend needs a client: "
            "create_limiter(..., backend='redis', client=redis.asyncio.Redis(...))"
        )

    if not isinstance(clock, _Unset):
        options["clock"] = clock
    elif backend == "memory":
        options["clock"] = default_clock

    try:
        return factory(limit=limit, window=window, **options)
    except TypeError as exc:
        unexpected = sorted(k for k in options if k != "clock")
        raise ValueError(
            f"{backend}/{algorithm} does not accept {unexpected}: {exc}"
        ) from None
