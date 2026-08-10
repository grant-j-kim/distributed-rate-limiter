"""Core types shared by every rate limiter implementation.

The interface is async even though the in-memory limiters never await: the
Redis-backed versions will, and so does FastAPI middleware. Making that
choice up front keeps one interface for both backends.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Callable

# A clock returns UNIX seconds as a float.
#
# Deliberately wall-clock rather than time.monotonic(): monotonic clocks are
# per-process, so two server instances would disagree about where a window
# boundary falls. Wall clock is a shared reference across instances. The
# Redis implementations go one step further and read the time from the Redis
# server itself, so the store is the single source of truth.
Clock = Callable[[], float]

default_clock: Clock = time.time


@dataclass(frozen=True)
class Decision:
    """The outcome of a single rate limit check."""

    allowed: bool
    """Whether this request may proceed."""

    limit: int
    """The configured ceiling, echoed back for X-RateLimit-Limit."""

    remaining: int
    """Requests still available right now. Never negative."""

    reset_after: float
    """Seconds until capacity is (at least partly) restored."""

    retry_after: float | None = None
    """Seconds the client should wait before retrying. None when allowed.

    This is what the 429 response's Retry-After header carries.
    """


class RateLimiter(abc.ABC):
    """One rate limiting algorithm bound to one storage backend.

    Each algorithm/backend pair is a concrete subclass (InMemoryFixedWindow,
    later RedisFixedWindow, ...) rather than one algorithm class over a
    swappable store. That is intentional: an atomic token bucket cannot be
    built on a generic get-then-set store, because the read-modify-write gap
    *is* the race condition. Atomicity requires fusing the algorithm into the
    storage operation (INCR+EXPIRE, or a Lua script). Subclasses share this
    interface and a single parameterized test suite instead of sharing code
    that could not be made correct.
    """

    limit: int
    window: float

    @abc.abstractmethod
    async def check(self, key: str) -> Decision:
        """Consume one unit of quota for `key` and report the outcome.

        This both decides *and* records. Callers must treat one call as one
        request; calling it twice consumes twice.
        """

    @abc.abstractmethod
    async def reset(self, key: str) -> None:
        """Forget all state for `key`. Primarily for tests."""
