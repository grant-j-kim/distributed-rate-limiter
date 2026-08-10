"""The three traffic patterns, and how the five limiters are configured.

Every scenario uses limit=20 per window=2s. The short window is deliberate:
window boundary behaviour is scale free in window units, so a 2 second window
shows exactly what a 60 second one does while letting a run that spans five
boundaries finish in ten seconds instead of five minutes. The time, the Redis
server and the concurrency are all real; only the window is compressed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from distributed_rate_limiter.base import RateLimiter
from distributed_rate_limiter.redis_backend.fixed_window import RedisFixedWindow
from distributed_rate_limiter.redis_backend.leaky_bucket import RedisLeakyBucket
from distributed_rate_limiter.redis_backend.sliding_window_counter import (
    RedisSlidingWindowCounter,
)
from distributed_rate_limiter.redis_backend.sliding_window_log import (
    RedisSlidingWindowLog,
)
from distributed_rate_limiter.redis_backend.token_bucket import RedisTokenBucket

from loadtest.traffic import Arrival, burst, merge, poisson, steady

LIMIT = 20
WINDOW = 2.0

# Order is fixed so every plot legend, every summary table and every colour
# assignment agrees across runs.
ALGORITHMS = [
    "fixed_window",
    "sliding_window_log",
    "sliding_window_counter",
    "token_bucket",
    "leaky_bucket",
]


def build_limiter(algorithm: str, client: Any, *, prefix: str) -> RateLimiter:
    """One Redis-backed limiter, configured to the same limit and window.

    The buckets are built through `from_limit_window` so all five are given
    the same budget in the same vocabulary; comparing a bucket configured by
    (capacity, rate) against a window configured by (limit, window) would be
    comparing two different allowances and calling the difference an
    algorithmic one.

    The leaky bucket gets `max_delay=WINDOW`. Without it the shaper queues
    rather than rejects, and would report zero rejections until its capacity
    filled -- technically true, and useless next to four algorithms that
    answer immediately. A queue longer than the window is indistinguishable
    from a rejection to any client that would have retried by then, so this
    makes its output comparable without changing what it does.
    """
    factories: dict[str, Callable[..., RateLimiter]] = {
        "fixed_window": RedisFixedWindow,
        "sliding_window_log": RedisSlidingWindowLog,
        "sliding_window_counter": RedisSlidingWindowCounter,
        "token_bucket": RedisTokenBucket.from_limit_window,
        "leaky_bucket": lambda **kw: RedisLeakyBucket.from_limit_window(
            max_delay=WINDOW, **kw
        ),
    }
    return factories[algorithm](client=client, limit=LIMIT, window=WINDOW, prefix=prefix)


def fresh_prefix(scenario: str, algorithm: str) -> str:
    """A unique key namespace, so no run ever inherits another's state."""
    return f"drlload:{scenario}:{algorithm}:{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    """What this scenario is actually asking. Stated so a plot that answers a
    different question is recognisable as a mistake rather than a finding."""
    arrivals: list[Arrival]
    duration: float
    clients: tuple[str, ...]


def steady_over_limit() -> Scenario:
    """Smooth demand at twice the sustainable rate, for five windows.

    The limit is 20 per 2s, i.e. 10/s sustained; this offers 20/s with no
    burstiness at all. Every algorithm should reject about half, so the
    interesting part is not *how many* but *when*: the fixed window admits its
    whole allowance in a clump at each boundary and then rejects for the rest
    of the window, while the buckets spread admissions evenly across it.
    """
    return Scenario(
        name="steady_over_limit",
        question="With identical smooth over-limit demand, how is the same "
        "allowance distributed in time?",
        arrivals=steady(rate=20.0, duration=10.0, client="client-a"),
        duration=10.0,
        clients=("client-a",),
    )


def boundary_burst() -> Scenario:
    """Two bursts straddling a window boundary. The discriminating case.

    30 requests land 0.15s before a boundary and 30 more land 0.15s after it.
    The fixed window has no memory across the boundary, so it should admit its
    full allowance twice in 0.3 seconds -- 2x the limit over an interval that
    is a fraction of a window, which is the textbook failure the sliding
    algorithms exist to prevent. A third burst lands mid-window later, as a
    control: with no boundary nearby, the five should agree far more closely.
    """
    return Scenario(
        name="boundary_burst",
        question="What happens to a burst that straddles a window boundary, "
        "and how much does each algorithm over-admit?",
        arrivals=merge(
            burst(WINDOW - 0.15, 30, client="client-a"),
            burst(WINDOW + 0.15, 30, client="client-a"),
            burst(WINDOW * 3.5, 30, client="client-a"),  # mid-window control
        ),
        duration=9.0,
        clients=("client-a",),
    )


def multi_client() -> Scenario:
    """Three clients with different appetites against one limiter.

    Quota is per key, so a client staying under its limit must be unaffected
    by one hammering far over it. Worth measuring rather than assuming: a
    limiter that shared state across keys -- or a Lua script that wrote to a
    key it was not given -- would show up here as collateral rejections for
    the well behaved client, and nowhere else in this suite.
    """
    return Scenario(
        name="multi_client",
        question="Does one client exceeding its quota cost a well behaved "
        "client anything?",
        arrivals=merge(
            steady(rate=5.0, duration=10.0, client="polite"),  # half the limit
            steady(rate=25.0, duration=10.0, client="greedy"),  # 2.5x the limit
            poisson(rate=10.0, duration=10.0, seed=2_718_281, client="spiky"),
        ),
        duration=10.0,
        clients=("polite", "greedy", "spiky"),
    )


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "steady_over_limit": steady_over_limit,
    "boundary_burst": boundary_burst,
    "multi_client": multi_client,
}
