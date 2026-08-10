"""Arrival schedules: when each request is sent, and by which client.

A schedule is generated *once* and replayed against every algorithm. That is
the whole point. If each algorithm were driven by its own freshly generated
load -- even from the same distribution -- then any difference between two
result sets would be a mix of algorithm behaviour and traffic variation,
and no part of it could be attributed to either. Handing all five the byte
identical arrival sequence makes every difference in the output attributable
to the algorithm alone.

Offsets are seconds relative to the start of the run, not absolute times, so
one schedule can be replayed at any wall clock moment. The runner is what
pins offset 0.0 to a real instant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Arrival:
    """One request: when it is sent, and who sends it."""

    offset: float
    """Seconds after the start of the run."""

    client: str
    """The rate limit key. Distinct clients hold independent quota."""


def steady(rate: float, duration: float, *, client: str = "client", start: float = 0.0) -> list[Arrival]:
    """Evenly spaced arrivals at `rate` per second for `duration` seconds.

    The pathological case for the fixed window and the friendliest one for
    the buckets: perfectly smooth demand, where the only thing that can cause
    a rejection is the rate genuinely exceeding the limit.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    gap = 1.0 / rate
    count = int(duration * rate)
    return [Arrival(start + i * gap, client) for i in range(count)]


def burst(at: float, count: int, *, client: str = "client") -> list[Arrival]:
    """`count` requests all sent at the same instant.

    Every arrival shares one offset, so the runner fires them concurrently
    and they race each other into Redis. This is the shape that separates a
    correct distributed limiter from one that only looks correct serially --
    and, placed either side of a window boundary, the shape that separates
    the five algorithms from each other.
    """
    return [Arrival(at, client) for _ in range(count)]


def poisson(
    rate: float,
    duration: float,
    *,
    seed: int,
    client: str = "client",
    start: float = 0.0,
) -> list[Arrival]:
    """Exponentially distributed gaps: memoryless arrivals at mean `rate`/s.

    Closer to real traffic than `steady`, which no client ever produces.
    Seeded explicitly because a schedule that changed between runs would make
    two runs incomparable -- the same reason all five algorithms share one
    schedule within a run.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    rng = random.Random(seed)
    arrivals: list[Arrival] = []
    t = start
    end = start + duration
    while True:
        t += rng.expovariate(rate)
        if t >= end:
            return arrivals
        arrivals.append(Arrival(t, client))


def merge(*schedules: list[Arrival]) -> list[Arrival]:
    """Combine schedules into one ordered timeline.

    Sorting matters only for readability of the logs and for the runner's
    sleep loop; arrivals sharing an offset stay concurrent either way.
    """
    combined = [a for schedule in schedules for a in schedule]
    combined.sort(key=lambda a: (a.offset, a.client))
    return combined
