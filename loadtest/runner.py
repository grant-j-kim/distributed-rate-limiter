"""The driver: replay an arrival schedule against one limiter, in real time.

Three things here are load bearing.

**Arrivals fire as independent tasks.** The obvious loop -- sleep, check,
sleep, check -- would let a slow check push every later arrival, so the
schedule the limiter actually saw would drift from the one under test, and it
would drift *more* the more the limiter is struggling. Giving each arrival its
own task that sleeps to its own deadline keeps the offered load fixed no
matter how the limiter responds. A burst of 200 scheduled at the same offset
genuinely arrives at once and queues at the bounded connection pool, which is
what a real server under a spike does.

**The run is aligned to a window boundary.** Three of the five algorithms key
off absolute wall clock windows (floor(now / window)), so a run starting at an
arbitrary phase puts the fixed window's cliff at an arbitrary place and two
runs cannot be compared. Alignment is computed from Redis's own TIME -- the
same clock the Lua scripts read -- so `offset 0.0` means "a window boundary"
to the limiter, not merely to this process.

**Shaping delays are awaited.** An allowed decision from the leaky bucket
carries a delay the caller owes before proceeding. A load generator that
recorded the admission and skipped the wait would be measuring a meter and
labelling it a shaper.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass

from distributed_rate_limiter.base import RateLimiter

from loadtest.traffic import Arrival


@dataclass(frozen=True)
class Record:
    """One request's full history, enough to re-derive every plot offline."""

    algorithm: str
    client: str
    scheduled: float
    """Offset the schedule asked for."""
    sent: float
    """Offset the check was actually issued at. Drift from `scheduled` is the
    generator's own error, and is reported so it can be ruled out as a cause
    of anything visible in the results."""
    allowed: bool
    remaining: int
    retry_after: float | None
    delay: float
    """Shaping delay owed before proceeding. Non-zero only for leaky bucket."""
    admitted: float | None
    """Offset the request was actually allowed to proceed at: `sent + delay`
    for an admitted request, None for a rejected one. For every algorithm but
    the leaky bucket this equals `sent`."""
    latency_ms: float
    """Round trip of the check itself, excluding any shaping delay."""


async def redis_now(client) -> float:
    """Read UNIX seconds from the Redis server, as the Lua scripts do."""
    seconds, microseconds = await client.time()
    return seconds + microseconds / 1_000_000


async def next_boundary(client, window: float, *, margin: float = 0.75) -> float:
    """Local timestamp of the next window boundary at least `margin` away.

    Computed in Redis's clock, then converted into this process's clock by
    offset. On one host those clocks are the same; doing the conversion anyway
    means the rig stays honest when Redis is on another machine, which is the
    deployment this whole project is about.
    """
    remote = await redis_now(client)
    local = time.time()
    skew = local - remote

    boundary = (int(remote // window) + 1) * window
    if boundary - remote < margin:
        boundary += window
    return boundary + skew


async def _fire(
    limiter: RateLimiter,
    algorithm: str,
    arrival: Arrival,
    t0: float,
    out: list[Record],
) -> None:
    """Wait for this arrival's moment, issue one check, record what happened."""
    wait = (t0 + arrival.offset) - time.time()
    if wait > 0:
        await asyncio.sleep(wait)

    sent = time.time()
    started = time.perf_counter()
    decision = await limiter.check(arrival.client)
    latency_ms = (time.perf_counter() - started) * 1000

    # Honour the shaper: an admitted request owes this wait before it may
    # proceed. Skipping it would turn the leaky bucket into a meter.
    if decision.allowed and decision.delay > 0:
        await asyncio.sleep(decision.delay)

    out.append(
        Record(
            algorithm=algorithm,
            client=arrival.client,
            scheduled=arrival.offset,
            sent=sent - t0,
            allowed=decision.allowed,
            remaining=decision.remaining,
            retry_after=decision.retry_after,
            delay=decision.delay,
            admitted=(sent - t0) + decision.delay if decision.allowed else None,
            latency_ms=latency_ms,
        )
    )


async def run_schedule(
    limiter: RateLimiter,
    algorithm: str,
    arrivals: list[Arrival],
    *,
    t0: float,
) -> list[Record]:
    """Replay `arrivals` against `limiter`, starting at local timestamp `t0`.

    Returns records in completion order; callers sort by `sent`.
    """
    records: list[Record] = []
    tasks = [
        asyncio.create_task(_fire(limiter, algorithm, arrival, t0, records))
        for arrival in arrivals
    ]
    await asyncio.gather(*tasks)
    return records


async def warm_up(limiter: RateLimiter) -> None:
    """Load the Lua script and open a connection before the clock matters.

    The first check against a fresh limiter pays a SCRIPT LOAD and a TCP
    handshake. Left in the data that shows up as a latency outlier on the
    first request of every run, which is an artefact of the rig rather than
    anything about the algorithm.
    """
    await limiter.check("__warmup__")
    await limiter.reset("__warmup__")


def to_json_rows(records: list[Record]) -> list[dict]:
    return [asdict(r) for r in sorted(records, key=lambda r: r.sent)]
