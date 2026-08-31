"""Replay one arrival schedule against one limiter, with no time passing.

The load test in `loadtest/` drives real Redis over real wall clock time: a
ten second scenario takes ten seconds. That is the right way to *measure*, and
the wrong way to serve a web page where the visitor drags a slider and expects
an answer.

So the playground replays the same schedules against the **in-memory**
limiters on a driven clock. `check()` reads whatever `now` this module last
set, so a ten second scenario resolves in microseconds and costs no Redis
commands at all. The algorithms are the real ones -- this is not a
reimplementation -- but the numbers it produces are not the numbers in
`loadtest/README.md`, and the page must say so. Those were measured with a
network and a connection pool in the way; these are the same algorithms minus
that noise.

Two things this deliberately does not do:

- **It does not advance the clock by `Decision.delay`.** A shaped request is
  deferred, but requests keep arriving *during* that deferral, so advancing
  the clock would fast forward past them. The delay is recorded on the outcome
  instead, which is also what lets the timeline draw a shaped request twice:
  hollow where it arrived, filled where it was actually let through.
- **It does not fire concurrent arrivals concurrently.** Arrivals sharing an
  offset are applied in order at one timestamp. Against Redis they genuinely
  race, and that race is the subject of the punchline demo; here determinism
  is worth more, because the same slider position must always give the same
  picture.
"""

from __future__ import annotations

from dataclasses import dataclass

from distributed_rate_limiter.base import RateLimiter
from loadtest.traffic import Arrival


@dataclass(frozen=True)
class Outcome:
    """What happened to one request, and when it was actually let through."""

    offset: float
    """When the request arrived, in seconds from the start of the run."""

    client: str
    """The rate limit key it was charged against."""

    allowed: bool

    remaining: int
    """Quota left immediately after this check, as the limiter reported it."""

    delay: float = 0.0
    """Seconds the caller must wait before proceeding. Non-zero only for the
    leaky bucket, which shapes rather than meters."""

    @property
    def departure(self) -> float:
        """When the request actually proceeds -- arrival plus any shaping."""
        return self.offset + self.delay


async def replay(schedule: list[Arrival], limiter: RateLimiter) -> list[Outcome]:
    """Drive `limiter` through `schedule` on a simulated clock.

    `limiter` must already have been built with the clock this function
    drives -- use `build` below rather than constructing one yourself, since
    a limiter holding the real wall clock would ignore the schedule entirely
    and quietly admit everything.
    """
    outcomes = []
    previous = float("-inf")

    for arrival in schedule:
        if arrival.offset < previous:
            # The limiters compute elapsed time by subtraction. A clock that
            # goes backwards yields negative refill and nonsense counts, with
            # nothing raised -- so refuse the schedule instead of reporting it.
            raise ValueError(
                f"schedule must be ordered by offset; {arrival.offset} follows {previous}. "
                "Use loadtest.traffic.merge() to combine schedules."
            )
        previous = arrival.offset

        _set_now(limiter, arrival.offset)
        decision = await limiter.check(arrival.client)
        outcomes.append(
            Outcome(
                offset=arrival.offset,
                client=arrival.client,
                allowed=decision.allowed,
                remaining=decision.remaining,
                delay=decision.delay,
            )
        )

    return outcomes


class _DrivenClock:
    """A clock the replay moves by hand.

    A plain mutable float in a closure would do, but attaching it to the
    limiter keeps the pairing visible: a limiter built by `build` carries the
    clock that drives it, so `replay` cannot be handed a limiter running on
    real time without failing loudly.
    """

    __slots__ = ("now",)

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _set_now(limiter: RateLimiter, now: float) -> None:
    clock = getattr(limiter, "_clock", None)
    if not isinstance(clock, _DrivenClock):
        raise TypeError(
            "replay() needs a limiter built by web.replay.build(); this one is "
            "running on a clock it does not control, so the schedule would be ignored."
        )
    clock.now = now


def build(algorithm: str, limit: int, window: float, **options: object) -> RateLimiter:
    """An in-memory limiter wired to a clock `replay` can drive.

    Always the memory backend: the playground must cost no Redis commands, and
    a driven clock is meaningless against a server that reads its own TIME.
    """
    from distributed_rate_limiter import create_limiter

    return create_limiter(
        algorithm, limit=limit, window=window,
        backend="memory", clock=_DrivenClock(), **options,
    )


def peak_admission(outcomes: list[Outcome], window: float) -> int:
    """Most requests admitted to any one key inside any window-length interval.

    The metric the page reports, and there are two ways to get it wrong that
    both produce a plausible number.

    **Count each admission at its departure, not its arrival.** For a shaper,
    the delay *is* the mechanism: a request admitted at 2.0s and released at
    2.4s did not occupy the window at 2.0s. Counting arrivals reports the leaky
    bucket as though it never shaped -- on the boundary_burst schedule that is
    23 (1.15x) instead of 21 (1.05x), which is the difference between "shapes
    traffic" and "behaves like everything else". `loadtest/runner.py` logs
    `sent + delay` for exactly this reason; this is the same decision, made
    again, in the rig that has to agree with it.

    **Compute per key, then take the maximum -- never pool.** Pooling adds up
    independent quotas and reports three well-behaved clients as a 3x breach.
    """
    from loadtest.analysis import max_in_sliding_window

    by_client: dict[str, list[float]] = {}
    for o in outcomes:
        if o.allowed:
            by_client.setdefault(o.client, []).append(o.departure)

    return max((max_in_sliding_window(times, window) for times in by_client.values()),
               default=0)
