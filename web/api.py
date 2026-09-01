"""The playground endpoint: one schedule, five algorithms, one response.

Everything here runs the *in-memory* limiters on a clock `web.replay` drives,
so a request costs no Redis commands and no wall clock time. The numbers this
returns are therefore not the numbers in `loadtest/README.md`, which were
measured against real Redis with a network in the way -- the page says so, and
links there rather than quoting a figure.

The whole result goes back in one response. The browser animates it locally by
revealing marks on a timer; there is no streaming and no second request, so
dragging the slider is as fast as the arithmetic.
"""

from __future__ import annotations

from fastapi import FastAPI, Query

from loadtest.traffic import burst, merge
from web.replay import build, peak_admission, replay

LIMIT = 20
WINDOW = 2.0
BURST_SIZE = 30
HALF_GAP = 0.15
"""Half the separation between the paired bursts. They sit at `center` plus and
minus this, so the pair spans 0.30s however the slider moves it."""

CONTROL_AT = WINDOW * 3.5
"""A third burst, far from any boundary. With nothing nearby to straddle, the
five should agree closely -- it is what shows the disagreement is a boundary
phenomenon rather than a general one."""

ALGORITHMS = (
    "fixed_window",
    "sliding_window_log",
    "sliding_window_counter",
    "token_bucket",
    "leaky_bucket",
)

app = FastAPI(
    title="Rate limiter playground",
    description="Five algorithms, one arrival schedule, replayed on a simulated clock.",
)


def _schedule(center: float):
    """Two bursts `HALF_GAP` either side of `center`, plus the control burst.

    Sliding `center` onto a window boundary is the entire experiment: the same
    sixty requests that one algorithm spreads across two windows, another
    counts once.
    """
    return merge(
        burst(center - HALF_GAP, BURST_SIZE, client="client-a"),
        burst(center + HALF_GAP, BURST_SIZE, client="client-a"),
        burst(CONTROL_AT, BURST_SIZE, client="client-a"),
    )


@app.get("/api/replay")
async def api_replay(
    center: float = Query(
        WINDOW,
        ge=HALF_GAP,
        le=CONTROL_AT - HALF_GAP,
        description="Where the burst pair sits, in seconds from the start of the run. "
        f"At {WINDOW} it straddles a window boundary.",
    ),
) -> dict:
    schedule = _schedule(center)
    results = []

    for algorithm in ALGORITHMS:
        outcomes = await replay(schedule, build(algorithm, LIMIT, WINDOW))
        peak = peak_admission(outcomes, WINDOW)
        results.append(
            {
                "algorithm": algorithm,
                "allowed": sum(1 for o in outcomes if o.allowed),
                "peak": peak,
                "ratio": round(peak / LIMIT, 3),
                "shapes": any(o.delay > 0 for o in outcomes),
                # Compact keys: 90 requests x 5 algorithms is 450 of these, and
                # the whole thing crosses the wire on every slider drag.
                "marks": [
                    {"t": o.offset, "ok": o.allowed, "d": round(o.delay, 4)}
                    for o in outcomes
                ],
            }
        )

    return {
        "limit": LIMIT,
        "window": WINDOW,
        "center": center,
        "offered": len(schedule),
        "boundaries": [WINDOW * i for i in range(1, int(CONTROL_AT / WINDOW) + 2)],
        "results": results,
    }
