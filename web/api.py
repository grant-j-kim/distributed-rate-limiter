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

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from loadtest.traffic import burst, merge
from web.replay import build, peak_admission, replay

LIMIT = 20
WINDOW = 2.0
BURST_SIZE = 30
HALF_GAP = 0.15
"""Half the separation between the paired bursts. They sit at `center` plus and
minus this, so the pair spans 0.30s however the slider moves it."""

CENTER_MAX = 3.0
"""How far the slider travels. 0.5 is mid-window, 2.0 is a boundary, 3.0 is
mid-window again -- the whole instructive range. Going further only repeats a
boundary, and would push the leaky bucket's releases off the right edge."""

DURATION = 5.5
"""The visible time axis. Fixed rather than fitted to each frame: an axis that
rescaled as the slider moved would make two positions incomparable, which is
the one thing this page exists to allow.

Sized by the leaky bucket, not by the arrivals. Everything else has finished by
`CENTER_MAX + HALF_GAP` = 3.15s, but the shaper is still releasing requests at
5.05s, and clipping those would hide exactly the behaviour that distinguishes
it."""

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
    """Two bursts, `HALF_GAP` either side of `center`.

    Sliding `center` onto a window boundary is the entire experiment: the same
    sixty requests that one algorithm spreads across two windows, another
    counts once.

    There is no third mid-window control burst, unlike the `boundary_burst`
    scenario in `loadtest/`. There it was the only way to show that the
    disagreement is a boundary phenomenon rather than a general one; here the
    slider shows that far better, by letting the visitor walk the same burst
    off the boundary and watch the gap close.
    """
    return merge(
        burst(center - HALF_GAP, BURST_SIZE, client="client-a"),
        burst(center + HALF_GAP, BURST_SIZE, client="client-a"),
    )


@app.get("/api/replay")
async def api_replay(
    center: float = Query(
        WINDOW,
        ge=HALF_GAP,
        le=CENTER_MAX,
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
        "duration": DURATION,
        "boundaries": [WINDOW * i for i in range(1, int(DURATION / WINDOW) + 1)],
        "results": results,
    }


_PAGE = Path(__file__).resolve().parent.parent / "public" / "index.html"
"""The page lives in `public/`, which Vercel serves from its CDN at
/index.html. It is served from the function at `/` as well, rather than
mounted: Vercel handles `public/` at the platform level and mounting it is
explicitly unsupported. Reading the file keeps one copy of the page and makes
local uvicorn behave exactly like the deployment."""


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))
