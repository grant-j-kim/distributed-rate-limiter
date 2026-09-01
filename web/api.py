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

import inspect
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from distributed_rate_limiter.keys import forwarded_for_key

from loadtest.traffic import burst, merge
from web import race
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


# ---------------------------------------------------------------- punchline

_RECORDING = Path(__file__).resolve().parent / "punchline_recording.json"


def _recording() -> dict | None:
    """The last real run, replayed when a live one is not available.

    Serving a recording is the graceful degradation; serving it *silently* is
    not. Every response that carries one says so, and the file carries the
    timestamp and Redis version it was produced against, so the page can show
    where the numbers came from instead of implying they happened just now.
    """
    if not _RECORDING.exists():
        return None
    return json.loads(_RECORDING.read_text(encoding="utf-8"))


@app.post("/api/race/start")
async def race_start(request: Request) -> dict:
    """Charge one run against the rationing, or hand back the recording."""
    shape = {
        "concurrency": race.CONCURRENCY,
        "limit": race.RACE_LIMIT,
        "monthly_budget": race.MONTHLY_RUNS,
        "per_ip": [race.PER_IP_RUNS, race.PER_IP_WINDOW],
    }

    if race.redis_url() is None:
        # No live Redis at all -- a preview deployment, or local development
        # without one. Not an error, and not something to hide.
        return {"live": False, "reason": "no-redis", "recording": _recording(), **shape}

    client = race._client()
    try:
        rationing = await race.check_rationing(client, forwarded_for_key(1)(request))
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        # An unreachable Redis is information. A page about correctness that
        # quietly swaps in a recording when the real thing fails is the one
        # failure mode worth refusing outright.
        return {"live": False, "reason": "error", "detail": f"{type(exc).__name__}: {exc}",
                "recording": _recording(), **shape}
    finally:
        await client.aclose()

    if not rationing.allowed:
        return {"live": False, "reason": rationing.reason,
                "retry_after": rationing.retry_after,
                "recording": _recording(), **shape}

    run_id, expires_at, token = race.new_run()
    return {"live": True, "run": run_id, "exp": expires_at, "token": token, **shape}


@app.get("/api/race/fire")
async def race_fire(
    run: str,
    exp: float,
    token: str,
    variant: Literal["naive", "lua"],
) -> dict:
    """One request of one race. The browser calls this `CONCURRENCY` times at once.

    Deliberately one check per invocation: that is what makes the requests
    genuinely concurrent rather than a loop wearing a costume.
    """
    if race.redis_url() is None:
        raise HTTPException(status_code=409, detail="no live Redis configured")
    if not race.verify_token(run, exp, token):
        raise HTTPException(status_code=403, detail="invalid or expired run token")

    client = race._client()
    try:
        limiter = race.build_limiter(client, run, variant)
        decision = await limiter.check("race")
    finally:
        await client.aclose()

    return {"allowed": decision.allowed, "remaining": decision.remaining}


@app.get("/api/race/code")
async def race_code() -> dict:
    """The two implementations, read from the running source.

    Read with `inspect` rather than pasted into the page, so the code a visitor
    is shown is necessarily the code that just ran.
    """
    from distributed_rate_limiter.redis_backend.fixed_window import CHECK_SCRIPT

    return {
        "naive": inspect.getsource(race.NaiveRedisFixedWindow.check),
        "lua": CHECK_SCRIPT.strip(),
    }
