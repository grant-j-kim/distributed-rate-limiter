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

import asyncio
import inspect
import json
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from distributed_rate_limiter.keys import forwarded_for_key
from distributed_rate_limiter.middleware import rate_limit

from loadtest.runner import redis_now
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

_INSTANCE = uuid.uuid4().hex[:8]
"""Generated once when this module is imported, so it identifies the *instance*
rather than the request. Counting distinct values across a volley is the only
direct way to see whether the platform fanned the requests out or ran them one
after another on a single warm instance."""

_IMPORTED_AT = time.time()


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


def _limited(func):
    """Apply this project's own per-endpoint limiters to the race.

    The site used to call `create_limiter` by hand here. It now uses
    `@rate_limit`, the public decorator a user of this package would actually
    write, so the demo runs the interface it documents rather than a private
    shortcut past it. Two of them stack: the per-IP one runs first, because
    rejecting one greedy visitor should not cost a slot in the global budget.

    Per endpoint rather than app-wide middleware on purpose. Vercel stops
    promoting static files to its CDN once top-level middleware exists, which
    would push `public/` back through the function on every request -- and the
    playground has nothing to protect anyway, since it touches no Redis.

    Backed by Redis rather than memory, and that is not a preference. This
    deployment was measured serving one volley across **49 instances**. An
    in-memory limiter would have given each visitor 49 independent counters and
    49x their quota, while passing every test you would think to write for it.
    The naive race's lesson, one level up.
    """
    func = rate_limit(
        limiter=race.quota_limiter(race.MONTHLY_RUNS, race.MONTHLY_WINDOW,
                                   "race:quota:global"),
        # One shared key: a per-key limit bounds each visitor and says nothing
        # about the total, and it is the total the free tier cares about.
        key_func=lambda request: "all",
    )(func)
    return rate_limit(
        limiter=race.quota_limiter(race.PER_IP_RUNS, race.PER_IP_WINDOW,
                                   "race:quota:ip"),
        key_func=forwarded_for_key(1),
    )(func)


async def _mint_run() -> dict:
    """A signed barrier token for one authorised run."""
    shape = {
        "concurrency": race.CONCURRENCY,
        "limit": race.RACE_LIMIT,
        "monthly_budget": race.MONTHLY_RUNS,
        "per_ip": [race.PER_IP_RUNS, race.PER_IP_WINDOW],
    }
    # The barrier instant lives on Redis's clock, so every fire -- wherever it
    # lands -- computes the same target and they arrive together.
    start_at = await redis_now(race._client()) + race.LEAD
    run_id, expires_at, token = race.new_run(start_at)
    return {"live": True, "run": run_id, "exp": expires_at, "token": token,
            "start": start_at, "lead": race.LEAD, "phase": race.PHASE, **shape}


@app.get("/api/race/recording")
async def race_recording() -> dict:
    """The last real run. Free, unrationed, and served when a live run is not.

    A separate endpoint because the limiter now answers a spent quota with a
    real 429 and Retry-After -- the library's actual behaviour, which is worth
    showing -- rather than a 200 carrying a consolation prize.
    """
    return {"live": False, "recording": _recording(),
            "concurrency": race.CONCURRENCY, "limit": race.RACE_LIMIT,
            "monthly_budget": race.MONTHLY_RUNS,
            "per_ip": [race.PER_IP_RUNS, race.PER_IP_WINDOW]}


def _self_base_url(request: Request) -> str:
    """This deployment's own origin, for fanning out to.

    Built from the forwarded headers rather than `request.base_url`, which
    behind Vercel's proxy reports the internal scheme and would produce an
    unreachable http:// URL for a site served over https.
    """
    host = request.headers.get("host") or request.url.netloc
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"


@app.post("/api/race/run")
@_limited
async def race_run(request: Request) -> dict:
    """Run both volleys by fanning out from the server, and report what happened.

    The browser cannot stage this. Measured against the deployed site, fifty
    `fetch` calls issued at once arrived at the server 110ms apart, one at a
    time, all on a single warm instance -- one network round trip each, so
    nothing ever overlapped and no barrier could assemble a volley.

    Issuing the fifty from here instead means they leave together and arrive
    together. Whether the platform then spreads them across instances is not
    something to assert: every fire reports the instance that served it, and
    the count comes back in the response. More than one is a genuinely
    distributed race; exactly one is fifty concurrent requests interleaving on
    a single event loop, which opens the same read-modify-write gap and is
    still a real race -- just a narrower claim, and the number is what makes
    the page say the narrower thing.
    """
    import httpx

    if race.redis_url() is None:
        return await race_recording()

    authorised = await _mint_run()

    query = {"run": authorised["run"], "exp": authorised["exp"],
             "start": authorised["start"], "token": authorised["token"]}
    base = _self_base_url(request)
    variants: dict[str, dict] = {}

    limits = httpx.Limits(max_connections=race.CONCURRENCY + 10)
    async with httpx.AsyncClient(base_url=base, timeout=60, limits=limits) as client:
        for variant in ("naive", "lua"):
            responses = await asyncio.gather(*(
                client.get("/api/race/fire", params={**query, "variant": variant})
                for _ in range(race.CONCURRENCY)), return_exceptions=True)

            fires = [r.json() for r in responses
                     if not isinstance(r, Exception) and r.status_code == 200]
            failed = race.CONCURRENCY - len(fires)
            stamps = sorted(f["t"] for f in fires)
            rtts = sorted(f["rtt_ms"] for f in fires)

            variants[variant] = {
                "allowed": [f["allowed"] for f in fires],
                "admitted": sum(1 for f in fires if f["allowed"]),
                "fired": len(fires),
                "failed": failed,
                "late": sum(1 for f in fires if f.get("late")),
                "instances": len({f["instance"] for f in fires}),
                "spread_ms": round((stamps[-1] - stamps[0]) * 1000, 2) if stamps else 0.0,
                "rtt_ms": rtts[len(rtts) // 2] if rtts else 0.0,
            }

    return {"live": True, "variants": variants,
            "redis_version": await race.redis_version(race._client()),
            **{k: v for k, v in authorised.items()
               if k in ("concurrency", "limit", "monthly_budget", "per_ip")}}


@app.get("/api/race/fire")
async def race_fire(
    run: str,
    exp: float,
    start: float,
    token: str,
    variant: Literal["naive", "lua"],
) -> dict:
    """One request of one race. The browser calls this `CONCURRENCY` times at once.

    Deliberately one check per invocation: that is what makes the requests
    genuinely concurrent rather than a loop wearing a costume.
    """
    if race.redis_url() is None:
        raise HTTPException(status_code=409, detail="no live Redis configured")
    if not race.verify_token(run, exp, token, start):
        raise HTTPException(status_code=403, detail="invalid or expired run token")

    client = race._client()
    # Read the shared clock once. The fifty fires may land on different
    # instances, and comparing their local clocks at the millisecond scale --
    # exactly the scale that decides whether a race happens -- would measure
    # skew as if it were stagger. Same reasoning that makes the Redis limiters
    # call TIME rather than take a clock argument.
    anchor = await redis_now(client)
    mark = time.perf_counter()

    # Wait for the barrier. This is what turns fifty independently scheduled
    # invocations into fifty simultaneous requests: whatever stagger Vercel
    # introduced is absorbed here instead of landing in the measurement.
    delay = race.target_for(start, variant) - anchor
    late = delay < 0
    if not late:
        await asyncio.sleep(delay)

    # This request's instant on the Redis timeline, without spending a second
    # TIME: anchored on the shared reading, advanced by locally measured
    # elapsed. A duration is process-local, so perf_counter is safe for it
    # even though a timestamp would not be.
    arrival = anchor + (time.perf_counter() - mark)

    # The naive limiter is handed that same reading, so the only difference
    # between the two variants is atomicity, not which clock they trust.
    limiter = race.build_limiter(client, run, variant, arrival)
    elapsed = time.perf_counter()
    decision = await limiter.check("race")
    elapsed = (time.perf_counter() - elapsed) * 1000

    return {
        "allowed": decision.allowed,
        "remaining": decision.remaining,
        "t": arrival,
        "rtt_ms": round(elapsed, 3),
        # Reported, not hidden: an invocation that started after the barrier
        # never joined the volley, and a run full of stragglers is a run whose
        # result means less.
        "late": late,
        # Which instance served this fire. Counting distinct values across a
        # volley is what lets the page claim "across instances" only when it
        # is true.
        "instance": _INSTANCE,
    }


@app.get("/api/race/ping")
async def race_ping() -> dict:
    """A concurrency probe with no Redis, no rationing and no cost.

    Two attempts to make the race overlap have now failed for reasons I
    guessed at and got wrong. This measures the platform itself instead: fire
    it N times at once and count distinct `instance` values. If they are all
    the same, the requests were serialised on one instance and no barrier will
    ever assemble a volley; if they differ, the fan-out works and the problem
    is timing.

    `age` distinguishes a cold start from a warm reuse, which is the other
    thing worth knowing and is invisible from the outside.
    """
    return {"instance": _INSTANCE, "t": time.time(),
            "age": round(time.time() - _IMPORTED_AT, 3)}


@app.get("/api/race/code")
async def race_code() -> dict:
    """The two implementations, read from the running source.

    Read with `inspect` rather than pasted into the page, so the code a visitor
    is shown is necessarily the code that just ran.
    """
    from distributed_rate_limiter.redis_backend.sliding_window_log import CHECK_SCRIPT

    return {
        "naive": inspect.getsource(race.NaiveRedisSlidingWindowLog.check),
        "lua": CHECK_SCRIPT.strip(),
    }
