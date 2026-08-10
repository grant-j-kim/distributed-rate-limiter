"""A runnable demo server exercising every algorithm.

    PYTHONPATH=src uvicorn examples.app:app --reload

PYTHONPATH is belt-and-braces: this project lives in an iCloud-synced folder
(~/Desktop), and iCloud sets the macOS UF_HIDDEN flag on files inside .venv.
Python 3.13 silently skips hidden .pth files, so the editable install's path
entry disappears and the package stops importing with no diagnostic. Setting
PYTHONPATH sidesteps the .pth entirely. The lasting fix is to keep the
project (or at least its virtualenv) outside iCloud.

Then try:

    curl -i localhost:8000/token-bucket        # 5 fast, then 429s
    curl -i localhost:8000/fixed-window
    curl -i localhost:8000/leaky-bucket        # admitted, but paced
    for i in $(seq 1 8); do curl -s -o /dev/null -w "%{http_code} " \
        localhost:8000/token-bucket; done; echo

Interactive docs (exempt from limiting) are at /docs.
"""

from __future__ import annotations

import time

from fastapi import FastAPI

from distributed_rate_limiter.keys import path_scoped, client_ip_key
from distributed_rate_limiter.middleware import RateLimitMiddleware, rate_limit

app = FastAPI(
    title="Distributed Rate Limiter demo",
    description="Every endpoint below is limited by a different algorithm.",
)

# App-wide backstop. Generous, so the per-endpoint limits below are what you
# actually hit. Keys are scoped by path so one endpoint's traffic does not
# consume another's global allowance.
app.add_middleware(
    RateLimitMiddleware,
    algorithm="token_bucket",
    limit=1000,
    window=60.0,
    key_func=path_scoped(client_ip_key),
)


@app.get("/")
async def index() -> dict:
    """Not rate limited beyond the app-wide backstop."""
    return {
        "try": [
            "/fixed-window",
            "/sliding-window-log",
            "/sliding-window-counter",
            "/token-bucket",
            "/leaky-bucket",
        ],
        "hint": "send 6+ requests quickly and watch for 429 + Retry-After",
    }


@app.get("/health")
async def health() -> dict:
    """Exempt from the middleware -- health checks must not be throttled."""
    return {"status": "ok"}


@app.get("/fixed-window")
@rate_limit(algorithm="fixed_window", limit=5, window=30.0)
async def fixed_window() -> dict:
    return {"algorithm": "fixed_window", "note": "resets hard on a 30s boundary"}


@app.get("/sliding-window-log")
@rate_limit(algorithm="sliding_window_log", limit=5, window=30.0)
async def sliding_window_log() -> dict:
    return {"algorithm": "sliding_window_log", "note": "exact trailing 30s window"}


@app.get("/sliding-window-counter")
@rate_limit(algorithm="sliding_window_counter", limit=5, window=30.0)
async def sliding_window_counter() -> dict:
    return {"algorithm": "sliding_window_counter", "note": "approximate, O(1) state"}


@app.get("/token-bucket")
@rate_limit(algorithm="token_bucket", limit=5, window=30.0)
async def token_bucket() -> dict:
    return {"algorithm": "token_bucket", "note": "burst of 5, then 1 per 6s"}


@app.get("/leaky-bucket")
@rate_limit(algorithm="leaky_bucket", limit=5, window=5.0, max_delay=3.0)
async def leaky_bucket() -> dict:
    """Admitted requests are *held* until their turn -- watch the latency.

    max_delay is the queue timeout: a request that would wait longer than 3s
    gets a 429 instead of tying up a connection nobody is still waiting on.
    """
    return {
        "algorithm": "leaky_bucket",
        "note": "requests are queued and paced, not rejected",
        "served_at": time.strftime("%H:%M:%S.") + f"{int(time.time() % 1 * 1000):03d}",
    }
