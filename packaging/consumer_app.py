"""A FastAPI application that knows drl-ratelimit only as an installed package.

Run by `verify_install.sh` from a temporary directory outside this repository,
in a virtualenv containing nothing but the built wheel and its extras. That
isolation is the point: an editable install papers over missing subpackages,
a missing `py.typed`, and wrong `packages` configuration, because the source
tree is on the path either way. This file can only see what the wheel shipped.

Two limiters, because they cover the two integration paths an application
actually has: app-wide middleware, and a per-endpoint decorator.
"""

import os

import redis.asyncio as redis
from fastapi import FastAPI

from distributed_rate_limiter import BACKENDS, __version__
from distributed_rate_limiter.keys import client_ip_key, path_scoped
from distributed_rate_limiter.middleware import (
    DEFAULT_EXEMPT_PATHS,
    RateLimitMiddleware,
    rate_limit,
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/13")
client = redis.from_url(REDIS_URL)

app = FastAPI(title=f"consumer of drl-ratelimit {__version__}")

# App-wide policy held in Redis, so every process running this app enforces
# one limit between them rather than one limit each.
app.add_middleware(
    RateLimitMiddleware,
    algorithm="token_bucket",
    backend="redis",
    client=client,
    limit=50,
    window=10.0,
    prefix="consumer:global",
    # /version reports which package got imported, which is metadata rather
    # than traffic. Leaving it metered would spend a request from the very
    # quota the next check is trying to measure.
    exempt_paths=(*DEFAULT_EXEMPT_PATHS, "/version"),
)


@app.get("/cheap")
async def cheap():
    return {"ok": True}


@app.get("/expensive")
@rate_limit(
    "sliding_window_log",
    limit=5,
    window=10.0,
    backend="redis",
    client=client,
    key_func=path_scoped(client_ip_key),
    prefix="consumer:expensive",
)
async def expensive():
    return {"ok": True, "cost": "high"}


@app.get("/version")
async def version():
    import distributed_rate_limiter

    # `loaded_from` is the assertion that matters: it must be inside
    # site-packages, never the repository's src/ tree.
    return {
        "version": __version__,
        "loaded_from": distributed_rate_limiter.__file__,
        "algorithms": sorted(BACKENDS["redis"]),
    }
