"""HTTP-layer behaviour: the middleware, the decorator, and the headers."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI, Request

from distributed_rate_limiter.keys import client_ip_key, forwarded_for_key, path_scoped
from distributed_rate_limiter.memory.token_bucket import InMemoryTokenBucket
from distributed_rate_limiter.middleware import (
    RateLimitMiddleware,
    rate_limit,
    retry_after_seconds,
)
from distributed_rate_limiter.registry import ALGORITHMS, create_limiter
from tests.conftest import FakeClock


def client_for(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


# --------------------------------------------------------------------------
# Retry-After conversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, 1),  # never tell a client to retry immediately
        (0.001, 1),
        (0.3, 1),  # round up: 0 would earn an instant second 429
        (1.0, 1),
        (1.2, 2),
        (59.4, 60),
    ],
)
async def test_retry_after_rounds_up_and_never_zero(seconds, expected):
    assert retry_after_seconds(seconds) == expected


# --------------------------------------------------------------------------
# Decorator
# --------------------------------------------------------------------------


async def test_decorator_allows_then_returns_429_with_retry_after():
    app = FastAPI()

    @app.get("/limited")
    @rate_limit(algorithm="fixed_window", limit=3, window=60.0)
    async def limited() -> dict:
        return {"ok": True}

    async with client_for(app) as client:
        for i in range(3):
            response = await client.get("/limited")
            assert response.status_code == 200, f"request {i + 1}"
            assert response.json() == {"ok": True}
            assert response.headers["X-RateLimit-Limit"] == "3"
            assert response.headers["X-RateLimit-Remaining"] == str(2 - i)

        denied = await client.get("/limited")
        assert denied.status_code == 429
        assert denied.headers["X-RateLimit-Remaining"] == "0"
        assert int(denied.headers["Retry-After"]) >= 1


async def test_decorator_leaves_handler_signature_usable():
    """Injecting Request/Response must not disturb the handler's own params."""
    app = FastAPI()

    @app.get("/items/{item_id}")
    @rate_limit(limit=10, window=60.0)
    async def read_item(item_id: int, q: str | None = None) -> dict:
        return {"item_id": item_id, "q": q}

    async with client_for(app) as client:
        response = await client.get("/items/42?q=hello")
        assert response.status_code == 200
        assert response.json() == {"item_id": 42, "q": "hello"}


async def test_decorator_passes_through_a_declared_request_param():
    """A handler that already takes a Request keeps receiving the real one."""
    app = FastAPI()

    @app.get("/echo")
    @rate_limit(limit=10, window=60.0)
    async def echo(request: Request) -> dict:
        return {"path": request.url.path}

    async with client_for(app) as client:
        response = await client.get("/echo")
        assert response.status_code == 200
        assert response.json() == {"path": "/echo"}


async def test_each_endpoint_has_independent_state():
    app = FastAPI()

    @app.get("/cheap")
    @rate_limit(algorithm="fixed_window", limit=5, window=60.0)
    async def cheap() -> dict:
        return {"endpoint": "cheap"}

    @app.get("/expensive")
    @rate_limit(algorithm="fixed_window", limit=1, window=60.0)
    async def expensive() -> dict:
        return {"endpoint": "expensive"}

    async with client_for(app) as client:
        assert (await client.get("/expensive")).status_code == 200
        assert (await client.get("/expensive")).status_code == 429
        # The cheap endpoint's budget is untouched by that.
        for _ in range(5):
            assert (await client.get("/cheap")).status_code == 200


async def test_rejects_sync_endpoints_at_decoration_time():
    with pytest.raises(TypeError, match="async endpoint"):

        @rate_limit(limit=5, window=60.0)
        def not_async() -> dict:  # pragma: no cover - never called
            return {}


async def test_unknown_algorithm_names_the_valid_options():
    with pytest.raises(ValueError, match="token_bucket"):
        create_limiter("tokenbucket", limit=5, window=60.0)


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
async def test_every_algorithm_works_over_http(algorithm):
    """Each registered algorithm must survive the HTTP path, not just unit tests.

    Asserts only what is true of all five: at most `limit` succeed, the excess
    gets a 429 with a usable Retry-After, and nothing 500s. Exact admission
    counts are algorithm-specific and belong in the per-algorithm files.

    The leaky bucket gets max_delay=0 so it rejects rather than holding the
    connection: at 2 per 60s its shaping delay would be 30 real seconds.
    """
    options = {"max_delay": 0.0} if algorithm == "leaky_bucket" else {}
    app = FastAPI()

    @app.get("/limited")
    @rate_limit(algorithm=algorithm, limit=2, window=60.0, **options)
    async def limited() -> dict:
        return {"ok": True}

    async with client_for(app) as client:
        responses = [await client.get("/limited") for _ in range(4)]

    codes = [r.status_code for r in responses]
    assert set(codes) <= {200, 429}, f"{algorithm} returned {codes}"
    assert codes.count(200) <= 2, f"{algorithm} allowed more than the limit: {codes}"
    assert 429 in codes, f"{algorithm} never rejected: {codes}"

    denied = next(r for r in responses if r.status_code == 429)
    assert int(denied.headers["Retry-After"]) >= 1


# --------------------------------------------------------------------------
# Shaping
# --------------------------------------------------------------------------


async def test_leaky_bucket_delays_the_response_instead_of_rejecting():
    """The shaper must actually hold the request, not just report a delay.

    Rate chosen so pacing is observable without making the suite slow:
    capacity 3 draining at 20/s paces requests 0.05s apart. Three requests
    therefore span two gaps -- 0.1s, not 0.15s.
    """
    app = FastAPI()

    @app.get("/shaped")
    @rate_limit(algorithm="leaky_bucket", limit=3, window=0.15)
    async def shaped() -> dict:
        return {"ok": True}

    async with client_for(app) as client:
        started = time.perf_counter()
        responses = [await client.get("/shaped") for _ in range(3)]
        elapsed = time.perf_counter() - started

    assert all(r.status_code == 200 for r in responses)
    assert elapsed >= 0.09, f"requests were not paced (took {elapsed:.3f}s)"
    assert elapsed < 1.0, f"pacing overshot badly (took {elapsed:.3f}s)"


async def test_metering_algorithms_do_not_delay():
    app = FastAPI()

    @app.get("/metered")
    @rate_limit(algorithm="token_bucket", limit=3, window=0.15)
    async def metered() -> dict:
        return {"ok": True}

    async with client_for(app) as client:
        started = time.perf_counter()
        for _ in range(3):
            assert (await client.get("/metered")).status_code == 200
        elapsed = time.perf_counter() - started

    assert elapsed < 0.05, f"token bucket should not pace requests (took {elapsed:.3f}s)"


# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------


async def test_middleware_limits_every_route():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, algorithm="fixed_window", limit=2, window=60.0)

    @app.get("/a")
    async def a() -> dict:
        return {"route": "a"}

    @app.get("/b")
    async def b() -> dict:
        return {"route": "b"}

    async with client_for(app) as client:
        assert (await client.get("/a")).status_code == 200
        assert (await client.get("/b")).status_code == 200
        # Shared budget: the third request loses regardless of route.
        assert (await client.get("/a")).status_code == 429


async def test_middleware_adds_headers_to_successful_responses():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, algorithm="fixed_window", limit=5, window=60.0)

    @app.get("/a")
    async def a() -> dict:
        return {"route": "a"}

    async with client_for(app) as client:
        response = await client.get("/a")

    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "5"
    assert response.headers["X-RateLimit-Remaining"] == "4"
    assert "X-RateLimit-Reset" in response.headers


async def test_endpoint_headers_win_over_middleware_headers():
    """A client must not receive two contradictory X-RateLimit-Limit values.

    With both layers active the endpoint's limit is the one actually binding,
    so the app-wide middleware must not append its own numbers alongside.
    """
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, algorithm="token_bucket", limit=1000, window=60.0)

    @app.get("/limited")
    @rate_limit(algorithm="fixed_window", limit=5, window=60.0)
    async def limited() -> dict:
        return {"ok": True}

    async with client_for(app) as client:
        response = await client.get("/limited")

    assert response.status_code == 200
    assert response.headers.get_list("X-RateLimit-Limit") == ["5"]
    assert response.headers.get_list("X-RateLimit-Remaining") == ["4"]


async def test_middleware_exempts_docs_and_health():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, algorithm="fixed_window", limit=1, window=60.0)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    async with client_for(app) as client:
        # Exhaust the budget on a normal route first.
        assert (await client.get("/health")).status_code == 200
        for _ in range(5):
            assert (await client.get("/health")).status_code == 200, "health must never throttle"
        assert (await client.get("/openapi.json")).status_code == 200


async def test_path_scoped_keys_separate_endpoints():
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        algorithm="fixed_window",
        limit=1,
        window=60.0,
        key_func=path_scoped(client_ip_key),
    )

    @app.get("/a")
    async def a() -> dict:
        return {"route": "a"}

    @app.get("/b")
    async def b() -> dict:
        return {"route": "b"}

    async with client_for(app) as client:
        assert (await client.get("/a")).status_code == 200
        assert (await client.get("/a")).status_code == 429
        assert (await client.get("/b")).status_code == 200, "separate path, separate budget"


async def test_middleware_shares_one_limiter_across_requests(clock: FakeClock):
    """State must persist between requests, and honour an injected clock."""
    app = FastAPI()
    limiter = InMemoryTokenBucket.from_limit_window(limit=2, window=60.0, clock=clock)
    app.add_middleware(RateLimitMiddleware, limiter=limiter)

    @app.get("/a")
    async def a() -> dict:
        return {"route": "a"}

    async with client_for(app) as client:
        assert (await client.get("/a")).status_code == 200
        assert (await client.get("/a")).status_code == 200
        assert (await client.get("/a")).status_code == 429

        clock.advance(60.0)  # a full refill
        assert (await client.get("/a")).status_code == 200


# --------------------------------------------------------------------------
# Key extraction
# --------------------------------------------------------------------------


async def test_default_key_ignores_forwarded_for():
    """A forged X-Forwarded-For must not buy a fresh budget.

    If the default trusted this header, any client could send a random value
    per request and never be limited at all.
    """
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, algorithm="fixed_window", limit=1, window=60.0)

    @app.get("/a")
    async def a() -> dict:
        return {"route": "a"}

    async with client_for(app) as client:
        assert (await client.get("/a")).status_code == 200
        forged = await client.get("/a", headers={"X-Forwarded-For": "10.0.0.99"})
        assert forged.status_code == 429, "forged header must not reset the limit"


async def test_forwarded_for_key_reads_the_trusted_hop():
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        algorithm="fixed_window",
        limit=1,
        window=60.0,
        key_func=forwarded_for_key(trusted_hops=1),
    )

    @app.get("/a")
    async def a() -> dict:
        return {"route": "a"}

    async with client_for(app) as client:
        # Last entry is what our own proxy appended, so these are two clients.
        first = await client.get("/a", headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"})
        second = await client.get("/a", headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.2"})
        assert first.status_code == 200
        assert second.status_code == 200

        repeat = await client.get("/a", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})
        assert repeat.status_code == 429, "same trusted hop is the same client"


async def test_forwarded_for_rejects_nonsense_hop_counts():
    with pytest.raises(ValueError):
        forwarded_for_key(trusted_hops=0)


async def test_concurrent_requests_do_not_exceed_the_limit():
    """The in-process baseline for Milestone 3's real concurrency tests.

    Twenty simultaneous requests against a limit of 5. This passes here only
    because the in-memory limiters mutate their state without awaiting, so the
    event loop cannot interleave a check. Redis has no such guarantee, which
    is exactly what the Lua scripting work has to solve.
    """
    app = FastAPI()

    @app.get("/limited")
    @rate_limit(algorithm="fixed_window", limit=5, window=60.0)
    async def limited() -> dict:
        return {"ok": True}

    async with client_for(app) as client:
        responses = await asyncio.gather(*(client.get("/limited") for _ in range(20)))

    allowed = sum(1 for r in responses if r.status_code == 200)
    assert allowed == 5, f"expected exactly 5 allowed under concurrency, got {allowed}"
