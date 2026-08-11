"""FastAPI/Starlette integration: app-wide middleware and a per-endpoint decorator.

Two pieces, because one cannot do both jobs. ASGI middleware runs *before*
routing resolves, so at that point there is no way to know which endpoint the
request will reach -- `scope["route"]` is not populated yet. App-wide policy
therefore lives in middleware; per-endpoint policy has to attach at the
endpoint itself.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import math
from typing import Any, Awaitable, Callable, Iterable

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from distributed_rate_limiter.base import Clock, Decision, RateLimiter
from distributed_rate_limiter.keys import KeyFunc, client_ip_key

# The sentinel travels with the clock rather than being resolved here: what a
# missing clock *means* depends on the backend (local wall clock for memory,
# the server's own TIME for Redis), and only create_limiter knows which
# backend was asked for. Defaulting to default_clock at this layer would hand
# every Redis limiter the local process clock and desync the instances.
from distributed_rate_limiter.registry import _UNSET, create_limiter

DEFAULT_EXEMPT_PATHS = ("/docs", "/redoc", "/openapi.json", "/health")


def retry_after_seconds(seconds: float) -> int:
    """Convert a float wait into a Retry-After header value.

    Rounded *up*, and never below 1. Rounding down would tell the client to
    retry before capacity exists, earning it an immediate second 429; a value
    of 0 would say "retry now", which is worse. The HTTP spec wants whole
    seconds, so the precision loss is unavoidable -- it just has to land on
    the safe side.
    """
    return max(1, math.ceil(seconds))


def rate_limit_headers(decision: Decision) -> dict[str, str]:
    """Standard advisory headers describing the client's remaining quota."""
    headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(retry_after_seconds(decision.reset_after)),
    }
    if decision.retry_after is not None:
        headers["Retry-After"] = str(retry_after_seconds(decision.retry_after))
    return headers


def too_many_requests(decision: Decision) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too Many Requests",
            "limit": decision.limit,
            "retry_after": decision.retry_after,
        },
        headers=rate_limit_headers(decision),
    )


class RateLimitMiddleware:
    """App-wide rate limiting for every request that is not exempt.

        app.add_middleware(RateLimitMiddleware, limit=100, window=60)

        app.add_middleware(                       # shared across instances
            RateLimitMiddleware,
            algorithm="token_bucket", backend="redis", client=redis_client,
            limit=100, window=60,
        )

    Written as raw ASGI rather than BaseHTTPMiddleware: BaseHTTPMiddleware
    wraps each request in an anyio task group and buffers the response, which
    adds per-request overhead for nothing this limiter needs.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        algorithm: str = "token_bucket",
        backend: str = "memory",
        limit: int = 100,
        window: float = 60.0,
        key_func: KeyFunc = client_ip_key,
        exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
        clock: Clock | None | Any = _UNSET,
        limiter: RateLimiter | None = None,
        **limiter_options: object,
    ):
        self.app = app
        self.limiter = limiter or create_limiter(
            algorithm, limit, window, backend=backend, clock=clock, **limiter_options
        )
        self.key_func = key_func
        self.exempt_paths = tuple(exempt_paths)

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope.get("path", "").startswith(self.exempt_paths):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        decision = await self.limiter.check(self.key_func(request))

        if not decision.allowed:
            await too_many_requests(decision)(scope, receive, send)
            return

        # Shaping limiters admit the request but hold it back; skipping this
        # would silently turn the leaky bucket into a plain meter.
        if decision.delay > 0:
            await asyncio.sleep(decision.delay)

        await self._send_with_headers(decision, scope, receive, send)

    async def _send_with_headers(
        self, decision: Decision, scope: dict, receive: Callable, send: Callable
    ) -> None:
        headers = {
            k.lower().encode(): v.encode() for k, v in rate_limit_headers(decision).items()
        }

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                existing = list(message.get("headers", []))
                # A @rate_limit endpoint has already described its own, tighter
                # quota. Appending the app-wide numbers on top would leave the
                # client with two contradictory X-RateLimit-Limit values and no
                # way to tell which one it is actually subject to.
                already_set = {name.lower() for name, _ in existing}
                message = dict(message)
                message["headers"] = existing + [
                    (name, value)
                    for name, value in headers.items()
                    if name not in already_set
                ]
            await send(message)

        await self.app(scope, receive, send_wrapper)


def rate_limit(
    algorithm: str = "token_bucket",
    limit: int = 100,
    window: float = 60.0,
    *,
    backend: str = "memory",
    key_func: KeyFunc = client_ip_key,
    clock: Clock | None | Any = _UNSET,
    limiter: RateLimiter | None = None,
    **limiter_options: object,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Per-endpoint rate limiting.

        @app.get("/search")
        @rate_limit(algorithm="token_bucket", limit=100, window=60)
        async def search():
            ...

        @app.get("/expensive")
        @rate_limit("sliding_window_log", 10, 60,
                    backend="redis", client=redis_client)
        async def expensive():
            ...

    The limiter is built once at decoration time, so all requests to the
    endpoint share its state.

    FastAPI decides what to inject by inspecting the endpoint's signature, so
    when the handler does not already declare Request/Response parameters this
    appends them to the wrapper's `__signature__` and strips them back out
    before calling the handler. That keeps the decorated function's own
    signature untouched from the caller's point of view.
    """
    endpoint_limiter = limiter or create_limiter(
        algorithm, limit, window, backend=backend, clock=clock, **limiter_options
    )

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@rate_limit requires an async endpoint; {func.__name__} is a plain function"
            )

        # eval_str resolves string annotations, which `from __future__ import
        # annotations` makes the norm. Without it every annotation is a bare
        # string, a handler declaring `request: Request` is never recognised,
        # and a duplicate parameter gets injected over the top of it.
        try:
            signature = inspect.signature(func, eval_str=True)
        except (NameError, TypeError):
            signature = inspect.signature(func)
        params = list(signature.parameters.values())

        request_param = _find_param(params, Request)
        response_param = _find_param(params, Response)

        injected: list[inspect.Parameter] = []
        if request_param is None:
            request_param = "_rate_limit_request"
            injected.append(
                inspect.Parameter(
                    request_param, inspect.Parameter.KEYWORD_ONLY, annotation=Request
                )
            )
        if response_param is None:
            response_param = "_rate_limit_response"
            injected.append(
                inspect.Parameter(
                    response_param, inspect.Parameter.KEYWORD_ONLY, annotation=Response
                )
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Request = kwargs[request_param]
            response: Response = kwargs[response_param]

            # Hide the injected parameters from the wrapped handler.
            for name in (p.name for p in injected):
                kwargs.pop(name, None)

            decision = await endpoint_limiter.check(key_func(request))
            headers = rate_limit_headers(decision)

            if not decision.allowed:
                raise HTTPException(status_code=429, detail="Too Many Requests", headers=headers)

            if decision.delay > 0:
                await asyncio.sleep(decision.delay)

            result = await func(*args, **kwargs)

            # Setting headers on the injected Response makes FastAPI merge
            # them into whatever the handler returned, so handlers can keep
            # returning plain dicts.
            for name, value in headers.items():
                response.headers[name] = value
            if isinstance(result, Response):
                for name, value in headers.items():
                    result.headers[name] = value
            return result

        if injected:
            wrapper.__signature__ = signature.replace(parameters=params + injected)  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _find_param(params: list[inspect.Parameter], annotation: type) -> str | None:
    """Name of the first parameter annotated with `annotation` or a subclass."""
    for param in params:
        declared = param.annotation
        if isinstance(declared, type) and issubclass(declared, annotation):
            return param.name
    return None
