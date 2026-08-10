"""Deriving a rate limit key from a request.

Which client a request belongs to is a security decision, not plumbing: get
it wrong and clients can either forge someone else's key or escape their own
limit entirely.
"""

from __future__ import annotations

from typing import Callable

from starlette.requests import Request

KeyFunc = Callable[[Request], str]


def client_ip_key(request: Request) -> str:
    """The direct socket peer address. The safe default.

    Deliberately ignores X-Forwarded-For. Any client can send that header, so
    trusting it by default would let anyone pick their own rate limit key --
    send a random value per request and the limit never applies. Behind a real
    proxy this key collapses to the proxy's address, which is wrong in the
    *safe* direction (over-limiting) rather than the exploitable one. Use
    forwarded_for_key explicitly when a trusted proxy is actually in front.
    """
    if request.client is None:  # e.g. ASGI transports without a peer
        return "unknown"
    return request.client.host


def forwarded_for_key(trusted_hops: int = 1) -> KeyFunc:
    """Read the client address from X-Forwarded-For. Opt-in only.

    `trusted_hops` is how many proxies you control at the *end* of the chain.
    The header is appended to by each hop, so the last entry comes from your
    own nearest proxy and the entries before it are progressively less
    trustworthy. Counting back `trusted_hops` from the right lands on the
    address your infrastructure observed, which a client cannot forge -- while
    anything further left is attacker-controlled and must not be used.

    Getting this number wrong is a real bypass: too many hops and you read a
    client-supplied value.
    """
    if trusted_hops < 1:
        raise ValueError("trusted_hops must be >= 1")

    def key(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if not forwarded:
            return client_ip_key(request)
        chain = [part.strip() for part in forwarded.split(",") if part.strip()]
        if not chain:
            return client_ip_key(request)
        index = len(chain) - trusted_hops
        if index < 0:
            # Chain shorter than expected: fall back to the socket peer rather
            # than reading an entry the client could have supplied.
            return client_ip_key(request)
        return chain[index]

    return key


def path_scoped(key_func: KeyFunc) -> KeyFunc:
    """Scope a key to the request path, so limits are per-endpoint per-client."""

    def key(request: Request) -> str:
        return f"{request.url.path}:{key_func(request)}"

    return key
