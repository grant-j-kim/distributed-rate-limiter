"""The punchline: a real race, against a real Redis, live.

The playground costs nothing and proves nothing about concurrency -- it is one
process stepping a clock by hand. This is the other half: fifty genuinely
concurrent requests against a shared Redis, run twice. Once against a naive
GET/SET limiter, which admits all fifty against a limit of five, and once
against the Lua one, which admits exactly five.

Three things make it honest rather than theatre:

- **The browser fires the fifty**, one `fetch` each, so each becomes its own
  function invocation. Doing it as one `asyncio.gather` inside a single request
  would look identical and demonstrate something strictly weaker -- an
  in-process interleaving race rather than a distributed one. (Vercel's Fluid
  compute may still land several on one instance, so the claim on the page is
  "concurrent requests", never "separate machines".)
- **The naive limiter here is a real one**, not a description of one. It is a
  deliberate copy of the control in `tests/test_redis_concurrency.py`. The two
  serve opposite masters -- that one is a control that must keep *failing*,
  this one is an exhibit -- so they are allowed to drift, and neither should be
  imported into the library.
- **It is rationed, and says so.** Every run spends real commands from a free
  tier, so a per-IP limit and a global monthly budget both have to pass before
  a run starts. Both are this project's own limiters, pointed at itself.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass

from distributed_rate_limiter import create_limiter
from distributed_rate_limiter.base import Decision, RateLimiter

CONCURRENCY = 50
"""Requests fired at once, per variant."""

RACE_LIMIT = 5
RACE_WINDOW = 10.0
"""The limit being raced. Small, so "50 admitted against a limit of 5" is a
number a visitor can hold in their head."""

PER_IP_RUNS, PER_IP_WINDOW = 3, 3600.0
MONTHLY_RUNS, MONTHLY_WINDOW = 1500, 30 * 24 * 3600.0
"""A run costs roughly 150 Redis commands: 50 EVALSHA for the Lua limiter, plus
50 GET and 50 SET for the naive one. 1500 runs is about 225k commands against a
free tier of 500k per month, leaving room for the rationing checks themselves
and a wide margin.

The monthly budget uses a **sliding window counter, not a fixed window**. A
month-long fixed window would let someone spend the whole budget on the last
day of one window and the whole of it again on the first day of the next --
2x the intended spend across a few hours. That is this project's own headline
finding, and it would be an embarrassing way to run out of credit."""

TOKEN_TTL = 120.0
"""How long a run token stays valid. Long enough for fifty round trips on a bad
connection, short enough that a leaked token is worthless."""


@dataclass(frozen=True)
class Rationing:
    """Why a run was refused, if it was."""

    allowed: bool
    reason: str = ""
    retry_after: float | None = None


class NaiveRedisFixedWindow(RateLimiter):
    """Read, decide, write. The mistake the Lua script exists to prevent.

    Three round trips with two gaps in between. Any number of clients can read
    the same count before any of them writes it back, so they all conclude they
    are under the limit and all proceed.

    Kept as a working implementation rather than a snippet because the page
    *runs* it: a described race is an assertion, a race you can watch fail is
    evidence.
    """

    def __init__(self, client, limit: int, window: float, *, prefix: str):
        self.client = client
        self.limit = limit
        self.window = window
        self.prefix = prefix

    async def check(self, key: str) -> Decision:
        redis_key = f"{self.prefix}:{key}"
        current = await self.client.get(redis_key)  # <-- the gap opens here
        count = int(current or 0)

        if count >= self.limit:
            return Decision(allowed=False, limit=self.limit, remaining=0,
                            reset_after=self.window, retry_after=self.window)

        # ... and closes here. Everything between is a window in which another
        # request reads the same count.
        await self.client.set(redis_key, count + 1, ex=int(self.window))
        return Decision(allowed=True, limit=self.limit,
                        remaining=self.limit - (count + 1), reset_after=self.window)

    async def reset(self, key: str) -> None:
        await self.client.delete(f"{self.prefix}:{key}")


def redis_url() -> str | None:
    """The live Redis, or None when there isn't one.

    Absence is the switch. `REDIS_URL` is set on the production environment
    only, so preview deployments have no live Redis and fall back to the
    recording without anyone having to remember to set a flag. A flag can be
    forgotten; a missing variable cannot be.
    """
    return os.environ.get("REDIS_URL") or None


def _client(max_connections: int = 2):
    """A client for the life of one request, closed by the caller.

    Upstash caps concurrent connections per database and does not publish the
    number; the documented fix for serverless is exactly this -- open inside
    the function, close when done -- so the connection count tracks in-flight
    invocations rather than accumulating.

    The default pool is 2, not `CONCURRENCY`. Each `/fire` invocation performs
    a single check, so a pool sized for the whole race would be one invocation
    claiming headroom for fifty. Only the recorder, which fires all fifty from
    one process, asks for the larger pool.
    """
    import redis.asyncio as redis

    return redis.from_url(redis_url(), max_connections=max_connections)


def _secret() -> bytes:
    """Signing key for run tokens, derived from the Redis URL.

    Every instance that can run a race already shares this string, so it needs
    no second environment variable and cannot drift between instances -- which
    a per-process random key would, the moment two instances are warm. It is
    hashed, never sent anywhere, and only ever used to sign an opaque run id.
    """
    return hashlib.sha256(("drl-race:" + (redis_url() or "")).encode()).digest()


def issue_token(run_id: str, expires_at: float) -> str:
    payload = f"{run_id}:{expires_at:.0f}".encode()
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()[:32]


def verify_token(run_id: str, expires_at: float, token: str) -> bool:
    """Validate a run token without touching Redis.

    Rationing is charged once, when a run starts. If each of the fifty fires
    re-checked a limiter, the check would cost more commands than the race it
    is protecting. Signing the run id instead makes an unauthorised fire free
    to refuse. A token can be replayed inside its window, but only against the
    keys of a run that has already been paid for and consumed, so replaying it
    buys nothing.
    """
    if expires_at < time.time():
        return False
    return hmac.compare_digest(token, issue_token(run_id, expires_at))


async def check_rationing(client, ip_key: str) -> Rationing:
    """Charge one run against the per-IP limit and the global monthly budget.

    Both must pass, and they answer different questions. The per-IP limit stops
    one visitor monopolising the demo; it says nothing at all about the total.
    Two hundred well-behaved strangers would exhaust a month between them
    without any of them breaking a rule -- which is the same per-key versus
    pooled distinction the multi_client scenario measures.
    """
    per_ip = create_limiter("sliding_window_counter", limit=PER_IP_RUNS,
                            window=PER_IP_WINDOW, backend="redis",
                            client=client, prefix="race:quota:ip")
    decision = await per_ip.check(ip_key)
    if not decision.allowed:
        return Rationing(False, "per-ip", decision.retry_after)

    budget = create_limiter("sliding_window_counter", limit=MONTHLY_RUNS,
                            window=MONTHLY_WINDOW, backend="redis",
                            client=client, prefix="race:quota:global")
    decision = await budget.check("all")
    if not decision.allowed:
        return Rationing(False, "budget", decision.retry_after)

    return Rationing(True)


def new_run() -> tuple[str, float, str]:
    run_id = uuid.uuid4().hex[:16]
    expires_at = time.time() + TOKEN_TTL
    return run_id, expires_at, issue_token(run_id, expires_at)


def build_limiter(client, run_id: str, variant: str) -> RateLimiter:
    """One limiter for one run, so concurrent visitors never share a counter.

    Keys carry the run id, which is why a replayed token is harmless: it can
    only reach a run that has already been charged for and filled up.
    """
    prefix = f"race:{run_id}:{variant}"
    if variant == "naive":
        return NaiveRedisFixedWindow(client, RACE_LIMIT, RACE_WINDOW, prefix=prefix)
    return create_limiter("fixed_window", limit=RACE_LIMIT, window=RACE_WINDOW,
                          backend="redis", client=client, prefix=prefix)
