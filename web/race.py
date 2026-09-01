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
RACE_WINDOW = 30.0
"""The limit being raced. Small, so the counts are numbers a visitor can hold.

The algorithm is the **sliding window log**, not the fixed window, and that is
a correctness requirement rather than a preference. A fixed window admits a
full fresh allowance either side of a boundary -- this project's own 2.00x
finding -- so a run that straddled one would admit 10 against a limit of 5 and
read on the page as the atomic limiter failing. It measured exactly that in
production: 50 requests spread over 7.3s crossed a 10s boundary and admitted
5 + 5. The sliding log is exact wherever it falls, so "exactly 5" holds
whenever a visitor happens to press the button."""

PER_IP_RUNS, PER_IP_WINDOW = 10, 3600.0
MONTHLY_RUNS, MONTHLY_WINDOW = 1000, 30 * 24 * 3600.0
"""A run costs about 300 Redis commands: each naive fire spends TIME, ZCOUNT,
ZADD and EXPIRE, and each atomic fire spends TIME and one EVALSHA. 1000 runs is
roughly 300k against a free tier of 500k per month, which leaves real margin --
1500 would not.

The two limits answer different questions, and only the global one protects
the free tier: the per-IP limit stops a single visitor monopolising the demo
and says nothing about the total. Ten an hour is enough to stop that while
still letting someone press the button a few times and watch the numbers move,
which is most of the point.

The monthly budget uses a **sliding window counter, not a fixed window**. A
month-long fixed window would let someone spend the whole budget on the last
day of one window and the whole of it again on the first day of the next --
2x the intended spend across a few hours. That is this project's own headline
finding, and it would be an embarrassing way to run out of credit."""

LEAD = 8.0
"""How long after `/start` the first volley fires.

Eight seconds because the participants have to *exist* first. Measured in
production with a 2s lead, 49 of the 50 requests started after the barrier had
already passed: the first request spins up an instance which then sleeps, and
Vercel cold-starts the rest at roughly 113ms each, so fifty of them take about
5.7 seconds to assemble. A barrier that fires before its participants are alive
is not a barrier. They do come up in parallel with the first one waiting -- the
5664ms spread showed that -- so the fix is lead time, not a different design.

Every fire sleeps until a common instant on Redis's clock before touching it,
so invocation stagger is absorbed by the sleep instead of landing in the
measurement. Without this the requests arrive hundreds of milliseconds apart --
7.3 seconds apart, measured in production -- while the gap they have to overlap
is one round trip, about one millisecond. A race whose participants arrive that
far apart is not a race.

This is the technique `loadtest/runner.py` already uses when it waits for
`next_boundary` before starting a scenario, and for the same reason: what is
being measured must not be at the mercy of when the harness happened to start.
The gap is untouched and the limiters are unmodified -- the only thing removed
is the scheduler."""

PHASE = 6.0
"""Seconds between the naive volley and the atomic one, so the two do not
contend for connections or instances while each is being measured.

Shorter than LEAD because the second volley reuses instances the first one
already warmed, so they assemble far faster than from cold."""

TOKEN_TTL = 120.0
"""How long a run token stays valid. Long enough for fifty round trips on a bad
connection, short enough that a leaked token is worthless."""


@dataclass(frozen=True)
class Rationing:
    """Why a run was refused, if it was."""

    allowed: bool
    reason: str = ""
    retry_after: float | None = None


class NaiveRedisSlidingWindowLog(RateLimiter):
    """Count, decide, append. The mistake the Lua script exists to prevent.

    A faithful naive counterpart to `RedisSlidingWindowLog`: the same algorithm,
    the same sorted set, the same pruning by score -- but issued as separate
    commands, so there is a gap between counting and appending. Any number of
    requests can count the same total before any of them appends, and all of
    them conclude they are under the limit.

    Deliberately the *same* algorithm as the atomic side, so the only
    difference between the two rows on the page is atomicity. Racing a naive
    counter against an atomic log would confound two variables and prove
    neither.

    Kept as a working implementation rather than a snippet because the page
    runs it: a described race is an assertion, a race you can watch fail is
    evidence. It is a deliberate copy of the control in
    `tests/test_redis_concurrency.py` -- that one is a control that must keep
    failing, this one is an exhibit, so they are allowed to drift and neither
    belongs in the library.
    """

    def __init__(self, client, limit: int, window: float, *, prefix: str, now: float):
        self.client = client
        self.limit = limit
        self.window = window
        self.prefix = prefix
        # Handed the same Redis clock reading the endpoint already took for its
        # timing. Reading a local clock here would add a second bug -- skew
        # between instances -- on top of the race, and muddy which one the page
        # is showing.
        self.now = now

    async def check(self, key: str) -> Decision:
        redis_key = f"{self.prefix}:{key}"
        cutoff = self.now - self.window

        count = await self.client.zcount(redis_key, cutoff, "+inf")  # <-- gap opens
        if count >= self.limit:
            return Decision(allowed=False, limit=self.limit, remaining=0,
                            reset_after=self.window, retry_after=self.window)

        # ... and closes here. Everything in between is a window in which
        # another request counts the same total.
        #
        # The member is a uuid because ZADD on an existing member updates its
        # score instead of inserting: a colliding member would silently make
        # the log hold one entry however many requests arrived.
        await self.client.zadd(redis_key, {uuid.uuid4().hex: self.now})
        await self.client.expire(redis_key, int(self.window) + 1)
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


_CACHED = None


def _client(max_connections: int = CONCURRENCY + 14):
    """One client per instance, reused across invocations, never closed here.

    Opening a client per request means a fresh TLS handshake to Upstash every
    time. Measured in production that showed up as 50 requests reaching Redis
    over 7.3 seconds -- about 145ms apart -- which is no overlap at all, and a
    race with no overlap is a race nobody loses. Reusing the connection is both
    the standard serverless pattern and the thing that gives the requests a
    chance to actually collide.

    The cap is generous on purpose. Connections are created lazily, so an
    instance handling four concurrent requests opens four -- the number is a
    ceiling, not a reservation. A tight cap is actively harmful here: redis-py
    raises MaxConnectionsError once it is hit, and the obvious alternative, a
    blocking pool, would be worse. Queueing for a connection *serialises* the
    requests, which would destroy the overlap the race depends on. A demo that
    cannot lose the race proves nothing.
    """
    global _CACHED
    import redis.asyncio as redis

    if _CACHED is None:
        _CACHED = redis.from_url(redis_url(), max_connections=max_connections)
    return _CACHED


def _secret() -> bytes:
    """Signing key for run tokens, derived from the Redis URL.

    Every instance that can run a race already shares this string, so it needs
    no second environment variable and cannot drift between instances -- which
    a per-process random key would, the moment two instances are warm. It is
    hashed, never sent anywhere, and only ever used to sign an opaque run id.
    """
    return hashlib.sha256(("drl-race:" + (redis_url() or "")).encode()).digest()


def issue_token(run_id: str, expires_at: float, start_at: float = 0.0) -> str:
    payload = f"{run_id}:{expires_at:.0f}:{start_at:.3f}".encode()
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()[:32]


def target_for(start_at: float, variant: str) -> float:
    """When this fire should touch Redis, derived from the signed start.

    Derived server-side from a signed value rather than taken from the client,
    so a caller cannot nominate its own firing instant and quietly spread the
    volley back out.
    """
    return start_at + (PHASE if variant == "lua" else 0.0)


def verify_token(run_id: str, expires_at: float, token: str,
                 start_at: float = 0.0) -> bool:
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
    return hmac.compare_digest(token, issue_token(run_id, expires_at, start_at))


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


def new_run(start_at: float) -> tuple[str, float, str]:
    run_id = uuid.uuid4().hex[:16]
    expires_at = time.time() + TOKEN_TTL
    return run_id, expires_at, issue_token(run_id, expires_at, start_at)


def build_limiter(client, run_id: str, variant: str, now: float) -> RateLimiter:
    """One limiter for one run, so concurrent visitors never share a counter.

    Keys carry the run id, which is why a replayed token is harmless: it can
    only reach a run that has already been charged for and filled up.
    """
    prefix = f"race:{run_id}:{variant}"
    if variant == "naive":
        return NaiveRedisSlidingWindowLog(client, RACE_LIMIT, RACE_WINDOW,
                                          prefix=prefix, now=now)
    # No clock argument: the Redis limiter reads Redis's own TIME, which is the
    # whole reason instances with skewed clocks still agree.
    return create_limiter("sliding_window_log", limit=RACE_LIMIT,
                          window=RACE_WINDOW, backend="redis",
                          client=client, prefix=prefix)
