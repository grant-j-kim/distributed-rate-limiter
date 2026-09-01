# drl-ratelimit

Five rate limiting algorithms — fixed window, sliding window log, sliding
window counter, token bucket, leaky bucket — each in two implementations: one
in-memory, and one backed by Redis that stays correct when several server
instances share a limit.

**[Live demo →](https://gklimiter-demo-two.vercel.app/)** — compare all five at
a burst boundary, then watch a limiter that counts, decides and appends as
three separate commands lose a real race. Run across 34 Vercel instances
against one Redis, it admitted **43 requests against a limit of 5**; the same
algorithm as one atomic Lua script admitted **exactly 5**.

```python
from distributed_rate_limiter import create_limiter

limiter = create_limiter("token_bucket", limit=100, window=60)

decision = await limiter.check(client_id)
if not decision.allowed:
    raise HTTPException(429, headers={"Retry-After": str(decision.retry_after)})
```

Point it at Redis and the same limit is enforced across every instance:

```python
import redis.asyncio as redis

limiter = create_limiter(
    "token_bucket", limit=100, window=60,
    backend="redis", client=redis.from_url("redis://localhost:6379"),
)
```

## Install

```bash
pip install drl-ratelimit                # algorithms only, no dependencies
pip install "drl-ratelimit[redis]"       # + redis-py
pip install "drl-ratelimit[server]"      # + FastAPI integration
```

The algorithms depend on nothing outside the standard library — including the
Redis-backed ones, which take the client as a duck-typed argument. redis-py is
a dependency of the application that supplies a client, not of this package.

## FastAPI

App-wide:

```python
from distributed_rate_limiter.middleware import RateLimitMiddleware

app.add_middleware(
    RateLimitMiddleware,
    algorithm="token_bucket", limit=100, window=60,
    backend="redis", client=redis_client,
)
```

Or per endpoint, which app-wide middleware cannot do — ASGI middleware runs
before routing resolves, so at that point there is no way to know which
endpoint a request will reach:

```python
from distributed_rate_limiter.middleware import rate_limit

@app.get("/expensive")
@rate_limit("sliding_window_log", limit=10, window=60,
            backend="redis", client=redis_client)
async def expensive():
    ...
```

Both return `429` with `Retry-After` and the `X-RateLimit-*` headers. When
both are active, the endpoint's own quota headers win rather than being
appended to — two contradictory `X-RateLimit-Limit` values would leave the
client no way to tell which one applies.

### Choosing a key

Which client a request belongs to is a security decision, not plumbing:

```python
from distributed_rate_limiter.keys import client_ip_key, forwarded_for_key, path_scoped
```

`client_ip_key` is the default and deliberately ignores `X-Forwarded-For` —
any client can send that header, so trusting it by default would let anyone
choose their own rate limit key and escape the limit entirely. Behind a real
proxy, use `forwarded_for_key(trusted_hops=n)`, where `n` is how many proxies
*you control* at the end of the chain.

## Which algorithm

Measured, not asserted — from `loadtest/`, replaying one identical arrival
schedule against all five at limit 20 per 2s. `max/win` is the most requests
admitted in any interval one window long, wherever it falls:

| algorithm | across a boundary | state per key | notes |
|---|---|---|---|
| fixed window | **2.00×** | O(1) | cheapest, and the only one that over-admits by accident |
| sliding window log | **1.00×** | O(n) | exact, and pays for it in memory |
| sliding window counter | **1.10×** | O(1) | the practical compromise |
| token bucket | 1.10–1.15× | O(1) | burst size decoupled from sustained rate; the range is a knife-edge, see `loadtest/README.md` |
| leaky bucket | 1.05× | O(1) | shapes traffic instead of rejecting it |

The fixed window admitted **40 requests against a limit of 20** when a burst
straddled a boundary, because it has no memory across one. If that matters,
the sliding window counter gets you to 1.10× for the same O(1) state.

The token bucket also measured ~2× peak admission under sustained load, but
for the opposite reason: a full starting bucket is a burst allowance it was
*configured* to grant. Peak admission alone ranks those two together, which is
why `loadtest/README.md` plots the curves rather than only the totals.

The leaky bucket is the only one that **shapes**. It returns
`Decision.delay`, and the caller must await it before proceeding — the
middleware does. Ignore it and the shaper silently degrades into a meter.

## Design

- **The interface is async.** `async def check(key) -> Decision`, chosen up
  front so Redis and FastAPI needed no rewrite.
- **Each (algorithm, backend) pair is its own class**, not one algorithm over
  a swappable store. An atomic token bucket cannot be built on a generic
  get-then-set store, because the read-modify-write gap *is* the race
  condition. Atomicity has to be fused into the storage operation — `INCR`
  plus `EXPIRE`, or a Lua script.
- **Redis limiters read Redis's own `TIME`** by default, so instances with
  skewed clocks still agree on where a window boundary falls. Passing a
  `clock` overrides that and is meant for tests.

## Correctness

Concurrency is the hard part of this project, so it is tested rather than
argued. A deliberately naive `GET`/`SET` limiter is kept in the suite as a
**control**: fired 50 concurrent requests against a limit of 5 it admitted all
50, every trial. The Lua implementations admitted exactly 5, every trial. If
the control ever starts passing, the concurrency suite has stopped exercising
real concurrency.

All five Redis backends held exactly to their limit at 10, 100 and 500
concurrent requests, in every trial. Every implementation, in-memory and Redis
alike, must also pass one shared scenario suite — equivalence is asserted, not
assumed.

```bash
pip install -e ".[dev,redis,server,loadtest]"
pytest -q          # 305 tests; Redis-backed ones skip if no server is running
```

## License

MIT
