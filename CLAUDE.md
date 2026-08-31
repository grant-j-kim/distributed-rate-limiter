# Distributed Rate Limiter

## Purpose
Building a rate limiter from scratch that controls how many requests a client
can make in a given time window, and works correctly across multiple distributed
server instances (not just a single in-memory process).

## Status
The five library milestones are complete. 305 tests passing.
Milestone 6 (hosted demo) is in progress: both deployment blockers
are resolved, no demo code written yet.

- [x] **1. Core algorithms** — all five, in-memory, in `memory/`.
- [x] **2. Middleware layer** — `RateLimitMiddleware` + `@rate_limit`, in
      `middleware.py`. Runnable demo in `examples/app.py`.
- [x] **3. Distributed correctness** — all five, Redis-backed, in
      `redis_backend/`. Atomicity proven under real concurrency.
- [x] **4. Load testing** — asyncio load generator in `loadtest/`, three
      scenarios, JSONL logs and plots. Findings in `loadtest/README.md`:
      the fixed window admits **2.00x** its limit across a boundary, the
      sliding log exactly **1.00x**, the counter **1.10x**.
- [x] **5. Real usage (stretch)** — packaged as `drl-ratelimit` (the import
      stays `distributed_rate_limiter`; `distributed-rate-limiter` is taken on
      PyPI by an unrelated project). `backend="redis"` is now first-class in
      `create_limiter`, the middleware and the decorator.
      `packaging/verify_install.sh` builds the wheel, installs it into a
      throwaway venv outside the repo and runs a real FastAPI app against it:
      **50 of 120 concurrent admitted against a limit of 50, identical with
      one server process and with two.** Not published to PyPI.
- [ ] **6. Hosted demo (in progress)** — an interactive demo on Vercel in
      `web/`: a five-algorithm comparison playground replaying an arrival
      schedule against the *in-memory* limiters on a driven clock (instant,
      zero Redis), plus a rationed live punchline against Upstash. Both
      spikes are green: `redis.call('TIME')` works inside Lua on Upstash, and
      the race reproduces there exactly as locally — **the naive GET/SET
      control admitted 50 of 50 against a limit of 5; the Lua limiter
      admitted exactly 5.** The pipeline deploys.

## Plan / Milestones
1. **Core algorithms** — fixed window counter, sliding window log, sliding
   window counter, token bucket, leaky bucket. Each standalone and
   independently testable.
2. **Middleware layer** — pluggable FastAPI middleware, configurable
   per-endpoint (`@rate_limit(algorithm="token_bucket", limit=100, window=60)`),
   returning `429` with `Retry-After`.
3. **Distributed correctness** — Redis instead of in-memory state, using
   atomic operations (`INCR`+`EXPIRE`, or Lua for anything with arithmetic),
   tested under real concurrency.
4. **Load testing** — load generator (locust or asyncio) simulating steady,
   bursty, and multi-client load. Log allowed/rejected per algorithm. Plot all
   5 algorithms' behaviour at burst boundaries.
5. **Real usage (stretch goal)** — pip-installable so it can be integrated
   into real, separate applications rather than only tested synthetically.
6. **Hosted demo** — a public page where the five algorithms can be compared
   interactively, so the findings in `loadtest/README.md` can be *played with*
   rather than only read. The playground must cost no Redis; only the
   distributed-correctness punchline may, and it is budgeted.

## Key decisions
- Language/stack: Python, FastAPI, Redis.
- Correctness under concurrency is the core hard problem of this project —
  prioritize getting the atomicity right over adding more algorithms.
- Numbers used anywhere (load test results, overhead, correctness percentages)
  must come from actually running the code, not estimates.
- **The interface is async** (`async def check(key) -> Decision`), chosen up
  front so Redis and FastAPI need no rewrite.
- **Each (algorithm, backend) pair is its own class**, not one algorithm over a
  swappable store. An atomic token bucket cannot be built on a generic
  get-then-set store, because the read-modify-write gap *is* the race
  condition.
- **The clock is injectable.** In-memory limiters take a wall clock (not
  monotonic — that is per-process and would desync instances). Redis limiters
  default to reading Redis's own `TIME`, so instances with skewed clocks agree
  on window boundaries; tests inject a clock only for determinism.
- **`Decision.delay`** is how the leaky bucket shapes traffic. Callers must
  await it before proceeding, or the shaper silently degrades into a meter.

## How the tests are organised
This is the part worth understanding before adding anything.

- `tests/test_common_scenarios.py` — 10 scenarios every limiter must satisfy,
  parameterized over **every** implementation in `conftest.ALL_LIMITERS`
  (10 in-memory + Redis = 100 test instances). A new backend joins that list
  and must pass unchanged. That is how equivalence is *asserted*, not assumed.
- `tests/test_redis_concurrency.py` — parameterized over
  `conftest.REDIS_LIMITERS`. Includes a deliberately naive `GET`/`SET`
  limiter as a **control**: it must over-admit. If that test ever starts
  passing, the concurrency suite has stopped exercising real concurrency.
- `tests/test_<algorithm>.py` — behaviour unique to one algorithm (the fixed
  window's boundary burst, the counter's measured divergence, and so on).
- `tests/test_loadtest.py` — the measurement rig itself. A bug in `loadtest/`
  is indistinguishable from a finding, so the sliding-interval peak metric
  and the runner's handling of shaping delay are pinned down here.
- Redis-backed tests use unique key prefixes on DB 15 and skip cleanly when no
  server is reachable. The load generator uses DB 14, so a run can never
  collide with the suite.

## Gotchas already paid for
- **TTL means something different in every algorithm.** Fixed window: never
  refresh (the TTL *is* the window; refreshing locks clients out). Sliding log:
  always refresh (expiry is only GC; the window is enforced by score pruning).
  Buckets: must outlive a full refill, or a throttled client gets a free reset
  by waiting. Sliding counter: must span two windows, since this window's count
  is read as `prev` in the next.
- **Float drift is real and crosses into Lua.** Incremental refill accumulates
  binary error, so `check()` tolerates being `1e-9` short. `retry_after` is
  reported exactly — padding it instead pushes it past its own window.
- **`ZADD` on an existing member updates its score** rather than inserting, so
  sliding-log entries carry a uuid member. A colliding variant admitted 20 of
  20 against a limit of 5.
- **This project sits in iCloud-synced `~/Desktop`.** iCloud sets `UF_HIDDEN`
  on `.venv` `.pth` files and Python 3.13 silently skips hidden `.pth` files,
  so the editable install stops importing with no diagnostic. pytest is immune
  (`pythonpath` in `pyproject.toml`); everything else needs `PYTHONPATH=src`.
- **Over-admission has to be measured over a *sliding* interval.** Bucketing
  admissions by the limiter's own window index reports 20 and 20 for a burst
  that straddles a boundary — both within limit, nothing visible — when the
  client actually got 40 inside one window's worth of time. Peaks are also
  computed per key and then maximised, never pooled: pooling adds up
  independent quotas and reports three well-behaved clients as a 3x breach.
- **Peak admission alone ranks the token bucket with the fixed window.** Both
  measured ~2x (1.95x and 2.00x), and they mean opposite things — one is a
  configured burst allowance, the other is a forgotten window.
- **`create_limiter` must not default `clock` to `default_clock`.** In-memory
  limiters want the local wall clock; Redis limiters want *no clock argument*,
  which is what makes them read Redis's own `TIME`. Forwarding a default here
  would silently give every Redis limiter the local process clock — nothing
  raises, no test fails, and instances with skewed clocks stop agreeing on
  window boundaries. Hence the `_UNSET` sentinel: `None` can't do the job
  because for Redis it is a meaningful value. `middleware.py` passes the
  sentinel straight through for the same reason.
- **The middleware meters every non-exempt path**, including whatever a test
  script hits first. `packaging/drive_consumer.py` runs in phases with a flush
  between them because 15 requests to a decorated endpoint also spend 15 of
  the app-wide quota — the first version of that script "failed" at 34 of 50
  for exactly that reason.
- **An editable install hides packaging bugs**, since the source tree is on
  `sys.path` either way. Missing subpackages, a missing `py.typed` and a wrong
  `packages` setting only show up from a venv that cannot see the repo.
- **Vercel never installs `[project.optional-dependencies]`.** It installs with
  `uv sync --active --no-dev --link-mode hardlink --no-editable` and passes no
  `--extra` — there is no `--extra`, `--all-extras` or `optional-dependencies`
  anywhere in its Python builder. So the `demo` extra is declared and silently
  ignored, the build *succeeds*, and the first request dies on
  `ModuleNotFoundError: fastapi`. Fixed with `[tool.vercel.scripts]`
  `vercel-build`, which runs *after* the default install. Deliberately not the
  `install` hook: a custom `install` script sets `assumeDepsInstalled` and
  replaces `uv sync` entirely, so you would then own installing the library too.
  A committed `requirements.txt` does not help either — `pyproject.toml` wins.
- **`rediss://` fails from this venv with a misleading error.** macOS Python
  has no CA bundle of its own, so connecting to Upstash raises
  `ConnectionError: ... CERTIFICATE_VERIFY_FAILED`, which reads like the server
  being unreachable or blocking you rather than a local trust-store gap. Fix is
  `SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')`, not
  disabling verification. Local only: the Linux build image has system CA
  certificates. Same family as the iCloud `UF_HIDDEN` gotcha — an environment
  problem wearing a library problem's error message.
- **Vercel refuses to guess between two FastAPI apps.** `examples/app.py` and
  `packaging/consumer_app.py` both export `app`, so detection fails outright
  rather than picking one. `[tool.vercel] entrypoint = "module:variable"` is
  required, and its module path must stay in step with wherever the demo app
  actually lives.

## Running things
```bash
# tests (starts nothing; Redis tests skip if no server)
.venv/bin/python -m pytest -q

# Redis, built from source at ~/.local/bin (no Homebrew: /opt/homebrew
# belongs to another user account and predates this macOS version)
~/.local/bin/redis-server --port 6379 --save '' --appendonly no --daemonize yes \
  --pidfile /tmp/redis-drl.pid --logfile /tmp/redis-drl.log
~/.local/bin/redis-cli ping
~/.local/bin/redis-cli shutdown nosave   # stop it

# demo server: one endpoint per algorithm, docs at /docs
PYTHONPATH=src .venv/bin/python -m uvicorn examples.app:app --reload

# load test: needs Redis running (uses DB 14), ~90s for all three scenarios
PYTHONPATH=src .venv/bin/python -m loadtest
PYTHONPATH=src .venv/bin/python -m loadtest --scenario boundary_burst
PYTHONPATH=src .venv/bin/python -m loadtest --plot-only  # rebuild from logs,
                                                         # no Redis, no traffic

# packaging: build the wheel, install it into a throwaway venv outside the
# repo, run a real FastAPI app against it on two processes. Needs Redis (DB 13)
.venv/bin/python -m build
./packaging/verify_install.sh
```

## Working style preferences
- Before writing any non-trivial logic (especially anything involving
  concurrency, atomicity, or the Redis/Lua scripting layer), explain the
  reasoning and approach first, then write the code. Don't jump straight to
  a code block for anything that isn't pure boilerplate.
- After any significant change (a new algorithm implemented, the middleware
  layer working, the distributed/Redis version passing concurrency tests,
  load test results produced), stop and prompt to commit — describe what
  changed and propose a commit message, but wait for confirmation before
  running `git commit`.
- Don't auto-commit without that prompt, even in auto-accept-style workflows.
- Commit messages: condensed. Subject plus a short body keeping measured
  numbers and any non-obvious fix; the per-case detail already lives in test
  docstrings.
