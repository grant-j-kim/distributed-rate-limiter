# Distributed Rate Limiter

## Purpose
Building a rate limiter from scratch that controls how many requests a client
can make in a given time window, and works correctly across multiple distributed
server instances (not just a single in-memory process).

## Plan / Milestones
1. **Core algorithms** — implement each of the following as standalone,
   independently testable modules:
   - Fixed window counter
   - Sliding window log
   - Sliding window counter
   - Token bucket
   - Leaky bucket
   Unit test each against the same scenarios: steady traffic under limit,
   burst at window boundary, burst exceeding limit, exactly-at-limit edge cases.
2. **Middleware layer** — wrap the algorithms as pluggable FastAPI middleware,
   configurable per-endpoint (e.g. `@rate_limit(algorithm="token_bucket",
   limit=100, window=60)`). Return proper `429 Too Many Requests` with a
   `Retry-After` header.
3. **Distributed correctness** — swap in-memory state for Redis. Handle the
   core race condition (two concurrent requests both read the same count,
   both increment, both get allowed) using atomic operations: Redis
   `INCR`+`EXPIRE` or a Lua script (`EVAL`) for the token bucket refill logic.
   Test correctness under real concurrency (multiple async clients hammering
   the same key simultaneously).
4. **Load testing** — write a load generator (locust or a simple asyncio
   script) simulating steady load, bursty load, and multiple clients. Log
   allowed/rejected requests per algorithm. Produce plots comparing all
   5 algorithms' behavior at burst boundaries.
5. **Real usage (stretch goal)** — package as a pip-installable middleware so
   it can be integrated into real, separate applications rather than only
   tested synthetically.

## Key decisions
- Language/stack: Python, FastAPI, Redis.
- Correctness under concurrency is the core hard problem of this project —
  prioritize getting the atomicity right over adding more algorithms.
- Numbers used anywhere (load test results, overhead, correctness percentages)
  must come from actually running the code, not estimates.

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
