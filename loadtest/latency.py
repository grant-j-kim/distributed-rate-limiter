"""Serial per-check latency: what one rate limit check actually costs.

    PYTHONPATH=src .venv/bin/python -m loadtest.latency

Deliberately separate from the scenarios in `__main__.py`. Those fire requests
concurrently, so their timings include queueing for a connection from a
bounded pool -- realistic for a server under a spike, but useless as a
per-check cost, since the number mostly describes the pool.

Here every check waits for the previous one to finish. Nothing queues, so the
median is the round trip plus the script's own execution, which is the figure
worth comparing between algorithms.

The limit is set far above the request count on purpose: a rejected request
takes a different (usually shorter) path through the Lua, and a run that
started rejecting halfway would report a blend of two different operations.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

from loadtest.scenarios import ALGORITHMS, build_limiter, fresh_prefix

REDIS_URL = "redis://localhost:6379/14"
CHECKS = 2000
WARMUP = 100


async def measure(limiter, checks: int) -> list[float]:
    """Round-trip milliseconds for `checks` sequential calls."""
    timings: list[float] = []
    for i in range(checks):
        started = time.perf_counter()
        await limiter.check(f"client-{i}")
        timings.append((time.perf_counter() - started) * 1000)
    return timings


async def main() -> int:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("redis-py is not installed: pip install -e '.[redis]'")
        return 1

    client = aioredis.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        print(f"no Redis server at {REDIS_URL}: {exc}")
        return 1

    print(f"{CHECKS} sequential checks per algorithm, one at a time\n")
    print(f"  {'algorithm':<24} {'median':>9} {'p95':>9} {'min':>9}")
    print(f"  {'-' * 24} {'-' * 9} {'-' * 9} {'-' * 9}")

    medians = []
    for algorithm in ALGORITHMS:
        limiter = build_limiter(
            algorithm, client, prefix=fresh_prefix("latency", algorithm)
        )
        # A high limit so nothing is rejected, and a warm-up so the SCRIPT LOAD
        # and TCP handshake are not counted as a first-request outlier.
        limiter.limit = 10**9
        await measure(limiter, WARMUP)

        timings = await measure(limiter, CHECKS)
        median = statistics.median(timings)
        medians.append(median)
        p95 = sorted(timings)[int(len(timings) * 0.95)]
        print(f"  {algorithm:<24} {median:>7.3f}ms {p95:>7.3f}ms {min(timings):>7.3f}ms")

    print(f"\n  median across all five: {min(medians):.3f}-{max(medians):.3f} ms")
    await client.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
