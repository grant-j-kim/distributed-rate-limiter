"""Run the load scenarios and write logs, tables and plots.

    PYTHONPATH=src .venv/bin/python -m loadtest              # run everything
    PYTHONPATH=src .venv/bin/python -m loadtest --scenario boundary_burst
    PYTHONPATH=src .venv/bin/python -m loadtest --plot-only  # re-plot the logs

`--plot-only` exists to keep the analysis honest: it regenerates every table
and figure from the JSONL alone, with no Redis and no traffic. If a number in
the writeup cannot be reproduced that way, it did not come from the data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from loadtest.analysis import format_table, summarize
from loadtest.plots import render
from loadtest.runner import Record, next_boundary, run_schedule, to_json_rows, warm_up
from loadtest.scenarios import (
    ALGORITHMS,
    LIMIT,
    SCENARIOS,
    WINDOW,
    Scenario,
    build_limiter,
    fresh_prefix,
)

RESULTS = Path(__file__).parent / "results"
REDIS_URL = "redis://localhost:6379/14"  # not 15: that is the test database


async def run_scenario(client, scenario: Scenario) -> list[Record]:
    """Replay one schedule against all five algorithms, one at a time.

    Sequentially rather than concurrently, and deliberately: five limiters
    hammering one Redis at once would contend for connections and CPU, and
    the latency figures would then describe the rig rather than the
    algorithms. Each gets its own boundary alignment and its own key
    namespace, so the runs are independent.
    """
    records: list[Record] = []
    for algorithm in ALGORITHMS:
        limiter = build_limiter(
            algorithm, client, prefix=fresh_prefix(scenario.name, algorithm)
        )
        await warm_up(limiter)
        t0 = await next_boundary(client, WINDOW)
        print(f"  {algorithm:<24} ", end="", flush=True)
        got = await run_schedule(limiter, algorithm, scenario.arrivals, t0=t0)
        allowed = sum(1 for r in got if r.allowed)
        print(f"{allowed}/{len(got)} allowed")
        records += got
    return records


def write_log(scenario: Scenario, records: list[Record]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{scenario.name}.jsonl"
    with path.open("w") as fh:
        for row in to_json_rows(records):
            fh.write(json.dumps(row) + "\n")

    meta = {
        "scenario": scenario.name,
        "question": scenario.question,
        "limit": LIMIT,
        "window": WINDOW,
        "duration": scenario.duration,
        "clients": list(scenario.clients),
        "offered_per_algorithm": len(scenario.arrivals),
        "algorithms": ALGORITHMS,
    }
    (RESULTS / f"{scenario.name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def read_log(name: str) -> tuple[list[Record], dict]:
    meta = json.loads((RESULTS / f"{name}.meta.json").read_text())
    records = [
        Record(**json.loads(line))
        for line in (RESULTS / f"{name}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return records, meta


def report(name: str, records: list[Record], meta: dict) -> None:
    by_client = len(meta["clients"]) > 1
    print(f"\n=== {name} ===")
    print(meta["question"])
    print()
    print(format_table(summarize(records, limit=LIMIT, window=WINDOW), by_client=False))
    if by_client:
        print()
        print(
            format_table(
                summarize(records, limit=LIMIT, window=WINDOW, by_client=True),
                by_client=True,
            )
        )


async def main_async(names: list[str]) -> None:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        sys.exit("redis-py is not installed: pip install -e '.[redis]'")

    pool = aioredis.BlockingConnectionPool.from_url(
        REDIS_URL, decode_responses=True, max_connections=64, timeout=20
    )
    client = aioredis.Redis(connection_pool=pool)
    try:
        await client.ping()
    except Exception as exc:
        sys.exit(f"no Redis server at {REDIS_URL}: {exc}")

    try:
        for name in names:
            scenario = SCENARIOS[name]()
            print(f"\nrunning {name}: {len(scenario.arrivals)} requests x "
                  f"{len(ALGORITHMS)} algorithms, limit={LIMIT}/{WINDOW}s")
            records = await run_scenario(client, scenario)
            write_log(scenario, records)
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip the load run; rebuild tables and figures from the logs",
    )
    args = parser.parse_args()
    names = args.scenario or list(SCENARIOS)

    if not args.plot_only:
        asyncio.run(main_async(names))

    for name in names:
        records, meta = read_log(name)
        report(name, records, meta)
        for path in render(name, records, meta, RESULTS):
            print(f"  wrote {path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
