"""Drive the consumer app and check the installed package behaves.

Run in phases, with the caller flushing Redis between them. That is not
tidiness -- the app-wide middleware meters *every* non-exempt path, so the
15 requests the endpoint phase sends to /expensive also spend 15 of the
app-wide bucket's 50. Measuring both in one window makes the global figure
depend on how many requests the previous check happened to make, which is a
property of this script rather than of the library.

Exits non-zero on any failed expectation, so the shell script can gate on it.
"""

from __future__ import annotations

import asyncio
import collections
import sys

import httpx

GLOBAL_LIMIT = 50
ENDPOINT_LIMIT = 5
CONCURRENT = 120

failures: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {actual}"
          + ("" if ok else f" (expected {expected})"))
    if not ok:
        failures.append(label)


async def phase_endpoint(client: httpx.AsyncClient, base: list[str]) -> None:
    version = (await client.get(f"{base[0]}/version")).json()
    print(f"\ndrl-ratelimit {version['version']}")
    print(f"  loaded from {version['loaded_from']}")
    check(
        "imported from the installed wheel, not the source tree",
        "site-packages" in version["loaded_from"],
        True,
    )

    print(f"\nper-endpoint  /expensive  sliding_window_log, {ENDPOINT_LIMIT} per 10s")
    results = [await client.get(f"{base[0]}/expensive") for _ in range(15)]
    codes = collections.Counter(r.status_code for r in results)
    check("admitted", codes[200], ENDPOINT_LIMIT)
    check("rejected", codes[429], 15 - ENDPOINT_LIMIT)

    rejected = next(r for r in results if r.status_code == 429)
    check("X-RateLimit-Limit", rejected.headers.get("x-ratelimit-limit"), "5")
    # Retry-After is rounded up and never below 1: telling a client to retry
    # before capacity exists just earns it an immediate second 429.
    retry_after = int(rejected.headers.get("retry-after", 0))
    check("Retry-After is a usable wait", 1 <= retry_after <= 10, True)


async def phase_global(client: httpx.AsyncClient, base: list[str]) -> None:
    print(f"\napp-wide      /cheap      token_bucket, {GLOBAL_LIMIT} per 10s, "
          f"across {len(base)} process(es)")
    responses = await asyncio.gather(
        *(client.get(f"{base[i % len(base)]}/cheap") for i in range(CONCURRENT))
    )
    codes = collections.Counter(r.status_code for r in responses)
    check(f"admitted from {CONCURRENT} concurrent", codes[200], GLOBAL_LIMIT)
    check("rejected", codes[429], CONCURRENT - GLOBAL_LIMIT)


async def main(phase: str, ports: list[int]) -> int:
    base = [f"http://127.0.0.1:{p}" for p in ports]
    async with httpx.AsyncClient(timeout=30) as client:
        await {"endpoint": phase_endpoint, "global": phase_global}[phase](client, base)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1], [int(p) for p in sys.argv[2:]])))
