"""Derived numbers, computed from the logged records rather than re-run.

Everything the summary tables and plots report comes from here, and here
reads only the JSONL. That separation is the point: a claim in the writeup
can be recomputed from the log on disk without trusting -- or re-running --
the generator that produced it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from loadtest.runner import Record


@dataclass(frozen=True)
class Summary:
    algorithm: str
    client: str
    offered: int
    allowed: int
    rejected: int
    allowed_pct: float
    max_in_window: int
    """The most requests admitted in *any* interval one window long.

    This is the number that actually matters, and it is not the same as the
    configured limit. A limiter can honour "20 per window" on every window it
    defines and still let a client land 40 requests inside a single window's
    worth of time, by splitting them across a boundary it happens to draw
    between them. Counting over a sliding interval ignores where the limiter
    chose to put its boundaries and asks what the *client* experienced.
    """
    over_admission: float
    """max_in_window as a multiple of the configured limit. 1.0 is exact."""
    p50_latency_ms: float
    p99_latency_ms: float
    max_delay: float
    """Largest shaping delay imposed on an admitted request."""


def max_in_sliding_window(admitted: list[float], window: float) -> int:
    """Most admissions falling inside any window-length interval.

    Two pointers over the sorted admission times: for each start, advance the
    end while it stays within `window`. The interval is half open -- an
    admission exactly `window` later belongs to the next interval, matching
    how every algorithm here treats its own boundary.
    """
    if not admitted:
        return 0
    times = sorted(admitted)
    best = 0
    end = 0
    for start in range(len(times)):
        if end < start:
            end = start
        while end < len(times) and times[end] - times[start] < window:
            end += 1
        best = max(best, end - start)
    return best


def summarize(
    records: list[Record],
    *,
    limit: int,
    window: float,
    by_client: bool = False,
) -> list[Summary]:
    """One row per algorithm, or per (algorithm, client) when `by_client`."""
    groups: dict[tuple[str, str], list[Record]] = {}
    for r in records:
        key = (r.algorithm, r.client if by_client else "*")
        groups.setdefault(key, []).append(r)

    rows: list[Summary] = []
    for (algorithm, client), group in groups.items():
        allowed = [r for r in group if r.allowed]
        latencies = [r.latency_ms for r in group]

        # Peak is computed per client and then maximised, never pooled. Quota
        # is per key, so pooling several clients' admissions would add up
        # independent budgets and report the total as over-admission -- three
        # clients each perfectly within a limit of 20 would look like 3x.
        # What the aggregate row should say is how badly the *worst* single
        # client was over-served.
        #
        # An admitted request counts at the moment it may proceed, which for a
        # shaper is after its delay: that delay is the entire mechanism by
        # which it stays under the limit.
        per_client: dict[str, list[float]] = {}
        for r in allowed:
            if r.admitted is not None:
                per_client.setdefault(r.client, []).append(r.admitted)
        peak = max(
            (max_in_sliding_window(times, window) for times in per_client.values()),
            default=0,
        )

        rows.append(
            Summary(
                algorithm=algorithm,
                client=client,
                offered=len(group),
                allowed=len(allowed),
                rejected=len(group) - len(allowed),
                allowed_pct=100.0 * len(allowed) / len(group) if group else 0.0,
                max_in_window=peak,
                over_admission=peak / limit,
                p50_latency_ms=statistics.median(latencies) if latencies else 0.0,
                p99_latency_ms=(
                    sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.99))]
                    if latencies
                    else 0.0
                ),
                max_delay=max((r.delay for r in group), default=0.0),
            )
        )

    rows.sort(key=lambda s: (s.algorithm, s.client))
    return rows


def format_table(rows: list[Summary], *, by_client: bool = False) -> str:
    """A plain text table, for the terminal and for pasting into notes."""
    headers = ["algorithm"]
    if by_client:
        headers.append("client")
    headers += ["offered", "allowed", "rejected", "allowed%", "max/win", "xlimit", "p50ms", "p99ms"]

    body = []
    for r in rows:
        cells = [r.algorithm]
        if by_client:
            cells.append(r.client)
        cells += [
            str(r.offered),
            str(r.allowed),
            str(r.rejected),
            f"{r.allowed_pct:.1f}",
            str(r.max_in_window),
            f"{r.over_admission:.2f}",
            f"{r.p50_latency_ms:.3f}",
            f"{r.p99_latency_ms:.3f}",
        ]
        body.append(cells)

    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(lines)
