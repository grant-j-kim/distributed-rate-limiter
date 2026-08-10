"""Figures. One renderer per scenario, each answering that scenario's question.

Reads only `Record`s, so every figure can be rebuilt from the JSONL without
re-running any traffic (`python -m loadtest --plot-only`).

Window boundaries are drawn on every time axis. They are where the algorithms
disagree, and a plot of this data without them is nearly unreadable -- the
fixed window's steps look arbitrary until you can see they land exactly on
the boundaries it draws.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in this environment, and none needed

import matplotlib.pyplot as plt  # noqa: E402

from loadtest.analysis import summarize  # noqa: E402
from loadtest.runner import Record  # noqa: E402

# Fixed per algorithm, so the same colour means the same algorithm in every
# figure. Ordered light to dark within the two families: windows warm,
# buckets cool.
COLOURS = {
    "fixed_window": "#d1495b",
    "sliding_window_log": "#edae49",
    "sliding_window_counter": "#f79256",
    "token_bucket": "#00798c",
    "leaky_bucket": "#30638e",
}
LABELS = {
    "fixed_window": "fixed window",
    "sliding_window_log": "sliding window log",
    "sliding_window_counter": "sliding window counter",
    "token_bucket": "token bucket",
    "leaky_bucket": "leaky bucket",
}

# Matplotlib stamps a "Software: Matplotlib version X.Y.Z" tEXt chunk into
# every PNG by default. Setting it to None suppresses the chunk, so a
# committed figure records what was measured and not which toolchain drew it.
# Keeps the images byte-reproducible across matplotlib upgrades too: without
# this, bumping the library changes all three files with no visual difference.
PNG_METADATA = {"Software": None}


def _boundaries(ax, window: float, duration: float) -> None:
    n = int(duration / window) + 1
    for i in range(n + 1):
        ax.axvline(i * window, color="0.75", linestyle=":", linewidth=0.9, zorder=0)


def _by_algorithm(records: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        grouped[r.algorithm].append(r)
    for group in grouped.values():
        group.sort(key=lambda r: r.sent)
    return grouped


def render(name: str, records: list[Record], meta: dict, outdir: Path) -> list[Path]:
    renderers = {
        "steady_over_limit": _render_steady,
        "boundary_burst": _render_burst,
        "multi_client": _render_multi_client,
    }
    return renderers[name](records, meta, outdir)


def _render_steady(records: list[Record], meta: dict, outdir: Path) -> list[Path]:
    """Same allowance, different distribution in time."""
    window, limit, duration = meta["window"], meta["limit"], meta["duration"]
    grouped = _by_algorithm(records)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[3, 2]
    )

    # Cumulative admissions. The ideal line is the sustainable rate: any
    # algorithm tracking it is converting a bursty allowance into a smooth
    # one, and any staircase above it is admitting in clumps.
    offered = sorted(r.sent for r in records if r.algorithm == meta["algorithms"][0])
    top.step(
        offered, range(1, len(offered) + 1),
        where="post", color="0.6", linestyle="--", linewidth=1.2,
        label=f"offered ({len(offered)} requests)",
    )
    top.plot(
        [0, duration], [0, duration * limit / window],
        color="black", linestyle=":", linewidth=1.4,
        label=f"sustainable rate ({limit / window:.0f}/s)",
    )
    for algorithm in meta["algorithms"]:
        group = grouped.get(algorithm, [])
        admitted = sorted(r.admitted for r in group if r.admitted is not None)
        top.step(
            admitted, range(1, len(admitted) + 1),
            where="post", color=COLOURS[algorithm], linewidth=1.8,
            label=LABELS[algorithm],
        )
    _boundaries(top, window, duration)
    top.set_ylabel("cumulative admitted")
    top.set_title(
        f"Steady demand at {2 * limit / window:.0f}/s against a limit of "
        f"{limit} per {window:g}s\n"
        "identical traffic replayed against all five algorithms",
        fontsize=11,
    )
    top.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Admission rate in fine bins: where the clumping is actually visible.
    bin_width = window / 8
    for algorithm in meta["algorithms"]:
        group = grouped.get(algorithm, [])
        counts = Counter(
            int(r.admitted / bin_width) for r in group if r.admitted is not None
        )
        xs = [b * bin_width for b in range(int(duration / bin_width) + 1)]
        ys = [counts.get(b, 0) for b in range(len(xs))]
        bottom.step(xs, ys, where="post", color=COLOURS[algorithm], linewidth=1.5)
    _boundaries(bottom, window, duration)
    bottom.axhline(
        limit * bin_width / window, color="black", linestyle=":", linewidth=1.4,
    )
    bottom.set_ylabel(f"admitted per {bin_width:g}s")
    bottom.set_xlabel("seconds from a window boundary")
    bottom.set_xlim(0, duration)

    fig.tight_layout()
    path = outdir / "steady_over_limit.png"
    fig.savefig(path, dpi=140, metadata=PNG_METADATA)
    plt.close(fig)
    return [path]


def _render_burst(records: list[Record], meta: dict, outdir: Path) -> list[Path]:
    """The boundary straddle, request by request."""
    window, limit, duration = meta["window"], meta["limit"], meta["duration"]
    grouped = _by_algorithm(records)
    order = meta["algorithms"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 9), height_ratios=[3, 2]
    )

    # Each request is a marker. Requests arriving together are stacked
    # vertically within their algorithm's band, so the height of a column is
    # literally how many that algorithm admitted at that instant.
    for row, algorithm in enumerate(order):
        group = grouped.get(algorithm, [])
        ranks: Counter[int] = Counter()
        for r in sorted(group, key=lambda r: (r.sent, not r.allowed)):
            t = r.admitted if r.admitted is not None else r.sent
            slot = int(t / 0.2)
            rank = ranks[slot]
            ranks[slot] += 1
            y = row - 0.34 + 0.022 * rank
            if r.allowed:
                top.plot(t, y, marker="o", markersize=3.2,
                         color=COLOURS[algorithm], zorder=3)
            else:
                top.plot(t, y, marker="x", markersize=3.2,
                         color="0.72", zorder=2)

    _boundaries(top, window, duration)
    top.set_yticks(range(len(order)))
    top.set_yticklabels([LABELS[a] for a in order])
    top.set_ylim(-0.6, len(order) - 0.2)
    top.set_xlim(0, duration)
    top.set_xlabel("seconds from a window boundary")
    top.set_title(
        f"30 requests at {window - 0.15:g}s and 30 more at {window + 0.15:g}s "
        f"-- 0.3s apart, either side of a boundary\n"
        f"limit {limit} per {window:g}s; coloured = admitted, grey x = rejected; "
        f"third burst at {window * 3.5:g}s is a mid-window control",
        fontsize=11,
    )

    # What the client actually got: the most admissions in any interval one
    # window long, wherever that interval happens to fall.
    rows = {s.algorithm: s for s in summarize(records, limit=limit, window=window)}
    peaks = [rows[a].max_in_window for a in order]
    bars = bottom.bar(
        range(len(order)), peaks,
        color=[COLOURS[a] for a in order], width=0.6,
    )
    bottom.axhline(limit, color="black", linestyle="--", linewidth=1.4)
    bottom.set_xlim(-0.75, len(order) - 0.25)
    bottom.text(
        -0.7, limit + 0.6, f"configured limit ({limit})",
        va="bottom", ha="left", fontsize=9,
    )
    for bar, peak in zip(bars, peaks):
        bottom.text(
            bar.get_x() + bar.get_width() / 2, peak + 0.4,
            f"{peak}  ({peak / limit:.2f}x)",
            ha="center", fontsize=9,
        )
    bottom.set_xticks(range(len(order)))
    bottom.set_xticklabels([LABELS[a].replace(" ", "\n", 1) for a in order], fontsize=9)
    bottom.set_ylabel("max admitted in any\none-window interval")
    bottom.set_ylim(0, max(peaks) * 1.25)

    fig.tight_layout()
    path = outdir / "boundary_burst.png"
    fig.savefig(path, dpi=140, metadata=PNG_METADATA)
    plt.close(fig)
    return [path]


def _render_multi_client(records: list[Record], meta: dict, outdir: Path) -> list[Path]:
    """Isolation: the greedy client's rejections must stay its own."""
    window, limit, duration = meta["window"], meta["limit"], meta["duration"]
    order = meta["algorithms"]
    clients = meta["clients"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[3, 4])

    rows = {
        (s.algorithm, s.client): s
        for s in summarize(records, limit=limit, window=window, by_client=True)
    }
    width = 0.8 / len(order)
    for i, algorithm in enumerate(order):
        left.bar(
            [c + i * width - 0.4 for c in range(len(clients))],
            [rows[(algorithm, client)].allowed_pct for client in clients],
            width=width, color=COLOURS[algorithm], label=LABELS[algorithm],
        )
    left.set_xticks(range(len(clients)))
    left.set_xticklabels(
        [
            f"{client}\n{rows[(order[0], client)].offered / duration:.0f}/s offered"
            for client in clients
        ],
        fontsize=9,
    )
    left.set_ylabel("% of that client's requests admitted")
    left.set_ylim(0, 128)
    left.legend(fontsize=7.5, loc="upper center", ncol=3, framealpha=0.95)
    left.set_title("Per-client admission rate", fontsize=11)

    # The polite client alone: it asks for half the limit, so a limiter with
    # correct per-key isolation admits all of it no matter what the greedy
    # client is doing. Any line falling below `offered` here is collateral
    # damage.
    polite = clients[0]
    offered = sorted(
        r.sent for r in records
        if r.client == polite and r.algorithm == order[0]
    )
    for algorithm in order:
        admitted = sorted(
            r.admitted for r in records
            if r.client == polite and r.algorithm == algorithm and r.admitted is not None
        )
        right.step(
            admitted, range(1, len(admitted) + 1),
            where="post", color=COLOURS[algorithm], linewidth=1.4,
            label=LABELS[algorithm],
        )
    # Drawn last and on top: all six traces coincide exactly, so whichever is
    # plotted last is the only one visible. Putting `offered` there makes the
    # result readable -- every admitted trace lies underneath it.
    right.step(
        offered, range(1, len(offered) + 1),
        where="post", color="0.35", linestyle="--", linewidth=2.2, zorder=5,
        label=f"offered by '{polite}'",
    )
    right.annotate(
        "all five algorithms lie exactly under the offered line:\n"
        f"{len(offered)}/{len(offered)} admitted, zero rejections, on every one",
        xy=(0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7"},
    )
    _boundaries(right, window, duration)
    right.set_xlim(0, duration)
    right.set_xlabel("seconds from a window boundary")
    right.set_ylabel(f"cumulative admitted for '{polite}'")
    right.set_title(
        f"'{polite}' stays under its quota while 'greedy' runs 2.5x over",
        fontsize=11,
    )
    right.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    path = outdir / "multi_client.png"
    fig.savefig(path, dpi=140, metadata=PNG_METADATA)
    plt.close(fig)
    return [path]
