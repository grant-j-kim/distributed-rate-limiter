"""Tests for the measurement rig itself.

The rig is not shipped, but every number in the writeup comes out of it, so a
bug here is indistinguishable from a finding. Two things in particular are
worth pinning down: the sliding-interval peak, which is the metric the whole
comparison rests on, and the runner's handling of shaping delay, where a
mistake would quietly turn the leaky bucket into a meter and report the
result as an algorithmic property.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from distributed_rate_limiter.base import Decision, RateLimiter
from loadtest.analysis import max_in_sliding_window, summarize
from loadtest.runner import Record, run_schedule
from loadtest.traffic import burst, merge, poisson, steady


# --------------------------------------------------------------------------
# The sliding-interval peak
# --------------------------------------------------------------------------


def test_peak_is_zero_without_admissions():
    assert max_in_sliding_window([], 2.0) == 0


def test_peak_counts_a_simultaneous_burst():
    assert max_in_sliding_window([1.0] * 30, 2.0) == 30


def test_peak_interval_is_half_open():
    """An admission exactly one window later starts the next interval.

    Every algorithm here treats its own boundary as half open, so a metric
    that counted `window` seconds later as still inside would report 1.05x for
    a limiter behaving exactly to spec.
    """
    assert max_in_sliding_window([0.0, 2.0], 2.0) == 1
    assert max_in_sliding_window([0.0, 1.999], 2.0) == 2


def test_peak_finds_a_straddling_burst_that_fixed_bins_would_miss():
    """The reason the metric slides instead of bucketing by window index.

    20 admissions just before a boundary and 20 just after sit in different
    fixed bins, so per-window counting reports 20 and 20 -- both within the
    limit, no problem visible. The client experienced 40 inside a single
    window's worth of time. That gap is precisely the fixed window's flaw,
    and a metric that bucketed by the limiter's own boundaries would be
    incapable of showing it.
    """
    admitted = [1.9] * 20 + [2.1] * 20

    by_fixed_bin: dict[int, int] = {}
    for t in admitted:
        by_fixed_bin[int(t // 2.0)] = by_fixed_bin.get(int(t // 2.0), 0) + 1
    assert max(by_fixed_bin.values()) == 20  # what naive bucketing would say

    assert max_in_sliding_window(admitted, 2.0) == 40


def test_peak_ignores_admissions_spread_beyond_the_window():
    # One per second over ten seconds, window 2s: never more than two at once.
    assert max_in_sliding_window([float(i) for i in range(10)], 2.0) == 2


def test_peak_is_order_independent():
    scrambled = [5.0, 0.1, 3.2, 0.2, 4.9, 0.3]
    assert max_in_sliding_window(scrambled, 1.0) == 3


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _record(algorithm="a", client="c", sent=0.0, allowed=True, delay=0.0) -> Record:
    return Record(
        algorithm=algorithm,
        client=client,
        scheduled=sent,
        sent=sent,
        allowed=allowed,
        remaining=0,
        retry_after=None if allowed else 1.0,
        delay=delay,
        admitted=sent + delay if allowed else None,
        latency_ms=0.5,
    )


def test_peak_never_pools_separate_clients():
    """Quota is per key, so admissions from different clients must not add.

    Three clients each sitting exactly on a limit of 20 is three limiters
    behaving perfectly. Pooling them would report 60 -- a 3x over-admission
    that never happened, and one that would grow with the number of clients
    in the scenario rather than with anything the limiter did.
    """
    records = [
        _record(client=client, sent=0.1 * i)
        for client in ("a", "b", "c")
        for i in range(20)
    ]
    rows = summarize(records, limit=20, window=2.0)

    assert len(rows) == 1
    assert rows[0].max_in_window == 20
    assert rows[0].over_admission == 1.0


def test_summary_counts_rejections_and_rates():
    records = [_record(allowed=True)] * 3 + [_record(allowed=False)]
    (row,) = summarize(records, limit=20, window=2.0)

    assert (row.offered, row.allowed, row.rejected) == (4, 3, 1)
    assert row.allowed_pct == 75.0


def test_shaped_admissions_count_when_they_proceed_not_when_they_arrive():
    """A delayed request is under the limit *because* of the delay.

    Ten requests arriving at once but paced half a second apart occupy five
    seconds, so no two-second interval holds more than four. Counting them at
    arrival would report all ten at once and make the shaper look like the
    worst offender in the comparison rather than the best.
    """
    records = [_record(sent=0.0, delay=0.5 * i) for i in range(10)]
    (row,) = summarize(records, limit=20, window=2.0)

    assert row.max_in_window == 4
    assert row.max_delay == pytest.approx(4.5)


# --------------------------------------------------------------------------
# Traffic schedules
# --------------------------------------------------------------------------


def test_steady_is_evenly_spaced_at_the_requested_rate():
    arrivals = steady(rate=20.0, duration=10.0)
    assert len(arrivals) == 200
    gaps = [b.offset - a.offset for a, b in zip(arrivals, arrivals[1:])]
    assert gaps == pytest.approx([0.05] * 199)


def test_burst_shares_one_offset():
    arrivals = burst(1.85, 30)
    assert len(arrivals) == 30
    assert {a.offset for a in arrivals} == {1.85}


def test_poisson_is_reproducible_for_a_seed():
    """A schedule that changed between runs would make runs incomparable.

    The same reason all five algorithms are handed one schedule within a run
    applies across runs: without it, a difference between today's numbers and
    last week's could be the traffic rather than the code.
    """
    first = poisson(rate=10.0, duration=5.0, seed=7)
    second = poisson(rate=10.0, duration=5.0, seed=7)
    other = poisson(rate=10.0, duration=5.0, seed=8)

    assert [a.offset for a in first] == [a.offset for a in second]
    assert [a.offset for a in first] != [a.offset for a in other]


def test_poisson_stays_inside_its_duration():
    arrivals = poisson(rate=50.0, duration=3.0, seed=1)
    assert all(0.0 <= a.offset < 3.0 for a in arrivals)


def test_merge_orders_a_multi_client_timeline():
    merged = merge(
        steady(rate=2.0, duration=1.0, client="a"),
        steady(rate=4.0, duration=1.0, client="b"),
    )
    assert [a.offset for a in merged] == sorted(a.offset for a in merged)
    assert {a.client for a in merged} == {"a", "b"}


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_invalid_rates_are_rejected(rate):
    with pytest.raises(ValueError):
        steady(rate=rate, duration=1.0)
    with pytest.raises(ValueError):
        poisson(rate=rate, duration=1.0, seed=1)


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


class StubLimiter(RateLimiter):
    """Answers instantly, or after a configured stall, and records call order."""

    def __init__(self, *, delay: float = 0.0, stall: float = 0.0):
        self.limit = 100
        self.window = 1.0
        self._delay = delay
        self._stall = stall
        self.calls: list[float] = []

    async def check(self, key: str) -> Decision:
        self.calls.append(time.time())
        if self._stall:
            await asyncio.sleep(self._stall)
        return Decision(
            allowed=True, limit=self.limit, remaining=0,
            reset_after=0.0, delay=self._delay,
        )

    async def reset(self, key: str) -> None:  # pragma: no cover - unused
        pass


async def test_offered_load_is_unaffected_by_a_slow_limiter():
    """The schedule must not bend to how the limiter is coping.

    Each arrival sleeps to its own deadline in its own task, so a limiter
    taking 100ms per check cannot push later arrivals. Awaiting checks in a
    loop instead would stretch a 0.3s schedule into 1.2s -- and it would
    stretch it *more* the harder the limiter was struggling, so the offered
    load would silently drop exactly when the results mattered most.
    """
    limiter = StubLimiter(stall=0.1)
    arrivals = [a for i in range(4) for a in burst(i * 0.1, 1)]

    t0 = time.time() + 0.05
    records = await run_schedule(limiter, "stub", arrivals, t0=t0)

    by_schedule = sorted(records, key=lambda r: r.scheduled)
    for record in by_schedule:
        assert record.sent == pytest.approx(record.scheduled, abs=0.05), (
            "arrival drifted from its scheduled offset"
        )


async def test_a_concurrent_burst_is_issued_concurrently():
    limiter = StubLimiter(stall=0.05)
    records = await run_schedule(
        limiter, "stub", burst(0.0, 20), t0=time.time() + 0.05
    )

    assert len(records) == 20
    assert max(r.sent for r in records) - min(r.sent for r in records) < 0.05


async def test_runner_waits_out_a_shaping_delay():
    """The shaper's contract, honoured by the generator.

    A load generator that recorded the admission and skipped the wait would
    be driving a meter while labelling the results a shaper's.
    """
    limiter = StubLimiter(delay=0.2)
    started = time.time()
    records = await run_schedule(
        limiter, "stub", burst(0.0, 1), t0=started
    )

    elapsed = time.time() - started
    assert elapsed >= 0.2, "shaping delay was not awaited"
    assert records[0].admitted == pytest.approx(records[0].sent + 0.2)


async def test_rejected_requests_have_no_admission_time():
    class Denier(StubLimiter):
        async def check(self, key: str) -> Decision:
            return Decision(
                allowed=False, limit=1, remaining=0,
                reset_after=1.0, retry_after=1.0,
            )

    records = await run_schedule(Denier(), "stub", burst(0.0, 3), t0=time.time())
    assert all(r.admitted is None for r in records)
    assert all(not r.allowed for r in records)
