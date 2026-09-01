"""The playground's replay rig.

Same reasoning as `test_loadtest.py`: a bug in the replay is indistinguishable
from a finding. A visitor who sees the fixed window admit 2.4x its limit cannot
tell whether that is the algorithm or a mistake in the loop that drove it, and
neither can we. So the claims the page makes are pinned here.
"""

from __future__ import annotations

import time

import pytest

from loadtest.analysis import max_in_sliding_window
from loadtest.traffic import Arrival, burst, merge, steady
from web.replay import build, replay

LIMIT = 20
WINDOW = 2.0

# The boundary_burst scenario from loadtest/: 30 requests just before a window
# boundary, 30 just after, 0.3s apart.
BOUNDARY_BURST = merge(burst(1.85, 30), burst(2.15, 30))


def admitted_offsets(outcomes):
    return [o.offset for o in outcomes if o.allowed]


async def test_fixed_window_admits_twice_its_limit_across_a_boundary():
    """The headline claim: 40 admitted against a limit of 20, in 0.3 seconds.

    This is the number the page is built to show. It is structural rather than
    measured -- the fixed window has no memory across a boundary, so both
    bursts get a full fresh allowance -- so the replay must reproduce it
    exactly, not approximately.
    """
    limiter = build("fixed_window", LIMIT, WINDOW)
    outcomes = await replay(BOUNDARY_BURST, limiter)

    allowed = admitted_offsets(outcomes)
    assert len(allowed) == 40
    peak = max_in_sliding_window(allowed, WINDOW)
    assert peak == 40
    assert peak / LIMIT == 2.0


async def test_sliding_window_log_is_exact_on_the_same_schedule():
    """1.00x on the schedule that makes the fixed window 2.00x.

    Replaying one schedule against both is the whole comparison; if the log
    were not exact here, the contrast the page is built around would be an
    artefact of the rig rather than a property of the algorithms.
    """
    limiter = build("sliding_window_log", LIMIT, WINDOW)
    outcomes = await replay(BOUNDARY_BURST, limiter)

    peak = max_in_sliding_window(admitted_offsets(outcomes), WINDOW)
    assert peak == LIMIT


async def test_replay_consumes_no_wall_clock_time():
    """A ten second scenario must resolve in microseconds, not ten seconds.

    This is the property the whole split architecture rests on: the playground
    is free and instant because no time passes and no Redis is touched. If a
    limiter ever reached for the real clock, this is where it would show up.
    """
    schedule = steady(rate=20, duration=10.0)
    assert len(schedule) == 200

    started = time.perf_counter()
    await replay(schedule, build("token_bucket", LIMIT, WINDOW))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"replay took {elapsed:.3f}s of real time"


async def test_shaping_delay_is_recorded_not_simulated():
    """The leaky bucket defers requests; the replay must not defer the clock.

    Advancing the clock by Decision.delay would fast forward past arrivals that
    happen during the deferral. So every outcome keeps the offset it arrived at,
    and carries the delay separately -- which is what lets the timeline draw a
    shaped request at both its arrival and its departure.
    """
    limiter = build("leaky_bucket", LIMIT, WINDOW)
    outcomes = await replay(BOUNDARY_BURST, limiter)

    assert [o.offset for o in outcomes] == [a.offset for a in BOUNDARY_BURST]

    shaped = [o for o in outcomes if o.allowed and o.delay > 0]
    assert shaped, "the leaky bucket admitted nothing with a shaping delay"
    for o in shaped:
        assert o.departure > o.offset
        assert o.departure == pytest.approx(o.offset + o.delay)


async def test_metering_algorithms_never_report_a_delay():
    """Only the shaper shapes. A stray delay would be drawn as a phantom
    departure mark on four rows that do not defer anything."""
    for algorithm in ("fixed_window", "sliding_window_log",
                      "sliding_window_counter", "token_bucket"):
        outcomes = await replay(BOUNDARY_BURST, build(algorithm, LIMIT, WINDOW))
        assert all(o.delay == 0.0 for o in outcomes), algorithm


async def test_clients_hold_independent_quota():
    """A greedy client must not spend a polite one's allowance.

    The page shows several keys on one timeline, so a limiter leaking between
    keys would look like an algorithm being harsh rather than a bug.
    """
    schedule = merge(
        steady(rate=5, duration=4.0, client="polite"),
        steady(rate=25, duration=4.0, client="greedy"),
    )
    outcomes = await replay(schedule, build("sliding_window_log", LIMIT, WINDOW))

    polite = [o for o in outcomes if o.client == "polite"]
    assert all(o.allowed for o in polite)
    assert any(not o.allowed for o in outcomes if o.client == "greedy")


async def test_out_of_order_schedule_is_refused():
    """A clock that goes backwards produces silent nonsense, not an error.

    The limiters compute elapsed time by subtraction, so a late arrival with an
    early offset yields negative refill and no exception. Refuse the schedule.
    """
    backwards = [Arrival(1.0, "c"), Arrival(0.5, "c")]
    with pytest.raises(ValueError, match="ordered by offset"):
        await replay(backwards, build("token_bucket", LIMIT, WINDOW))


async def test_limiter_on_an_undriven_clock_is_refused():
    """create_limiter defaults the memory backend to the real wall clock.

    Handed such a limiter, the replay would set a `now` nobody reads, every
    arrival would land at the same real instant, and the page would show a
    plausible-looking wrong answer. Fail loudly instead.
    """
    from distributed_rate_limiter import create_limiter

    wall_clock_limiter = create_limiter("token_bucket", limit=LIMIT, window=WINDOW)
    with pytest.raises(TypeError, match="does not control"):
        await replay(BOUNDARY_BURST, wall_clock_limiter)


async def test_peak_counts_a_shaped_request_at_departure_not_arrival():
    """The leaky bucket measures 1.05x, not 1.15x, and the gap is the shaping.

    Counting admissions where they *arrived* ignores the delay entirely and
    reports the shaper as though it were another meter. This is the same
    decision loadtest/runner.py makes when it logs `sent + delay`, and the two
    must agree or the page will contradict the published table.
    """
    from web.replay import peak_admission

    schedule = merge(
        burst(WINDOW - 0.15, 30, client="a"),
        burst(WINDOW + 0.15, 30, client="a"),
        burst(WINDOW * 3.5, 30, client="a"),
    )
    outcomes = await replay(schedule, build("leaky_bucket", LIMIT, WINDOW))

    assert peak_admission(outcomes, WINDOW) == 21          # 1.05x, as measured
    naive = max_in_sliding_window(admitted_offsets(outcomes), WINDOW)
    assert naive == 23, "the arrival-time metric should over-report the shaper"


async def test_peak_is_per_key_then_maximised_never_pooled():
    """Three clients each exactly at their limit is 1.00x, not 3.00x.

    Pooling admissions across keys adds up independent quotas and reports
    well-behaved clients as a breach. The page draws several keys at once, so
    this is the metric bug most likely to reach a visitor.
    """
    from web.replay import peak_admission

    schedule = merge(*(burst(0.5, LIMIT, client=f"c{i}") for i in range(3)))
    outcomes = await replay(schedule, build("fixed_window", LIMIT, WINDOW))

    assert sum(1 for o in outcomes if o.allowed) == 3 * LIMIT
    assert peak_admission(outcomes, WINDOW) == LIMIT


async def test_token_bucket_boundary_burst_sits_on_a_knife_edge():
    """22 or 23 depending on a tenth of a millisecond -- do not "fix" this.

    Refill is limit/window = 10 tokens/s and the bursts are 0.30s apart, so the
    third token arrives exactly when the second burst does. The replay is
    exact, so it always lands on the 3-token side and reports 1.15x; the run
    logged in loadtest/results/ dispatched 1.1ms early and reported 1.10x.
    Both are correct.

    This lives with the replay rather than in test_token_bucket.py because the
    risk it guards against is someone adjusting the *rig* to reproduce the
    published 22, which would mean introducing an error to match a sample.
    """
    from web.replay import peak_admission

    def schedule_with_second_burst_at(t):
        return merge(burst(1.85, 30, client="a"), burst(t, 30, client="a"),
                     burst(7.0, 30, client="a"))

    just_under = await replay(schedule_with_second_burst_at(2.1499),
                              build("token_bucket", LIMIT, WINDOW))
    exact = await replay(schedule_with_second_burst_at(2.1500),
                         build("token_bucket", LIMIT, WINDOW))

    assert peak_admission(just_under, WINDOW) == 22   # 2.999 tokens -> 1.10x
    assert peak_admission(exact, WINDOW) == 23        # 3.000 tokens -> 1.15x
