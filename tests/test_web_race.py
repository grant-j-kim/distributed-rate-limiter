"""The punchline: rationing, run tokens, and the recorded fallback.

The race itself is already proven in `test_redis_concurrency.py`, where the
same naive implementation is kept as a control that must keep failing. These
cover what the *demo* adds around it: that an unpaid-for run cannot be fired,
that the fallback is honest about being a fallback, and that the committed
recording is a real result rather than a plausible-looking file.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from web import race
from web.api import app

RECORDING = pathlib.Path("web/punchline_recording.json")


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_run_token_round_trips(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid/0")
    run_id, expires_at, token = race.new_run(0.0)

    assert race.verify_token(run_id, expires_at, token)
    assert not race.verify_token(run_id, expires_at, "0" * 32)
    assert not race.verify_token("other-run", expires_at, token)
    # The barrier instant is signed too, so a caller cannot move its own
    # firing time and quietly spread the volley back out.
    assert not race.verify_token(run_id, expires_at, token, start_at=99.0)


async def test_expired_token_is_refused(monkeypatch):
    """A token is only good for its window, so a leaked one decays to useless."""
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid/0")
    run_id = uuid.uuid4().hex[:16]
    stale = time.time() - 1

    assert not race.verify_token(run_id, stale, race.issue_token(run_id, stale))


async def test_firing_without_a_valid_token_is_refused(client, monkeypatch):
    """Rationing is charged once per run, at /start, so /fire must not be open.

    If a forged fire were served, the per-IP limit and the monthly budget would
    both be decoration: anyone could spend the whole free tier by skipping the
    endpoint that does the charging.
    """
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid/0")
    response = await client.get("/api/race/fire", params={
        "run": "deadbeef", "exp": time.time() + 60, "start": 0.0,
        "token": "0" * 32, "variant": "lua"})

    assert response.status_code == 403


async def test_without_redis_the_fallback_says_so(client):
    """Absence of REDIS_URL is the switch that makes previews serve a recording.

    The response must be explicit about it. A page about correctness that
    silently swapped a recording in for a live result would be the one
    dishonesty that matters here.
    """
    body = (await client.post("/api/race/start")).json()

    assert body["live"] is False
    assert body["reason"] == "no-redis"
    assert body["recording"]["lua_admitted"] == race.RACE_LIMIT


async def test_the_committed_recording_is_a_real_race():
    """The fallback must show a race that actually happened.

    Same control logic as the concurrency suite, applied to the artefact: if
    the naive limiter in the recording did not over-admit, the file is showing
    two limiters agreeing and demonstrates nothing.
    """
    record = json.loads(RECORDING.read_text())

    assert record["lua_admitted"] == record["limit"]
    assert record["naive_admitted"] > record["limit"]
    assert record["recorded_at"] and record["redis_version"] != "unknown"
    assert len(record["naive"]) == len(record["lua"]) == record["concurrency"]


async def test_code_panel_reads_the_running_source(client):
    """The page shows source read by inspect, not a paste that can drift."""
    body = (await client.get("/api/race/code")).json()

    assert "self.client.zcount(" in body["naive"]
    assert "self.client.zadd(" in body["naive"]
    assert "redis.call" in body["lua"]


async def test_naive_control_over_admits_against_real_redis(redis_client):
    """The exhibit copy must fail the same way the test control does.

    web/race.py deliberately duplicates the naive control rather than
    importing it, so the two can drift. This is the assertion that catches a
    drift that would matter: an exhibit that stopped over-admitting would leave
    the demo showing nothing.
    """
    if redis_client is None:
        pytest.skip("no Redis server")

    now = (await redis_client.time())[0]
    naive = race.NaiveRedisSlidingWindowLog(
        redis_client, limit=5, window=30.0,
        prefix=f"racetest:{uuid.uuid4().hex}", now=now)
    decisions = await asyncio.gather(*(naive.check("c") for _ in range(50)))

    assert sum(1 for d in decisions if d.allowed) > 5


async def test_fire_reports_the_two_numbers_that_decide_the_outcome(monkeypatch, redis_client):
    """Every fire carries a shared-clock timestamp and its own round trip.

    Without both, an admission count is uninterpretable: a read-modify-write
    limiter is only visibly wrong while requests overlap its gap, and that gap
    is one round trip wide. The timestamp comes from Redis rather than the
    process, because the fires may land on different instances and comparing
    their local clocks at millisecond scale would measure skew as stagger.
    """
    if redis_client is None:
        pytest.skip("no Redis server")

    from tests.conftest import REDIS_URL as TEST_REDIS_URL
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)

    # Clear the rationing counters first. They are real limiter state with a
    # one-hour window, so without this the test quietly starts *skipping* after
    # three runs -- passing by not running, which is the worst way to fail.
    async for key in redis_client.scan_iter(match="race:quota:*"):
        await redis_client.delete(key)
    race._CACHED = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        start = (await c.post("/api/race/start")).json()
        assert start.get("live"), f"race not live: {start.get('reason')}"

        body = (await c.get("/api/race/fire", params={
            "run": start["run"], "exp": start["exp"], "start": start["start"],
            "token": start["token"], "variant": "lua"})).json()

    assert body["allowed"] is True
    assert body["t"] > 1_700_000_000, "timestamp should be UNIX seconds from Redis"
    assert body["rtt_ms"] >= 0


async def test_probe_reports_instance_identity(client):
    """The probe must cost nothing and identify its instance.

    It exists because two explanations for the race failing to overlap were
    guessed and both were wrong. Counting distinct instance ids across a volley
    is the direct measurement: one id means the platform serialised the
    requests and no barrier can assemble them, many means the fan-out works.
    It touches no Redis, so it stays free to run and needs no rationing.
    """
    first = (await client.get("/api/race/ping")).json()
    second = (await client.get("/api/race/ping")).json()

    assert first["instance"] == second["instance"], "same process, same id"
    assert len(first["instance"]) == 8
    assert second["age"] >= first["age"], "age should increase within one instance"
