"""The playground endpoint.

Thin over `web.replay`, so these pin the contract the page depends on rather
than re-testing the algorithms: that the slider actually changes the answer,
that the response carries what the timeline needs to draw, and that the
endpoint costs no Redis.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from web.api import ALGORITHMS, LIMIT, WINDOW, app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_default_center_straddles_a_boundary(client):
    """The page's landing state must be the interesting one.

    Round-tripping the default is what a visitor sees before touching
    anything, so it has to show the finding rather than a flat comparison.
    """
    body = (await client.get("/api/replay")).json()
    by_algo = {r["algorithm"]: r for r in body["results"]}

    assert by_algo["fixed_window"]["ratio"] == 2.0
    assert by_algo["sliding_window_log"]["ratio"] == 1.0


async def test_sliding_the_pair_off_the_boundary_collapses_the_difference(client):
    """Mid-window, the fixed window is exact -- the failure is positional.

    This is the hero slider's whole payload: the same sixty requests are a 2.00x
    breach in one place and perfectly compliant in another, and nothing about
    the algorithm changed.
    """
    straddling = (await client.get("/api/replay", params={"center": WINDOW})).json()
    mid_window = (await client.get("/api/replay", params={"center": 1.0})).json()

    def ratio(body, algo):
        return next(r for r in body["results"] if r["algorithm"] == algo)["ratio"]

    assert ratio(straddling, "fixed_window") == 2.0
    assert ratio(mid_window, "fixed_window") == 1.0
    # The sliding log is positional-invariant: that contrast is the point.
    assert ratio(straddling, "sliding_window_log") == 1.0
    assert ratio(mid_window, "sliding_window_log") == 1.0


async def test_response_carries_what_the_timeline_draws(client):
    """Every mark needs a time, an outcome and a delay, and every row a
    boundary set to draw against."""
    body = (await client.get("/api/replay")).json()

    assert [r["algorithm"] for r in body["results"]] == list(ALGORITHMS)
    assert body["limit"] == LIMIT
    assert WINDOW in body["boundaries"]

    for result in body["results"]:
        assert len(result["marks"]) == body["offered"]
        for mark in result["marks"]:
            assert set(mark) == {"t", "ok", "d"}

    shaping = [r["algorithm"] for r in body["results"] if r["shapes"]]
    assert shaping == ["leaky_bucket"], "only the shaper should report a delay"


async def test_center_is_bounded(client):
    """A center outside the run puts bursts at negative offsets, which the
    replay refuses -- reject it at the edge instead of raising inside."""
    assert (await client.get("/api/replay", params={"center": 0.0})).status_code == 422
    assert (await client.get("/api/replay", params={"center": 99})).status_code == 422


async def test_root_serves_the_page(client):
    """`/` must return the page the CDN also holds.

    public/ is read at runtime rather than mounted, so an excludeFiles glob
    that dropped it would deploy a working API behind a 500 -- and only at the
    root, which is the one path a visitor actually opens.
    """
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/api/replay?center=" in response.text, "page must call the endpoint"
