"""The name -> implementation lookup, and the packaged public surface.

Most of this is configuration plumbing, with one exception that earns its own
section: which clock a Redis limiter ends up using. Getting that wrong raises
nothing, fails no other test, and only shows up as two production instances
disagreeing about where a window boundary falls.
"""

from __future__ import annotations

import uuid

import pytest

import distributed_rate_limiter as drl
from distributed_rate_limiter.base import RateLimiter, default_clock
from distributed_rate_limiter.redis_backend.token_bucket import RedisTokenBucket
from distributed_rate_limiter.registry import BACKENDS, create_limiter
from tests.conftest import FakeClock


@pytest.fixture
def redis_or_skip(redis_client):
    if redis_client is None:
        pytest.skip("no Redis server available")
    return redis_client


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


def test_every_backend_offers_every_algorithm():
    """The backends must stay interchangeable by name.

    An application selects `backend` from configuration; if one backend were
    missing an algorithm, moving from memory to redis would fail at startup
    for some configurations and not others.
    """
    names = [set(table) for table in BACKENDS.values()]
    assert all(n == names[0] for n in names), "backends offer different algorithms"
    assert names[0] == {
        "fixed_window",
        "sliding_window_log",
        "sliding_window_counter",
        "token_bucket",
        "leaky_bucket",
    }


@pytest.mark.parametrize("algorithm", sorted(BACKENDS["memory"]))
def test_memory_backend_builds_every_algorithm(algorithm):
    limiter = create_limiter(algorithm, limit=5, window=60.0)
    assert isinstance(limiter, RateLimiter)
    assert limiter.limit == 5


@pytest.mark.parametrize("algorithm", sorted(BACKENDS["redis"]))
def test_redis_backend_builds_every_algorithm(redis_or_skip, algorithm):
    limiter = create_limiter(
        algorithm, limit=5, window=60.0,
        backend="redis", client=redis_or_skip, prefix=f"drltest:{uuid.uuid4().hex}",
    )
    assert isinstance(limiter, RateLimiter)
    assert limiter.limit == 5


def test_memory_is_the_default_backend():
    assert type(create_limiter("fixed_window", 5, 60.0)) is BACKENDS["memory"][
        "fixed_window"
    ]


def test_unknown_backend_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown backend .*memory, redis"):
        create_limiter("fixed_window", 5, 60.0, backend="postgres")


def test_unknown_algorithm_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown algorithm"):
        create_limiter("tokenbucket", 5, 60.0)


def test_redis_backend_without_a_client_says_so():
    """A missing client is the likeliest mistake, so it gets a real message.

    Left to the constructor this surfaces as a TypeError about a positional
    argument, which reads like a bug in the library rather than a gap in the
    caller's configuration.
    """
    with pytest.raises(ValueError, match="redis backend needs a client"):
        create_limiter("fixed_window", 5, 60.0, backend="redis")


def test_unsupported_option_names_the_pair_and_the_option():
    with pytest.raises(ValueError, match="memory/fixed_window does not accept"):
        create_limiter("fixed_window", 5, 60.0, max_delay=5.0)


def test_algorithm_specific_options_are_forwarded():
    limiter = create_limiter("leaky_bucket", 5, 60.0, max_delay=2.5)
    assert limiter.max_delay == 2.5


# --------------------------------------------------------------------------
# The clock default, which differs by backend on purpose
# --------------------------------------------------------------------------


def test_memory_limiters_get_the_local_wall_clock():
    limiter = create_limiter("fixed_window", 5, 60.0)
    assert limiter._clock is default_clock


def test_redis_limiters_are_left_on_the_server_clock(redis_or_skip):
    """The silent-failure case this sentinel exists to prevent.

    A Redis limiter with `clock=None` reads Redis's own TIME, so instances
    with skewed system clocks still agree on where a window boundary falls --
    the entire reason the Redis path does not use a local clock. If
    create_limiter forwarded its own `default_clock` default here, every
    limiter built through the registry would silently switch to the local
    process clock. Nothing would raise, no other test would fail, and the
    distributed guarantee would be gone.
    """
    limiter = create_limiter(
        "fixed_window", 5, 60.0,
        backend="redis", client=redis_or_skip, prefix="drltest:clock",
    )
    assert limiter._clock is None


def test_an_explicit_clock_still_reaches_a_redis_limiter(redis_or_skip):
    """Overriding is allowed -- that is how the tests stay deterministic."""
    clock = FakeClock()
    limiter = create_limiter(
        "fixed_window", 5, 60.0,
        backend="redis", client=redis_or_skip, prefix="drltest:clock", clock=clock,
    )
    assert limiter._clock is clock


async def test_a_registry_built_redis_limiter_enforces_its_limit(redis_or_skip):
    """End to end: built by name, with no clock, against a real server."""
    limiter = create_limiter(
        "fixed_window", limit=3, window=60.0,
        backend="redis", client=redis_or_skip, prefix=f"drltest:{uuid.uuid4().hex}",
    )

    allowed = 0
    for _ in range(6):
        if (await limiter.check("client-a")).allowed:
            allowed += 1
    assert allowed == 3


async def test_two_registry_built_limiters_share_one_limit(redis_or_skip):
    """What `backend="redis"` is actually for: separate instances, one quota."""
    prefix = f"drltest:{uuid.uuid4().hex}"
    build = lambda: create_limiter(  # noqa: E731 - two identical instances
        "fixed_window", limit=4, window=60.0,
        backend="redis", client=redis_or_skip, prefix=prefix,
    )
    instance_a, instance_b = build(), build()

    allowed = 0
    for limiter in (instance_a, instance_b) * 4:
        if (await limiter.check("client-a")).allowed:
            allowed += 1
    assert allowed == 4, "the two instances did not share a limit"


# --------------------------------------------------------------------------
# The packaged surface
# --------------------------------------------------------------------------


def test_everything_in_all_is_importable_from_the_top_level():
    """__all__ is the package's promise to applications that install it."""
    for name in drl.__all__:
        assert hasattr(drl, name), f"{name} is exported but missing"


def test_redis_classes_are_exported_without_redis_py():
    """The Redis limiters must not drag redis-py into the import path.

    They take the client duck-typed and only call register_script on it, which
    is what lets the package declare no dependencies at all. An accidental
    `import redis` in that layer would make the whole package unimportable for
    anyone who installed it without the extra.
    """
    assert drl.RedisTokenBucket is RedisTokenBucket

    import distributed_rate_limiter.redis_backend as backend

    for module in vars(backend).values():
        assert "redis." not in str(getattr(module, "__module__", "")), module


def test_version_is_exported():
    assert drl.__version__.count(".") >= 2
