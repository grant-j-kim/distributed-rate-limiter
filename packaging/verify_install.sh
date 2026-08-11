#!/usr/bin/env bash
#
# Build the wheel, install it somewhere that cannot see this repository, and
# run a real FastAPI application against it.
#
# This is what separates "the metadata looks right" from "an application can
# actually install and use this". An editable install keeps the source tree on
# sys.path, so it hides exactly the failures that matter here: a subpackage
# missing from the wheel, a missing py.typed, a wrong `packages` setting. The
# temporary venv is created outside the repository so an accidental relative
# import cannot succeed.
#
# Usage:  ./packaging/verify_install.sh
# Needs:  a Redis server on localhost:6379 (uses DB 13)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REDIS_CLI="${REDIS_CLI:-$HOME/.local/bin/redis-cli}"
PORT_A=8137
PORT_B=8138
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/drl-verify-XXXXXX")"

cleanup() {
  for pid in "${PID_A:-}" "${PID_B:-}"; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
    # Reap it here, otherwise the shell prints its own "Terminated" notice
    # after the script's last line and the run looks like it failed.
    wait "$pid" 2>/dev/null || true
  done
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "==> Redis"
"$REDIS_CLI" ping >/dev/null || { echo "no Redis on localhost:6379"; exit 1; }
"$REDIS_CLI" -n 13 flushdb >/dev/null

echo "==> building wheel"
rm -rf "$REPO/dist"
(cd "$REPO" && .venv/bin/python -m build --wheel) | tail -1
WHEEL="$(ls "$REPO"/dist/*.whl)"

echo "==> installing into a fresh venv at $WORKDIR"
python3 -m venv "$WORKDIR/.venv"
"$WORKDIR/.venv/bin/python" -m pip install --quiet --upgrade pip
"$WORKDIR/.venv/bin/python" -m pip install --quiet "${WHEEL}[redis,server]" httpx

echo "==> zero-dependency check"
# The core package must install with nothing behind it. The Redis limiters
# take a duck-typed client and import no third-party module, so redis-py
# belongs to the application that supplies one.
python3 -m venv "$WORKDIR/.bare"
"$WORKDIR/.bare/bin/python" -m pip install --quiet "$WHEEL"
DEPS="$("$WORKDIR/.bare/bin/python" -m pip list --format=freeze 2>/dev/null \
        | grep -viE '^(pip|setuptools|wheel|drl-ratelimit)==' || true)"
if [[ -n "$DEPS" ]]; then
  echo "    FAIL  unexpected dependencies: $DEPS"; exit 1
fi
"$WORKDIR/.bare/bin/python" -c "
import distributed_rate_limiter as drl
drl.create_limiter('token_bucket', limit=1, window=1)
assert drl.RedisTokenBucket  # importable with no redis-py present
print('    PASS  installs and imports with zero dependencies')
"

echo "==> starting two independent server processes"
cp "$REPO/packaging/consumer_app.py" "$WORKDIR/app.py"
cd "$WORKDIR"
"$WORKDIR/.venv/bin/python" -m uvicorn app:app --port "$PORT_A" --log-level warning \
  >"$WORKDIR/a.log" 2>&1 & PID_A=$!
"$WORKDIR/.venv/bin/python" -m uvicorn app:app --port "$PORT_B" --log-level warning \
  >"$WORKDIR/b.log" 2>&1 & PID_B=$!

for _ in $(seq 30); do
  curl -fs "http://127.0.0.1:$PORT_A/version" >/dev/null 2>&1 && break
  sleep 0.5
done

# Each phase starts from a clean slate: the app-wide middleware meters every
# non-exempt path, so one phase's traffic would otherwise eat into the quota
# the next phase is measuring.
echo "==> per-endpoint decorator"
"$REDIS_CLI" -n 13 flushdb >/dev/null
"$WORKDIR/.venv/bin/python" "$REPO/packaging/drive_consumer.py" endpoint "$PORT_A"

echo "==> app-wide middleware, single process"
"$REDIS_CLI" -n 13 flushdb >/dev/null
"$WORKDIR/.venv/bin/python" "$REPO/packaging/drive_consumer.py" global "$PORT_A"

echo "==> app-wide middleware, two processes sharing one Redis"
# The real claim: one limit enforced *between* processes rather than once per
# process. A limiter holding state in memory would admit double here and pass
# every single-process test above.
"$REDIS_CLI" -n 13 flushdb >/dev/null
"$WORKDIR/.venv/bin/python" "$REPO/packaging/drive_consumer.py" global "$PORT_A" "$PORT_B"

echo
echo "==> packaging verified"
