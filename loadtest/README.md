# Load test results

Every number here was produced by running the code against a real Redis
server over real time, and can be recomputed from the logs in `results/`:

```bash
PYTHONPATH=src .venv/bin/python -m loadtest              # run all three scenarios
PYTHONPATH=src .venv/bin/python -m loadtest --plot-only  # rebuild tables and
                                                         # figures from the logs
```

All three scenarios use **limit 20 per 2s window** against the five
Redis-backed limiters. The short window is a compression, not a
simplification: boundary behaviour is scale-free in window units, so 2s shows
what 60s does while a five-boundary run finishes in ten seconds. The traffic,
the concurrency and the Redis server are real.

Two properties of the rig make the comparison mean anything:

- **One schedule, replayed five times.** The arrival times are generated once
  and handed identically to every algorithm. Otherwise a difference in the
  results would be part algorithm and part traffic, and no part of it could
  be attributed to either.
- **Runs start on a window boundary**, read from Redis's own `TIME` — the
  same clock the Lua scripts use. Starting at an arbitrary phase would put
  the fixed window's cliff in an arbitrary place and make two runs
  incomparable.

## The metric that matters: `max/win`

Not "how many were rejected" but **the most requests admitted in any interval
one window long**, wherever that interval falls.

A limiter can honour "20 per window" on every window *it* defines and still
let a client land 40 requests inside a single window's worth of time, by
drawing a boundary between them. Counting over a sliding interval ignores
where the limiter chose to put its boundaries and asks what the client
actually experienced. Peaks are computed per key and then maximised, never
pooled — pooling would add up independent quotas and report three
well-behaved clients as a 3× violation.

---

## 1. Burst straddling a boundary

`results/boundary_burst.png` — 30 requests at 1.85s, 30 more at 2.15s
(0.3 seconds apart, either side of a boundary), and a third burst of 30 at
7.0s mid-window as a control.

| algorithm | allowed / 90 | max in any 2s | × limit |
|---|---|---|---|
| fixed window | 60 | **40** | **2.00** |
| sliding window log | 40 | 20 | 1.00 |
| sliding window counter | 42 | 22 | 1.10 |
| token bucket | 42–43 | 22–23 | 1.10–1.15 |
| leaky bucket | 43 | 21 | 1.05 |

**The fixed window admitted exactly twice its limit inside 0.3 seconds.** It
has no memory across a boundary, so both bursts got a full fresh allowance.
This is the textbook failure, and it reproduced at exactly the textbook
factor rather than approximately.

**The sliding window log admitted exactly 20 — 1.00×.** It is the reference
implementation of the intent, and it costs O(n) memory per key to be exact.

The remaining three land between 1.05× and 1.15×, and the reasons differ: the
counter's 1.10× is its known approximation error, the token bucket's is the
0.3 seconds of refill arriving between the two bursts, and the leaky bucket's
1.05× is what survives after pacing. On the mid-window control burst, where
no boundary is nearby, all five agree closely — the disagreement is a
boundary phenomenon, not a general one.

**The token bucket is the one row here that does not reproduce exactly, and
the reason is worth more than the number.** Refill is `limit / window` =
10 tokens/s and the bursts are 0.30s apart, so the third token arrives at
precisely the instant the second burst does. Whether it counts is decided by
sub-millisecond dispatch jitter: the logged run in `results/` dispatched
1.1 ms early, giving 2.990 tokens and 2 admissions (1.10×), while five later
runs on the same machine dispatched on time, giving 3.000 tokens and 3
admissions (1.15×). Moving the second burst by 0.1 ms flips it. The other
four algorithms are immune because their state is discrete — counts and log
entries — where a millisecond changes nothing; the token bucket is the only
one with a continuous quantity crossing an integer threshold at exactly the
moment it is sampled. The scenario is not broken, it is sitting on a
discontinuity, and the honest report is the range.

## 2. Steady demand at twice the sustainable rate

`results/steady_over_limit.png` — 200 requests at a smooth 20/s for 10s,
against a sustainable 10/s.

| algorithm | allowed / 200 | max in any 2s | × limit |
|---|---|---|---|
| fixed window | 100 (50.0%) | 21 | 1.05 |
| sliding window log | 100 (50.0%) | 21 | 1.05 |
| sliding window counter | 100 (50.0%) | 21 | 1.05 |
| token bucket | **119 (59.5%)** | **39** | **1.95** |
| leaky bucket | **119 (59.5%)** | 21 | 1.05 |

Two results worth separating.

**The buckets admitted 19 more requests than the windows, and that is
correct.** A bucket starts full, so its capacity of 20 is a burst allowance
*on top of* the sustained 10/s × 10s = 100. Capacity decoupled from rate is
the token bucket's entire reason for existing; here it is worth +19 requests.

**The token bucket peaked at 1.95×, almost exactly the fixed window's 2.00×
from scenario 1 — and it means the opposite.** The fixed window over-admits
because it forgot the previous window. The token bucket over-admits because
it was configured to allow a burst of 20 and then sustain 10/s, and the
sliding metric catches the burst and the following window's refill in one
interval. One is a boundary artefact; the other is the specification. A
comparison that only reported peak admission would rank these two together,
which is why the shape of the curve matters as much as the peak.

**The leaky bucket admitted the same 119 requests but never exceeded 1.05×**,
because it paced them: identical volume, spread instead of clumped. That gap
— 119 admitted, 1.95× versus 1.05× — is the difference between metering and
shaping, measured.

The cumulative plot shows the distribution the totals hide: the fixed window
climbs to 20 in the first fraction of each window then flat-lines until the
next boundary, while the buckets track the sustainable rate as a straight
line.

## 3. Multiple clients

`results/multi_client.png` — three keys against one limiter: `polite` at 5/s
(half its limit), `greedy` at 25/s (2.5× over), `spiky` at Poisson 10/s
(exactly at the limit).

| algorithm | polite | greedy | spiky |
|---|---|---|---|
| fixed window | 50/50 (100%) | 100/250 (40.0%) | 95/113 (84.1%) |
| sliding window log | 50/50 (100%) | 100/250 (40.0%) | 90/113 (79.6%) |
| sliding window counter | 50/50 (100%) | 100/250 (40.0%) | 92/113 (81.4%) |
| token bucket | 50/50 (100%) | 119/250 (47.6%) | 112/113 (99.1%) |
| leaky bucket | 50/50 (100%) | 119/250 (47.6%) | 112/113 (99.1%) |

**The polite client was admitted 50 of 50 on every algorithm, with zero
rejections**, while the greedy client next to it was rejected 150 times.
Isolation holds — worth measuring rather than assuming, since a Lua script
writing to a key it was not handed would show up here and nowhere else in
the suite.

`spiky` is the interesting column. Its arrivals are Poisson at a mean of
10/s against a sustainable 10/s, and this particular draw landed 113 requests
in the ten seconds — so it is fractionally over its budget, and the rest of
its rejections come from clustering rather than volume. The buckets absorb
nearly all of it (99.1%, one rejection); the sliding log, which is exact and
therefore unforgiving, rejects 23. Being exact and being useful are not the
same property.

## Latency

Median check latency ranged **0.44–2.36 ms** across the scenarios above, but
that figure describes the rig under load rather than the cost of a check:
most of those requests arrive in concurrent bursts and queue for a connection
from a bounded pool.

For the per-check cost, `python -m loadtest.latency` runs 2000 sequential
checks per algorithm, where nothing queues:

| algorithm | median | p95 |
|---|---|---|
| fixed window | 0.166 ms | 0.195 ms |
| sliding window log | 0.173 ms | 0.198 ms |
| sliding window counter | 0.171 ms | 0.201 ms |
| token bucket | 0.171 ms | 0.205 ms |
| leaky bucket | 0.174 ms | 0.204 ms |

**0.166–0.174 ms across all five.** A Lua script costs nothing measurable
over a bare `INCR`, so choosing between these algorithms is a question of
correctness and memory, not speed.

## What the three scenarios add up to

- Under smooth traffic well below the limit, all five are indistinguishable.
  Every difference is a boundary or burst phenomenon.
- Only the fixed window over-admits *by accident*, and it does so by a
  full 2.00×.
- The sliding window log is exactly right and pays O(n) memory per key for it.
  The counter buys O(1) state for a measured 1.10×.
- The buckets admit deliberately more, and the choice between them is not
  about volume — 119 either way — but about whether that volume arrives in a
  clump (token, 1.95×) or paced (leaky, 1.05×).
