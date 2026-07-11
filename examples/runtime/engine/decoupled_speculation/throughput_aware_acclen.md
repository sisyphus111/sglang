# Throughput-Aware Acceptance-Length Estimation

This document describes how the decoupled verifier estimates acceptance length
for `--speculative-adaptive-strategy throughput_aware`, and why the estimator
has more state than the standalone `ema` strategy.

## Overall design

The two adaptive strategies use acceptance observations differently:

- `ema` treats the EMA of the batch-average number of correct drafts as the
  policy output and maps it directly to a speculative step tier.
- `throughput_aware` uses EMA statistics only to estimate the useful output of
  each candidate tier. It then divides that estimate by the profiled verifier
  cost at the current batch size and context length.

For candidate step `s`, the controller selects by estimated useful-token rate:

```text
expected_tokens(s) = 1 + sum(position_accept_rate[k] for k in 1..s)

score(s) = expected_tokens(s) /
           (profile_cost(batch_slot, s, context_bucket) + 3 ms)
```

The leading `1` is the target model's bonus token. The acceptance input
`num_correct_drafts_per_req` contains correct draft tokens only and excludes
that bonus token.

The estimator has two complementary views:

1. A global per-position acceptance EMA provides a counterfactual estimate for
   candidate tiers that are not currently active.
2. A `(CUDA graph batch slot, active tier)` expected-output EMA records what an
   actually executed tier produced, including partial-draft-tail effects.

The controller prefers a sufficiently warmed tier-specific observation. When
that observation becomes stale, it decays continuously toward the current
per-position estimate. A cold tier is not force-run merely to collect a sample;
it runs only after its estimated score wins the normal comparison.

## 1. Track acceptance by draft position

For every verify batch, position `k` contributes a binary observation per
request:

```text
position k is accepted iff num_correct_drafts >= k
```

The batch sample is therefore:

```text
batch_rate[k] = count(num_correct_drafts >= k) / batch_size
```

Each observed position has an independent EMA:

```text
position_ema[k] = (1 - alpha) * position_ema[k]
                  + alpha * batch_rate[k]
```

The default `alpha` is `0.2`.

### Example

Suppose the active tier is DL3 and four requests return the following numbers
of correct drafts:

```text
[3, 1, 0, 2]
```

The batch samples are:

```text
position 1: 3 / 4 = 0.75
position 2: 2 / 4 = 0.50
position 3: 1 / 4 = 0.25
```

The estimated useful output for DL3 is then:

```text
1 bonus + 0.75 + 0.50 + 0.25 = 2.50 tokens/request
```

### Why position statistics instead of one mean acclen

The identity

```text
E[min(correct_drafts, s)] = sum(P(correct_drafts >= k), k=1..s)
```

lets the controller score every candidate prefix from the same observations.
A single mean such as `mean([3, 1, 0, 2]) = 1.5` does not say how much of that
mean belongs to position 1, 2, or 3, so it cannot safely estimate a different
candidate width.

## 2. Preserve the monotonic acceptance invariant

Acceptance is cumulative: a request cannot accept position 3 without accepting
positions 1 and 2. The estimator must therefore maintain:

```text
p1 >= p2 >= p3 >= ...
```

Only positions reached by the active tier receive fresh samples. If a newly
updated lower-position EMA drops below preserved deeper EMAs, the deeper values
are clamped to the lower value and marked `projected`. Their warmup counts are
reset until real observations refresh them.

### Example

Assume an earlier DL3 phase produced:

```text
[p1, p2, p3] = [0.80, 0.60, 0.40]
```

The controller later runs DL1 and the fresh `p1` EMA falls to `0.30`. Keeping
the old deeper values would produce the impossible state:

```text
[0.30, 0.60, 0.40]
```

Instead, the controller projects it to:

```text
[0.30, 0.30, 0.30]
```

`p2` and `p3` can still provide a conservative estimate, but they no longer
count as warmed evidence. This prevents a stale high-acceptance phase from
immediately pushing the controller back to a wide tier.

## 3. Explore at most one unseen position

When exactly the next position is unobserved, the controller extrapolates it
with the recent geometric decay between known positions. With only `p1`
available, it uses `p2 = p1 * p1`.

It does not extrapolate two or more unseen positions at once. Positive
candidate steps must therefore be contiguous; a sparse ladder such as
`[0, 1, 3]` is rejected, while `[0, 1, 2, 3]` is valid.

### Example

After running DL1, suppose:

```text
p1 = 0.75
```

The controller may estimate:

```text
p2 = 0.75 * 0.75 = 0.5625
expected_tokens(DL2) = 1 + 0.75 + 0.5625 = 2.3125
```

It cannot yet estimate DL3. If DL2 wins by score and executes, position 2
becomes a real observation; only then can DL3 be considered.

### Why exploration is progressive

Copying `p1` into every unseen position can make DL1 jump directly to DL3 even
though no request has demonstrated that positions 2 and 3 survive. Progressive
exploration bounds the error of cold-start counterfactuals and ensures every
new decision exposes the next statistic needed by a wider tier.

## 4. Track actual output per batch slot and tier

The controller also maintains an EMA keyed by:

```text
(routed CUDA graph batch slot, active DL)
```

For an executed tier, its batch observation is:

```text
tier_expected_tokens = 1 + mean(num_correct_drafts_per_req)
```

The batch size is routed with the same ceiling rule used by CUDA graphs. For
example, an actual BS37 batch belongs to slot BS40 when the captured slots are
`[..., 32, 40, 48, ...]`.

### Example with `allow_partial=True`

At BS40, the draft pipeline may not supply the same effective tail at DL1 and
DL2. Suppose real executions observe:

```text
slot BS40, DL1: expected output EMA = 1.55
slot BS40, DL2: expected output EMA = 1.91
```

Those measurements include both acceptance quality and the partial draft tail
that actually arrived for each width. Reconstructing DL2 only from DL1's
position EMA would incorrectly assume that draft availability is identical.

### Why the key includes both batch slot and tier

- Verifier and partial-pipeline behavior changes with routed batch shape.
- A sample from DL1 is not an actual-output sample for DL2.
- Keeping width-specific samples avoids transferring partial-tail loss or gain
  across unrelated tiers.

## 5. Decay stale tier observations

A warmed tier-specific EMA is authoritative while it is current. Once the
controller leaves that tier, its workload state continues moving through batch
size and context length. The old value is blended toward the current
per-position estimate:

```text
confidence = (1 - alpha) ** age_in_verify_batches

expected = confidence * tier_ema
           + (1 - confidence) * position_expected
```

### Example

With `alpha=0.2`, a DL2 observation last updated 10 verify batches ago has:

```text
confidence = 0.8 ** 10 ~= 0.107
```

If its old tier EMA is `2.20` but the current position estimate is `1.60`, the
score uses approximately:

```text
0.107 * 2.20 + 0.893 * 1.60 ~= 1.66
```

### Why use continuous decay

Keeping `2.20` forever lets an old high-acceptance regime repeatedly pull the
controller back to DL2. A hard expiry threshold creates a discontinuous score
jump. EMA-consistent decay removes stale authority gradually without adding a
second age threshold to tune.

## 6. Do not force every cold tier to run

A tier with fewer than the default 10 width-specific samples uses the global
position estimate. It remains eligible if that estimate and the cost profile
produce a valid score, but a new batch slot does not force a probe of every
tier.

### Example

When the workload first enters BS24, DL2 may have no `(BS24, DL2)` samples. If
the position estimate predicts `1.70` useful tokens and the profiled cost makes
its score lower than current DL1, the controller stays on DL1. It does not pay
for DL2 merely to warm the slot.

### Why probes are score-gated

Forced probes are real verifier iterations. Across many batch slots and tiers,
their cost can exceed the small theoretical gain available from dynamic
switching. Score-gated exploration collects a tier-specific sample only when
the current evidence already says that running the tier is worthwhile.

DL0 is the exception: because it exposes no draft acceptance positions, the
controller probes the smallest positive tier when it cannot construct any
positive score.

## 7. Combine expected output with verifier cost

The cost table is indexed by candidate DL, routed batch size, and nearest
profiled context bucket. The runtime adds a fixed 3 ms CPU-overhead estimate
before scoring. Every five verify batches by default, the controller compares
eligible candidates and switches only when the best score is more than 5%
above the current score.

### Example

At a particular BS/ctx state:

```text
DL1: expected=1.57, cost=59.98 ms, score=0.0262
DL2: expected=1.90, cost=64.10 ms, score=0.0297
```

DL2 is about 13% better by score, so it clears the 5% hysteresis and wins. At a
larger batch size where DL2 validation cost grows faster, the same acceptance
rates may leave DL1 with the higher score.

### Why acclen alone is insufficient

The standalone `ema` strategy answers "how many drafts are likely to be
correct?" The throughput-aware strategy must answer "which width produces the
most useful tokens per unit verifier time at this BS and ctx?" A wider tier can
have a larger accept length and still be slower end to end.

## Observability

Each throughput-aware switch log includes:

- the old and new DL;
- BS, average context length, and reevaluation batch count;
- current/best scores and their ratio;
- per-candidate expected output, profile cost, and total cost;
- per-position acceptance values and whether they are EMA, projected, or
  probe estimates;
- whether expected output came from fixed DL0 output, position EMA,
  tier-specific EMA, or age-decayed tier EMA;
- tier update count and age.

These fields are intended to make a wrong switch diagnosable as either an
acceptance-estimation error, a stale observation, or a verifier-cost-profile
error.
