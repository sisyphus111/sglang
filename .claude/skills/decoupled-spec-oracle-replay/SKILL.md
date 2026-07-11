---
name: decoupled-spec-oracle-replay
description: Estimate an actionable optimal dynamic draft-length schedule and E2E time from normalized SGLang decoupled-spec trajectories and a static candidate matrix. Use when computing oracle speedup, oracle-gain capture targets, per-state policy regret, or evidence-driven acceptance-statistic improvements without request-level traces.
---

# Decoupled Spec Oracle Replay

Consume `decoupled-spec-trajectory-analysis` outputs. Treat the result as a
fluid replay estimate, not a request-level exact counterfactual.

## Required Evidence

Require one static reference trajectory plus static measurements for every DL
candidate. A single log cannot reveal counterfactual rates for unobserved DLs
unless an independent cost/rate model is supplied. Prefer identical sampled
requests, deterministic inference, equal generated-token totals, no queueing,
and complete routed BS/context coverage.

## Workflow

1. Confirm `analysis/decode_points.csv` and `analysis/e2e_summary.csv` came from
   the same completed matrix.
2. Run the full candidate set:

   ```bash
   python .claude/skills/decoupled-spec-oracle-replay/scripts/replay_oracle.py \
     --analysis-dir <run-dir>/analysis --target-capture 0.7
   ```

   Pass `--reference-label` when the desired baseline is not the fastest static
   case. A `--static-steps` sensitivity run must use a separate `--output-dir`;
   never silently replace the primary full-matrix replay.
3. Read `oracle_replay/summary.json` first. If `decision_grade=false`, report
   the reasons before quoting speedup. Then inspect `bucket_policy.csv` and
   `replay_points.csv` for selected DLs and regret segments.
4. Optimize the online estimator or switching policy using the largest regret
   segments. Measure the fraction of oracle time saving captured:

   ```text
   (T_static - T_dynamic_normalized) / (T_static - T_oracle)
   ```

   Do not use `0.7 * oracle_speedup`; interpolate saved E2E time.

## Interpretation Rules

- State is `(allow_partial, routed_batch_slot, context_bucket)`. Estimate each
  static tier with aggregate `sum(output_tokens) / sum(iteration_time)`.
- Replay the reference path in generated-token space and replace each chunk's
  service time with the best measured tier rate for that state.
- Normalize a measured dynamic run to the reference generated-token total.
- Low candidate coverage, token drift, queueing, missing tiers, or unobserved
  states make the result diagnostic rather than an upper bound.
- Keep the offline oracle out of production. Use regret evidence to improve
  acclen statistics and controller decisions.

Read `references/methodology.md` for equations, assumptions, and audit fields.
