---
name: decoupled-spec-trajectory-analysis
description: Normalize and visualize SGLang decoupled-speculation runtime trajectories from completed static or dynamic scheduler logs. Use when plotting throughput and acceptance-length changes, reconstructing the controller EMA throughput model over time, reproducing controller-aligned batch/context profile lookup, extracting step switches, comparing runtime and modeled latency, or producing stable CSV/JSON inputs for oracle replay.
---

# Decoupled Spec Trajectory Analysis

Consume raw artifacts from `decoupled-spec-experiment`; do not launch or mutate
experiment cases. The normalized CSV files are the handoff to oracle replay.

## Workflow

1. Verify that each `logs/<case>.log` contains one scheduler stream and that the
   profile cache covers its candidate steps and routed states. Confirm the active
   Python can import `matplotlib`; if not, inspect the environment before adding
   it to a project-local environment or isolated temporary target.
2. Analyze scheduler INFO logs against the cache:

   ```bash
   python .claude/skills/decoupled-spec-trajectory-analysis/scripts/analyze.py \
     --run-dir <result-dir> --profile <profile.json>
   ```

3. Inspect `decode_points.csv` before summaries. For new dynamic logs, compare
   observed throughput, profile-plus-observed-acclen throughput, and controller
   EMA throughput. Then inspect controller switches, queue maximum, raw/smooth
   profile-fit plots, E2E summaries, and metadata.
4. Treat a missing `accept_len` as unavailable (for example, a static DL0
   baseline scheduler stream). Never synthesize speculative acceptance.

## Measurement Invariants

- Do not require `SGLANG_TA_DEBUG`, `SGLANG_LOG_FORWARD_ITERS`, or
  `SGLANG_RECORD_STEP_TIME`; scheduler INFO is the base data source.
- Match the runtime controller exactly: choose the smallest profile BS greater than
  or equal to runtime BS (clamp above the largest), then choose nearest ctx bucket
  with lower-bucket tie breaking.
- Add the configured runtime CPU overhead, currently 3ms, exactly once when deriving
  modeled verifier iteration latency from raw profile cost. Scheduler-provided
  `modeled throughput` already includes it; derive its modeled ITL directly instead
  of adding 3ms again.
- Treat `EMA expected tokens` as the only runtime-only input for the controller
  EMA curve. Derive its throughput offline from existing BS and modeled cost;
  do not require per-position EMA state in periodic scheduler INFO. Do not apply
  a second moving average to this already-smoothed controller estimate.
- Preserve raw points. Smooth plots use a centered moving average over nearby points;
  never replace the raw CSV or raw figure.
- Keep startup/tail filtering explicit in analysis metadata. Default analysis
  removes latency outliers at or above 100ms. Profile-fit plots keep full-batch
  points, while throughput/acclen trajectories retain every batch size so the
  small-batch tail remains visible.

## Artifacts

The experiment skill owns `logs/`, `runs/`, `status.jsonl`, and `manifest.json`.
This skill writes only under `analysis/`: normalized decode points, controller
switches, profile tables, case summaries, and raw/smooth figures. See
`references/artifact-schema.md` when consuming these files from another tool.
