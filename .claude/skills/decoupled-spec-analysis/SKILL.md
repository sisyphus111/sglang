---
name: decoupled-spec-analysis
description: Run, replay, and analyze SGLang decoupled-speculation experiments using scheduler INFO logs and throughput-aware verifier profile caches. Use for static or dynamic step matrices, profile-cache validation, controller-aligned bs/ctx lookup, iteration-latency and throughput plots, controller switch extraction, E2E summaries, or reproducible Qwen decoupled-spec benchmark artifacts.
---

# Decoupled Spec Analysis

Use the bundled scripts as the deterministic implementation. Keep experiment-specific
paths and matrix choices in TOML instead of editing Python constants.

## Workflow

1. Inspect the active checkout, `spec.md`, the requested model/data paths, available
   GPUs, and any existing profile cache.
2. Copy and edit `references/qwen35-27b-08b-default.toml` for the run. Record the
   exact checkout and config beside the artifacts.
3. Generate or validate the profile before dynamic experiments:

   ```bash
   python .claude/skills/decoupled-spec-analysis/scripts/run_matrix.py \
     profile --config <config.toml> --run-dir <result-dir>
   ```

4. Run the requested static/dynamic matrix. The runner resumes completed cases by
   default and records commands, environment overrides, status, logs, and summaries:

   ```bash
   python .claude/skills/decoupled-spec-analysis/scripts/run_matrix.py \
     run --config <config.toml> --run-dir <result-dir>
   ```

5. Analyze scheduler INFO logs against the cache:

   ```bash
   python .claude/skills/decoupled-spec-analysis/scripts/analyze.py \
     --run-dir <result-dir> --profile <profile.json>
   ```

6. Report the config, profile coverage, failed cases, queue maximum, raw and smooth
   figures, structured CSV/JSON, and any unverified assumptions.

## Measurement Invariants

- Keep `max_running_requests >= batch_size + 1`; reject configs that can queue the
  measured requests.
- Leave `ignore_eos=false` unless the user explicitly requests changed outputs.
- Leave `spec_trace_dir` unset unless traces are explicitly requested.
- Do not require `SGLANG_TA_DEBUG`, `SGLANG_LOG_FORWARD_ITERS`, or
  `SGLANG_RECORD_STEP_TIME`; scheduler INFO is the base data source.
- Unless overridden by the requested experiment, capture/profile CUDA graph batch
  sizes must include `1,4,8,16,24,32,40,48,56,64`.
- Match the runtime controller exactly: choose the smallest profile BS greater than
  or equal to runtime BS (clamp above the largest), then choose nearest ctx bucket
  with lower-bucket tie breaking.
- Add the configured runtime CPU overhead, currently 3ms, exactly once when deriving
  modeled verifier iteration latency from raw profile cost. Scheduler-provided
  `modeled throughput` already includes it; derive its modeled ITL directly instead
  of adding 3ms again.
- Preserve raw points. Smooth plots use a centered moving average over nearby points;
  never replace the raw CSV or raw figure.
- Keep startup/tail filtering explicit in analysis metadata. Default analysis removes
  latency outliers at or above 100ms and keeps full-batch points for profile-fit plots.

## Artifacts

The runner owns `logs/`, `runs/`, `status.jsonl`, and `manifest.json`. The analyzer
writes only under `analysis/`: normalized decode points, controller switches,
profile tables, case summaries, and raw/smooth figures. See
`references/artifact-schema.md` when consuming these files from another tool.

