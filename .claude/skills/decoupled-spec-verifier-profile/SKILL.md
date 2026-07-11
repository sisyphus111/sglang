---
name: decoupled-spec-verifier-profile
description: Generate and validate an SGLang decoupled-verifier throughput cost profile for a supplied target model, single-node target parallel topology, and requested batch-size/context-length/speculative-step grid. Use when a user needs the runtime `profile.json` consumed by the throughput-aware controller, wants to fill or rebuild missing verifier cost points, or wants profile provenance without running a draft model or benchmark dataset.
---

# Decoupled Spec Verifier Profile

Generate the verifier cost cache keyed by `(batch_size, steps, ctx_len)`. This is
not `utils/target-profile.py`, which measures target-only decode token-batch
throughput and recommends a budget rather than producing the runtime cost cache.

## Required Input

Collect these fields before launching anything:

- target: repository checkout, model path, visible GPU ids, TP size, DP size,
  whether DP attention is enabled, and any EP/MoE/dtype overrides;
- grid: batch sizes, context lengths, and steps;
- output directory and whether an existing profile may be replaced.

Do not infer model paths, GPU placement, TP/DP topology, or overwrite intent.
The first version is single-node. Use the existing multi-node runtime workflow
when the target engine spans nodes.

## Workflow

1. Inspect the active checkout and environment. Confirm the configured GPUs are
   visible and idle enough for profiling, and confirm the target model path is
   accessible.
2. Copy `references/profile-config.toml` into a new output directory and fill in
   the user's exact values. Steps must be the contiguous runtime candidate set
   `0..max_step`; reject sparse lists rather than silently profiling another set.
3. Validate the configuration without importing SGLang or using GPUs:

   ```bash
   python .claude/skills/decoupled-spec-verifier-profile/scripts/profile_verifier.py \
     --config <output-dir>/profile-config.toml --check
   ```

4. Run:

   ```bash
   python .claude/skills/decoupled-spec-verifier-profile/scripts/profile_verifier.py \
     --config <output-dir>/profile-config.toml
   ```

   Add `--force` only after the user authorizes replacing the current result.
5. Require terminal `status=ok` in `manifest.json`. Check the reported profile
   hash, fingerprint, requested/actual point counts, missing points, extra points,
   and duplicate count before returning the result path.
6. For a newly supported topology, first run a small probe such as BS `[1,8]`,
   ctx `[1024,4096]`, steps `[0,1]`. Expand to the full requested grid only after
   the probe generates a valid cache and the worker shuts down cleanly.

## Invariants

- The runner launches only the target `DECOUPLED_VERIFY` engine. It does not
  require a draft model, prompts, or a dataset.
- Profiling uses the runtime's startup CUDA Graph capture and timing path; the
  skill does not implement a second cost model.
- BS and ctx values are positive and unique. Steps are exactly `0..max_step`.
- Without DP attention, `dp_size` must be 1. With DP attention, visible GPU count
  must cover `tp_size * dp_size`.
- A valid result contains every requested point exactly once, no extra points,
  finite positive costs, and a target/TP/DP/DP-attention fingerprint match.
- Config is frozen as `profile-config.lock.toml`. A later invocation with changed
  input must use a new output directory.
- Timeout, Ctrl-C, and worker failure terminate the worker process group and do
  not claim success.

## Artifacts

```text
<output-dir>/
  profile.json
  profile-config.lock.toml
  manifest.json
  profile.log
```

`profile.json` is the only runtime input. The other files are provenance and
validation evidence. Read `references/profile-schema.md` when consuming or
reviewing the result.
