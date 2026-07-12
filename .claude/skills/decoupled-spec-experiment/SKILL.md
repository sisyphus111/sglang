---
name: decoupled-spec-experiment
description: Run reproducible SGLang decoupled-speculation static and dynamic experiment matrices through examples/runtime/engine/decoupled_speculation. Use when preparing a baseline matrix, validating an existing throughput profile cache, launching Qwen decoupled-spec benchmarks, resuming interrupted cases, or collecting raw logs and E2E summaries for later trajectory analysis. Use decoupled-spec-verifier-profile when the profile itself must be generated.
---

# Decoupled Spec Experiment

Own experiment execution and raw provenance. Do not interpret controller quality
or compute an oracle here; hand completed artifacts to the downstream skills.

## Workflow

1. Verify a clean active checkout, GPU topology, model/data paths, requested
   workload, and existing profile cache. The bundled matrix runner targets
   `single-node.py`. For a Ray/multi-node run, inspect `multi-node.py --help` and
   invoke it directly with explicit `--nnodes`, `--ray-address`, port, and GPU
   topology arguments; do not pass a multi-node entrypoint to this runner.
2. Copy `references/qwen35-27b-08b-default.toml` into a new result directory and
   edit only the copied config before the first invocation. The first invocation
   creates `config.lock.toml`; later config drift is rejected. Keep static and dynamic candidates,
   `allow_partial`, sampling seed, and profile buckets explicit.
3. Generate the target-only verifier cost cache with
   `decoupled-spec-verifier-profile`, then set its `profile.json` path in this
   experiment config. Validate it before a throughput-aware dynamic run:

   ```bash
   python .claude/skills/decoupled-spec-verifier-profile/scripts/profile_verifier.py \
     --config <profile-output>/profile-config.toml
   ```

4. Run or resume the matrix:

   ```bash
   python .claude/skills/decoupled-spec-experiment/scripts/run_matrix.py \
     run --config <run-dir>/config.toml --run-dir <run-dir>
   ```

5. Check `status.jsonl`, every terminal return code, `manifest.json`, logs, and
   summaries. A present `summary.json` is the resume boundary. `--force` reruns
   every selected matrix case and removes each stale summary before launch.

## Measurement Invariants

- Keep `max_running_requests >= batch_size + 1`; queueing changes the state path.
- Leave `ignore_eos=false` unless changed outputs are explicitly requested.
- Lock the sampling seed when comparing policies. Keep deterministic inference
  disabled unless the user explicitly requests it; the matrix runner does not
  expose a deterministic option.
- Profile every candidate step and every routed BS/context bucket needed by the
  workload. Require an exact target model/TP/DP/DP-attention/GPU fingerprint; a
  dynamic run must not silently accept an incompatible cache.
- Preserve one scheduler stream per case log. If an example launch also runs a
  normal baseline, capture it separately before trajectory analysis.
- Do not depend on debug-only environment variables; scheduler INFO is the
  portable input contract.

## Output Contract

This skill owns only raw artifacts:

```text
<run-dir>/
  config.toml
  config.lock.toml
  manifest.json
  status.jsonl  # lifecycle plus checkout/config provenance
  logs/<case>.log
  runs/<case>/summary.json
```

Case labels are `ap{0,1}_{static,dynamic}_dl<step>`. Static cases are the
measured policy baselines for trajectory and oracle analysis. Read
`references/artifact-contract.md` before changing filenames or resume rules.
