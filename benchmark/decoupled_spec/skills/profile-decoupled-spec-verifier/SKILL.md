---
name: profile-decoupled-spec-verifier
description: Generate, resume, validate, and inspect one single-JSON scheduler-cycle cost profile for SGLang decoupled verifier dynamic step selection. Use before an adaptive decoupled-spec experiment or when profile coverage/fingerprint is in question.
---

# Profile a Decoupled Verifier

Generate the verifier cost input used by production adaptive K selection. This
skill profiles only `DECOUPLED_VERIFY`; it does not start a drafter or benchmark
normal acceptance quality.

## Workflow

1. Read [references/profile-contract.md](references/profile-contract.md).
2. Inspect the active checkout, GPU/Ray ownership, model path, profile YAML,
   requested output path, and any resume input. Do not overwrite an existing
   output without explicit `--overwrite`.
3. Run the entry point in check mode first:

   `python3 -m benchmark.decoupled_spec.server.profile --config CONFIG --output PROFILE.json --check`

4. Run the same command without `--check`. For resume, pass an old partial or
   complete JSON through `--input`; only missing compatible grid points may run.
5. Require `status=complete`, the full requested coverage, a stable SHA256, full
   acceptance, `measurement_clock=scheduler_verify_commit_gap`, and positive
   finite `cost_ms` for every point.
6. Record the absolute profile path and SHA. Production server configs inject
   it through `SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH`; the unified Ray
   launcher stages the exact bytes onto every verifier node.

The persistent profile product is exactly one JSON file. Temporary job specs,
Ray state, and stdout logs are execution details, not profile artifacts.

## Output Contract

Return the config path, output path, profile SHA256, requested/reused/profiled
point counts, fingerprint, schedule mode, ReplaySSM state, K/BS/context grid,
and the first failed invariant when incomplete.
