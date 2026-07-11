# Verifier Throughput Profile Schema

The runtime writes `profile.json` atomically with three top-level fields:

- `fingerprint`: `target_model_path`, per-DP-lane `target_tp_size`,
  `target_dp_size`, `enable_dp_attention`, and the detected `gpu_name`;
- `costs`: one row for every requested `(batch_size, steps, ctx_len)` with a
  finite positive `cost_ms`;
- `summary`: the runtime cost-table summary for logging and inspection.

The profiler requires the actual point set to equal the requested Cartesian
product. Missing, duplicate, or extra rows fail validation. The GPU name is
recorded rather than guessed from configuration; subsequent runtime loading
performs the same fingerprint compatibility check.

`manifest.json` records the normalized request, checkout status, config and
profile hashes, command, terminal status, and validation summary. A profile is
usable only when the manifest has `status: "ok"`.
