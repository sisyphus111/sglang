# Run Lifecycle

The benchmark is an Agent-coordinated sequence, not a combined process
launcher.

## State machine

| Stage | Entry condition | Required evidence before continuing |
| --- | --- | --- |
| `preflight` | Unified server/client/collector tuple is known | Ray/GPU checks and unified config validation pass |
| `initialized` | Output root is writable | Unique `RUN_DIR` and `provenance/run_start.json` exist |
| `servers_ready` | Unified launcher is running | `server/manifest.json` is ready and every verifier `/health` plus drafter `/model_info` succeeds |
| `collector_active` | All manifest engines are ready | Every engine has a successful zero-waiting baseline sample |
| `client_complete` | Collector is active | Client status is `completed` and all batch members have final responses |
| `processes_stopped` | Formal window is complete | Collector summary exists; launcher has collected remote artifacts and removed owned Ray resources |
| `derived` | Raw run artifacts are stable | Four independent plot/report manifests and their declared outputs exist |
| `pre_seal_audited` | All derived files exist | Artifact audit exits zero |
| `sealed` | Pre-seal audit passes | `run_manifest.json` and `SHA256SUMS` exist |
| `verified` | Run is sealed | Read-only sealed audit and checksum verification pass |

## Run identity

Use a descriptive run name containing actual values rather than an opaque case
number. Include the axes needed to interpret the result, for example:

```text
qwen35-target-tp4-draft-tp1-k3-f1-bs1-dapo-thinking-out1k-overlap-cpp
```

The resolved configs are authoritative. A label is only a readable summary and
must not substitute for saved configuration.

## Failure handling

- Do not reuse a partially written `RUN_DIR` for a clean retry.
- Keep partial files unsealed for diagnosis.
- Stop processes by the exact sessions or PIDs started for this run. Do not use
  broad process-name kills.
- Capture the earliest relevant server/client/collector error and identify the
  stage that failed.
- A positive verifier or drafter waiting queue is an invalid benchmark result,
  not a low-performance result. Preserve the attempt unsealed and diagnose
  admission before retrying.
- A collector sampling error is not automatically a model-serving failure, but
  it prevents a fully observable successful run until its coverage is checked.

## Seal boundary

Run single-run plotting and the pre-seal audit before sealing. The pre-seal
audit report may be written to `RUN_DIR/audit/pre_seal.json`; it is then covered
by `SHA256SUMS`. Post-seal validation must be read-only or write outside
`RUN_DIR`.
