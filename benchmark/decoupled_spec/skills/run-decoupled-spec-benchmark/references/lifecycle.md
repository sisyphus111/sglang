# Run Lifecycle

The benchmark is an Agent-coordinated sequence, not a combined process
launcher.

## State machine

| Stage | Entry condition | Required evidence before continuing |
| --- | --- | --- |
| `preflight` | Unified server/client/observer tuple is known | Ray/GPU checks and unified config validation pass |
| `run_dir_ready` | Output and runtime roots are writable | The caller has manually created one unique `RUN_DIR` and disposable `RUNTIME_DIR` |
| `servers_ready` | Unified launcher is running | `<RUNTIME_DIR>/server/manifest.json` is ready and every verifier `/health` plus drafter `/model_info` succeeds |
| `observer_active` | All manifest engines are ready | Every engine has a successful zero-waiting baseline sample |
| `client_complete` | Observer is active | All three Client files exist and all batch members have final responses |
| `processes_stopped` | Formal window is complete | Observer samples/window exist; the owning environment reports no live job/pod for this run |
| `completed` | Raw run data is stable | Standard plots and the Markdown report exist |

## Run identity

Use a descriptive run name containing actual values rather than an opaque case
number. Include the axes needed to interpret the result, for example:

```text
qwen35-target-tp4-draft-tp1-k3-f1-bs1-dapo-thinking-out1k-overlap-cpp
```

`RUN_DIR/config.json` is authoritative. A label is only a readable summary and
must not substitute for saved configuration.

## Failure handling

- Do not reuse a partially written `RUN_DIR` for a clean retry.
- Keep partial files for diagnosis.
- Stop processes by the exact sessions or PIDs started for this run. Do not use
  broad process-name kills.
- Capture the earliest relevant server/client/observer error and identify the
  stage that failed.
- A positive verifier or drafter waiting queue is an invalid benchmark result,
  not a low-performance result. Preserve the attempt and diagnose
  admission before retrying.
- An Observer sampling error is not automatically a model-serving failure, but
  it prevents a fully observable successful run until its coverage is checked.

## Directory ownership

The caller creates `RUN_DIR` and `RUNTIME_DIR` before components start. The
final result contains only `config.json`, Client data, Observer data, and
derived plots. Server control state stays disposable. There is no initializer,
seal boundary, checksum file, or centralized artifact validator.
