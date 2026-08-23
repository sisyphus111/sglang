# Artifact Relationships

## Identity and effective configuration

- `provenance/run_start.json` identifies run creation, checkout, interpreter,
  platform, and initial working-tree state.
- `server/resolved_config.json` is the unified Ray fleet input.
- `server/manifest.json` is the authoritative engine inventory. While traffic
  is running it has `state=ready`; after owned shutdown it has `state=stopped`.
  Each engine entry records its stable `engine_id`, role/rank, node/GPU
  placement, HTTP URL, transport endpoint, sparse quota peers, and run-relative
  config/status/log paths.
- `server/engines/<engine_id>/resolved_config.json`, the matching status file,
  and `logs/server/<engine_id>.log` are the effective per-engine artifacts
  copied back from the Ray node during shutdown.
- `roles/server/status.json` records the unified launcher lifecycle and local
  driver PID. Legacy runs may instead have singular
  `roles/{verifier,drafter}/resolved_config.json` and status files.
- `client/resolved_config.json` and
  `observability/resolved_config.json` are the effective request/collector
  configs.
- `roles/{client,observability}/status.json` records consumer lifecycle state
  and PID.

## Client cardinality chain

The following values must agree:

```text
client.resolved_config.batch.size
  = formal_window.batch_size
  = summary.batch_size
  = summary.request_count
  = sampled_requests.jsonl rows
  = responses.jsonl rows
  = request_metrics.csv rows
  = raw_batch_response.json entries
```

For a successful benchmark, `summary.completed_count` equals the same value and
`summary.failed_count` is zero.

## Observability chain

`observability/summary.json` describes `samples.jsonl`:

```text
number of JSONL records = sample_ct * target_ct
observed unsuccessful records = error_ct
```

Every raw sample carries `target_id`, role, rank, and base URL. Startup
snapshots live under `observability/startup/<engine_id>/`; every manifest
engine must have both `model_info.json` and `server_info.json`.

When decode windows are present,
`observability/summary.json.decode_metrics_by_target` must reproduce the unique
`(target_id, dp_rank, window_id)` entries from `samples.jsonl`.
`decode_metrics` is the compatible per-role aggregate. Verifier and drafter
iteration latency statistics remain separate. The serialized compatibility
key may still be named `scheduler_cycle_ms`.

Successful `/v1/loads` samples also preserve `num_running_reqs` separately for
verifier and drafter. The derived observability output presents these samples
as “Running batch size over time”; it must not substitute decode-window
`mean_batch_size` or fill missing samples with zero.

Every engine requires successful samples at or before formal start and at or
after formal finish. Runs at least one sampling interval long also require a
successful sample inside the formal window.

## Plot provenance

Each derived artifact has its own manifest:

- `plots/request_latency_manifest.json`
- `plots/request_speculative_manifest.json`
- `observability/plots/plot_manifest.json`
- `plots/run_report_manifest.json`

Every `sources` entry contains a run-relative path and SHA-256 value. Every
source hash must match and every declared output must exist. The speculative
manifest may have an empty output list when request-level speculative metrics
are unavailable.

## Seal

`run_manifest.json` is a centralized JSON index generated at seal time.
`SHA256SUMS` covers all regular files present after `run_manifest.json` is
written, excluding the checksum file itself. Adding any artifact inside the run
after sealing invalidates exact file-set coverage.
