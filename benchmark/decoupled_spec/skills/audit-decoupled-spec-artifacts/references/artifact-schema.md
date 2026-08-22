# Artifact Relationships

## Identity and effective configuration

- `provenance/run_start.json` identifies run creation, checkout, interpreter,
  platform, and initial working-tree state.
- `roles/verifier/resolved_config.json` and
  `roles/drafter/resolved_config.json` are the server-side effective configs.
- `client/resolved_config.json` and
  `observability/resolved_config.json` are the effective request/collector
  configs.
- `roles/*/status.json` records component lifecycle state and PID.

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

When decode windows are present, `observability/summary.json.decode_metrics`
must reproduce the unique per-role window counts from `samples.jsonl`. Verifier
and drafter scheduler-cycle statistics remain separate.

Both roles require successful samples at or before formal start and at or after
formal finish. Runs at least one sampling interval long also require a
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
