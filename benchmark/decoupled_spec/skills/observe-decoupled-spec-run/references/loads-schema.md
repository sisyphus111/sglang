# Loads Sample Schema

`observability/samples.jsonl` contains one record per manifest engine per
sampling round.

## Wrapper fields

| Field | Meaning |
| --- | --- |
| `sample_id` | Sampling-round index shared by the concurrent role requests |
| `target_id` | Unique manifest `engine_id`, such as `verifier-1` |
| `role` | Semantic role, `verifier` or `drafter` |
| `rank` | Rank within that role |
| `base_url` | Exact HTTP target sampled by the collector |
| `path` | Requested endpoint including `include` groups |
| `collected_wall_time` | Wall-clock time immediately before the HTTP request |
| `latency_ms` | HTTP collection latency measured with a monotonic clock |
| `status_code` | HTTP status, or null when the request failed before a response |
| `error` | Serialized collection exception, or null on success |
| `payload` | Raw JSON response, or null on failure |

A successful load sample has no error, a 2xx status, and a non-empty
`payload.loads` list.

## Load payload

The plotting path currently reads the first element of `payload.loads` and may
use:

- `num_running_reqs`: the server's instantaneous running request count at this
  HTTP sample, used as the user-facing running batch size
- `num_waiting_reqs`
- `gen_throughput`
- `token_usage`
- verifier `speculative.accept_length`
- verifier `speculative.accept_rate`
- verifier `speculative.proposed_draft_length`
- `decode_metrics_windows`, a bounded history of completed fixed-iteration
  decode windows. Each named window contains:
  - `window_id` and engine wall-clock `end_time`
  - `num_decode_iters` and `iter_latency_ms`; expose the latter as
    `iteration latency` in plots and reports
  - raw `num_decode_rows` and `sum_context_lens`
  - derived `mean_batch_size` and `mean_context_length`; the former is a mean
    over decode iterations and is not the sampled running batch size
  - raw `num_verify_rows`, `num_accept_tokens`, and `num_proposed_drafts`
  - derived `accept_length` and `proposed_draft_length` when speculative
    verification is active

The same bounded history can appear in consecutive HTTP samples. Offline tools
must deduplicate it by `(target_id, dp_rank, window_id)` and reject conflicting copies.
`proposed_draft_length` is the valid draft length actually presented to each
verify request-row; it excludes the bonus token.

`observability/summary.json.decode_metrics_by_target` aggregates each engine
independently; `decode_metrics` additionally aggregates by role. The
compatibility fields `verifier.scheduler_cycle_ms` and
`drafter.scheduler_cycle_ms` store separate role-level iteration latency distributions
with mean/min/p50/p95/max fields. Their means are weighted by each window's
`num_decode_iters`; batch/context/spec ratios are recomputed from raw window
numerators and denominators rather than averaging ratios.

Plot `num_running_reqs` directly from successful samples as “Running batch size
over time” (Chinese report: “运行 BS 随时间变化”). Use one line per `target_id`,
a linear y-axis whose lower bound is zero, and the collector's
wall-clock time axis. Preserve baseline and trailing samples. Missing samples
remain gaps rather than zeros. This plot is distinct from decode-window
`mean_batch_size`.

Field availability depends on the server's enabled metrics and requested
`loads.include` groups. Preserve the complete payload rather than projecting it
to only these fields.

For decoupled-spec performance validity, `num_waiting_reqs` is a hard
invariant: every verifier and drafter engine must report zero throughout the formal
window. A missing/non-integer value cannot prove the invariant and is rejected;
any positive value invalidates the run.

Startup snapshots under `observability/startup/<target_id>/` use the same wrapper
shape with `sample_id=-1` for `/model_info` and `/server_info`.
