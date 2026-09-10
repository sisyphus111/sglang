# Loads Sample Schema

`observer/samples.jsonl` contains one record per manifest engine per
sampling round.

## Wrapper fields

| Field | Meaning |
| --- | --- |
| `sample_id` | Sampling-round index shared by the concurrent role requests |
| `interval_s` | Configured observer polling interval |
| `observer_started_wall_time` | Observer wall-clock start shared by all records |
| `target_id` | Unique manifest `engine_id`, such as `verifier-1` |
| `role` | Semantic role, `verifier` or `drafter` |
| `rank` | Rank within that role |
| `base_url` | Exact HTTP target sampled by the observer |
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
  - optional verifier `decoupled_spec.tail_select`, containing raw selector
    row/reason/freshness counters and integer length histograms
  - optional role-local `decoupled_spec.transport`, containing raw frame/token
    counters and non-cumulative latency histograms
  - optional verifier `decoupled_spec.adaptive_verify`, containing active/Kmax,
    profile status/SHA, cumulative and window-local reevaluation/switch counts,
    K residency, per-position supply and conditional-accept EMAs, latest
    candidate cost/TPS rows, and bounded switch events

The same bounded history can appear in consecutive HTTP samples. Offline tools
must deduplicate it by `(target_id, dp_rank, window_id)` and reject conflicting copies.
`proposed_draft_length` is the valid draft length actually presented to each
verify request-row; it excludes the bonus token.

Plot and report code derive each engine and role aggregate directly from raw,
deduplicated windows. Iteration-latency means are weighted by each window's
`num_decode_iters`; batch/context/spec ratios are recomputed from raw window
numerators and denominators rather than averaging ratios.

## Decoupled-spec histogram contracts

Every tail histogram has this exact integer-bin representation:

```json
{
  "offset": -1,
  "counts": [0, 10, 20],
  "underflow_count": 0,
  "overflow_count": 0
}
```

`counts[i]` is the count for the exact integer value `offset + i`.
`sum(counts) + underflow_count + overflow_count` equals
`tail_select.num_select_rows`. `reason_counts` also assigns exactly one reason
to every select row. For speculative step count `K`, selected length uses
`offset=0` with `K+1` bins; raw, consumable, and pending length use `offset=-1`
with `2K+3` bins; logical delta uses `offset=-(2K+1)` with `4K+3` bins. Values
outside those fixed domains remain visible in underflow/overflow counters. The
canonical fields are:

- `num_select_rows`, `num_select_valid_rows`, and `reason_counts`
- `selected_draft_length_histogram`
- `raw_draft_tail_length_histogram`
- `consumable_draft_tail_length_histogram`
- `logical_delta_histogram`
- `pending_prefix_length_histogram`
- `num_publish_seq_initial`, `num_publish_seq_same`, and
  `num_publish_seq_advance`
- `num_pending_prefix_fast_forwards`: landing events where a value-matching
  drafter span confirmed one or more residual pending-prefix tokens. An event
  may make partial progress and leave a residual pending prefix without
  publishing a suffix. Several landing updates may fast-forward between two
  selector observations, so this count is non-negative but not bounded by
  `num_select_rows`
- `num_protocol_errors`
- `num_seqlock_retry_rows`: selector rows that observed an in-progress or
  changed per-seat writer version and spun until a stable snapshot was available
- `num_seqlock_retries`: total failed seqlock read attempts across those rows
- `max_seqlock_retries`: largest failed-attempt count for one selector row in
  the window

The three seqlock fields form one optional atomic schema extension for legacy
saved windows. In current windows all three are present. Retry rows cannot exceed
selector rows, total retries cannot be smaller than retry rows, and the maximum
is zero exactly when retry rows are zero.

`publish_seq` is the current request epoch's drafter-APPEND arrival sequence.
OPEN and CLOSE reset it to `-1`; verifier-local VERIFY_COMMIT updates do not
advance it. Therefore `same` means no new current-epoch drafter frame arrived
between selector observations, while `advance` means at least one did.
Publish-sequence counters may sum to less than `num_select_rows`: a row with
`publish_seq < 0` has no freshness classification. Plots expose the difference
as `unavailable` instead of renormalizing the reported classes to 100%.

Every transport latency histogram has this representation:

```json
{
  "count": 30,
  "sum_us": 415.0,
  "bucket_upper_bounds_us": [5, 10, 20],
  "bucket_counts": [2, 8, 19, 1]
}
```

The bucket counts are non-cumulative. For `N` finite upper bounds there are
`N + 1` counts; the last count is the unbounded overflow bucket. `count` must
equal `sum(bucket_counts)`. Offline consumers merge raw counts and `sum_us`
before calculating mean or quantiles. They must not average window p50/p95
values. A quantile that falls in the overflow bucket is unknown, not the last
finite bound.

The communication time-series plot processes samples in `sample_id` order and
assigns each `(target_id, dp_rank, window_id)` to the first poll where it is
seen. At one observer point it merges every newly observed, in-boundary engine
window for that target and DP rank, then plots the exact weighted mean
`sum(sum_us) / sum(count)` at `collected_wall_time`. A repeated history window
is not consumed again. No new window, or a merged count of zero, is a gap rather
than zero. This plot-level grouping does not alter the raw engine histograms or
the whole-run merged p50/p95 values in the report.

Both roles report `num_draft_result_frames` and
`num_draft_result_tokens`. Drafter-only fields are
`draft_send_queue_latency_us` and `draft_send_queue_depth_max`. Verifier-only
fields are `draft_receive_to_gpu_publish_enqueue_latency_us`, sampled
`draft_gpu_publish_completion_latency_us`,
`draft_transport_one_way_latency_us`,
`draft_result_ready_to_receive_latency_us`, `gpu_publish_staging_slots_max`,
`clock_sync_valid`, `clock_error_bound_us`,
`num_clock_sync_valid_peers`, and `num_clock_sync_invalid_peers`.

`clock_sync_valid` means every configured peer is calibrated at window-drain
time; it is not a gate for the complete histogram. One-way and
result-ready-to-receive samples are admitted only under a valid matching epoch,
then retained even if recalibration starts before the window is drained.
Consequently a retained histogram may have positive `count` while the current
valid-peer gauge is zero. `clock_error_bound_us` is then the maximum error bound
associated with the retained samples. Consumers retain both calibrated
histograms whenever their own count is positive. The communication time-series
plots send-to-receive and labels it as record-time calibrated;
result-ready-to-receive remains available in raw telemetry and the whole-run
report but is not plotted. A raw timestamp from another host is never sufficient
by itself; an uncalibrated peer contributes no latency sample rather than a zero.

Across decode windows, peer counts are gauges rather than events. Derived
summaries report their min/max and all-valid window count; they do not sum them.

Plot `num_running_reqs` directly from successful samples as “Running batch size
over time” (Chinese report: “运行 BS 随时间变化”). Use one line per `target_id`,
a linear y-axis whose lower bound is zero, and the observer's
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

The observer does not persist startup `/model_info` or `/server_info` snapshots.
