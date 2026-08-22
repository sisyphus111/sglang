# Loads Sample Schema

`observability/samples.jsonl` contains one record per role per sampling round.

## Wrapper fields

| Field | Meaning |
| --- | --- |
| `sample_id` | Sampling-round index shared by the concurrent role requests |
| `role` | Collector target name, normally `verifier` or `drafter` |
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

- `num_running_reqs`
- `num_waiting_reqs`
- `gen_throughput`
- `token_usage`
- verifier `speculative.accept_length`
- verifier `speculative.accept_rate`
- verifier `speculative.draft_occupancy_rate`
- verifier `speculative.proposed_draft_length`
- `decode_metrics_windows`, a bounded history of completed fixed-iteration
  decode windows. Each named window contains:
  - `window_id` and engine wall-clock `end_time`
  - `num_decode_iters` and `iter_latency_ms`
  - raw `num_decode_rows` and `sum_context_lens`
  - derived `mean_batch_size` and `mean_context_length`
  - raw `num_verify_rows`, `num_accept_tokens`, and `num_proposed_drafts`
  - derived `accept_length` and `proposed_draft_length` when speculative
    verification is active

The same bounded history can appear in consecutive HTTP samples. Offline tools
must deduplicate it by `(role, dp_rank, window_id)` and reject conflicting copies.
`proposed_draft_length` is the valid draft length actually presented to each
verify request-row; it excludes the bonus token.

Field availability depends on the server's enabled metrics and requested
`loads.include` groups. Preserve the complete payload rather than projecting it
to only these fields.

Startup snapshots under `observability/startup/<role>/` use the same wrapper
shape with `sample_id=-1` for `/model_info` and `/server_info`.
