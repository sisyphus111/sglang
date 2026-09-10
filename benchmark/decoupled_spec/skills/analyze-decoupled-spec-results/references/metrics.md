# Fixed Client Metric Semantics

Read the canonical fixed schema first:
[client-artifact-contract.md](../../send-decoupled-spec-workload/references/client-artifact-contract.md).
Do not add, rename, or reinterpret result fields unless the user explicitly
requests a contract change.

## Request Metrics

`client/requests.csv` contains one row per request, sorted by
`batch_row_index`. The main scalar metrics are:

| Field | Definition |
| --- | --- |
| `prompt_len` | Number of exact input IDs sent for this request |
| `resp_len` | Number of exact output IDs returned for this request |
| `spec_verify_ct` | Request verify count |
| `valid_draft_len` | Actual proposed drafts divided by verify count; excludes the bonus token |
| `acc_len` | Output tokens divided by verify count; includes the bonus token |
| `e2e_latency_s` | Formal batch start to the request's last token-increase event, in seconds |

The three per-position columns are compact JSON arrays. Position `i` counts
verify rows where draft position `i` was actually present/correct; its rate is
`correct/proposed`, or null when its proposed count is zero.

## Batch Metrics

`client/batch.json` is intentionally small:

| Field | Definition |
| --- | --- |
| `output_tokens` | Sum of request `resp_len` |
| `batch_elapsed_latency_s` | Maximum request `e2e_latency_s` |
| `batch_thpt` | `output_tokens / batch_elapsed_latency_s`, in tokens/s |
| `mean_valid_draft_len` | Arithmetic mean of request `valid_draft_len` |
| `acclen` | Arithmetic mean of request `acc_len` |

There is no `requests_per_s`, TTFT, or TPOT result in this fixed contract. Do
not reconstruct one as a first-class benchmark metric.

## Content

`client/content.json` preserves exact rendered inputs and final outputs.
Use `input_ids`/`output_ids` for token-level equality and
`input_text`/`output_text` for human inspection. The input text is the rendered
chat-template result, not the raw dataset prompt.

## Attribution Boundary

Request E2E latency and batch throughput are client-observed HTTP streaming
results. A change can include server scheduling, target/draft compute,
Decoupled-Spec transport, HTTP/SSE buffering, and client receive work. Use
controlled experiment tuples plus profiler evidence before making a
component-level causal claim.

## Decode-Window Metrics

Service-side `decode_metrics_windows` are fixed-iteration windows, not Client
request records. For a formal-request statistic, follow the
[first decode-window boundary rule](../../observe-decoupled-spec-run/references/timing-boundaries.md#first-decode-window-boundary):
deduplicate by `(target_id, dp_rank, window_id)`, select by formal `end_time`,
then exclude the earliest selected window for each `(target_id, dp_rank)` before
aggregating. The current schema cannot prove that this first window began after
the Client boundary; it may include an earlier request, flush, or idle gap.

For iteration latency, weight every retained `iter_latency_ms` by its
`num_decode_iters` within one run before comparing or averaging runs. Record the
boundary-window exclusion separately from presentation outliers. Do not apply
this exclusion to Client-owned `batch.json.acclen` or
`batch.json.mean_valid_draft_len`.
