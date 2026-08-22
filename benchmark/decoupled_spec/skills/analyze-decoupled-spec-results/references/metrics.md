# Metric Semantics

## Batch metrics

| Field | Definition |
| --- | --- |
| `batch_elapsed_s` | Formal batch POST start until the complete SSE stream ends |
| `output_tokens_per_s` | Sum of completed output tokens divided by `batch_elapsed_s` |
| `prompt_tokens` | Sum of saved request prompt lengths |
| `completion_tokens` | Sum of verifier-reported completion-token counts |

There is no `requests_per_s` result in this benchmark. Do not reconstruct or
report it as a first-class metric.

## Request latency

| Field | Definition |
| --- | --- |
| `ttft_ms` | Formal batch start to the first token-increase SSE event for one request |
| `tpot_ms` | First-to-last token-increase duration divided by `completion_tokens - 1` |
| `e2e_latency_ms` | Formal batch start to the last token-increase event for one request |

The summary reports mean, p50, p95, and p99 over request-level non-null values.
The mean TPOT is the arithmetic mean of per-request TPOT values, not a
token-weighted batch duration.

`1000 / mean_tpot_ms` is an approximate single-request steady-generation token
rate. It is not interchangeable with `output_tokens_per_s`, which measures the
whole batch over its complete HTTP window.

## Speculative metrics

| Field | Aggregate definition |
| --- | --- |
| `spec_verify_ct` | Sum of request verify counts |
| `spec_num_proposed_drafts` | Sum of draft tokens actually presented to the verifier |
| `spec_num_correct_drafts` | Sum of correct draft tokens |
| `spec_accept_rate` | Correct drafts divided by proposed drafts |
| `spec_draft_occupancy_rate` | Actual proposed drafts divided by `verify_ct * K` |
| `spec_proposed_draft_length` | Actual proposed drafts divided by `verify_ct` |
| `spec_accept_length` | Total completion tokens divided by verify count |

Decoupled verification also reports the proposed-length histogram and per-position
proposed/correct counts. Position `i` counts verify rows where draft position `i`
was actually present/correct; their ratio is `spec_accept_rate_by_position[i]`.

These values describe speculative effectiveness, not the duration of an
individual verify round. Null values mean the server response did not provide
enough information; do not convert them to zero unless zero is actually
recorded.

## Attribution boundary

All latency values are observed at the client. A change can include server
scheduling, target/draft compute, decoupled transport, HTTP/SSE buffering, and
client receive work. Use controlled experiment tuples plus profiler evidence
before making a component-level causal claim.
