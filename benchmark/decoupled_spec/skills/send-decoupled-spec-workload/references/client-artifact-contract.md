# Fixed Client Result Contract

This contract defines the complete human-facing output of one successful
Decoupled-Spec client batch. It is intentionally fixed. Do not add, remove,
rename, reorder, relocate, or reinterpret a file or field unless the user
explicitly requests a contract change.

## Boundary

The three business-result files live together and alone under:

```text
<RUN_DIR>/client/
├── requests.csv
├── batch.json
└── content.json
```

`client/` must contain exactly these three files and no subdirectory. Effective
server and client configuration lives in `<RUN_DIR>/config.json`; the formal
request boundary lives in
`<RUN_DIR>/observer/bench_timeline.json.client_{started,finished}_wall_time`.

The following legacy business artifacts are forbidden:

- `client/sampled_requests.jsonl`
- `client/stream_timing_events.jsonl`
- `client/raw_batch_response.json`
- `client/responses.jsonl`
- `client/request_metrics.csv`
- `client/summary.json`

## `requests.csv`

One row represents one request. Rows are sorted by `batch_row_index` ascending,
and indices are exactly `0..batch_size-1`. The header order is fixed:

```text
batch_row_index,dataset_idx,verifier_rank,prompt_len,resp_len,spec_verify_ct,valid_draft_len,acc_len,e2e_latency_s,spec_num_proposed_drafts_by_position,spec_num_correct_drafts_by_position,spec_accept_rate_by_position
```

| Column | Logical type | Meaning |
| --- | --- | --- |
| `batch_row_index` | integer | Zero-based row in this submitted batch; primary sort key |
| `dataset_idx` | integer | Zero-based physical row in the selected dataset file before optional shuffle; synthetic inputs use their generated row index |
| `verifier_rank` | integer | Manifest verifier rank that served the complete batch; direct-URL mode means rank 0 |
| `prompt_len` | integer | `len(input_ids)` |
| `resp_len` | integer | `len(output_ids)` and verifier `completion_tokens` |
| `spec_verify_ct` | integer | Request verify count; positive for a successful formal Decoupled-Spec result |
| `valid_draft_len` | number | Verifier `spec_proposed_draft_length`: actual proposed drafts per verify row, excluding the bonus token |
| `acc_len` | number | Verifier `spec_accept_length`: output tokens per verify row, including the bonus token |
| `e2e_latency_s` | number | Client-observed request E2E latency in seconds |
| `spec_num_proposed_drafts_by_position` | JSON array of integers | Per-position proposed-draft counts |
| `spec_num_correct_drafts_by_position` | JSON array of integers | Per-position correct-draft counts |
| `spec_accept_rate_by_position` | JSON array of number or null | Per-position `correct/proposed`; null where proposed count is zero |

The three array cells use compact JSON, for example `[10,8,3]` or
`[1.0,0.5,null]`. Python repr, whitespace-dependent formatting, and alternate
delimiters are forbidden.

## `batch.json`

This file is one JSON object with exactly these fields in this order:

| Field | Type | Definition |
| --- | --- | --- |
| `output_tokens` | integer | Sum of `requests.csv.resp_len` |
| `batch_elapsed_latency_s` | number | Maximum request `e2e_latency_s` |
| `batch_thpt` | number | `output_tokens / batch_elapsed_latency_s`, in tokens/s |
| `mean_valid_draft_len` | number | Arithmetic mean of request `valid_draft_len` |
| `acclen` | number | Arithmetic mean of request `acc_len` |

Do not substitute the HTTP stream duration for `batch_elapsed_latency_s`.

## `content.json`

This file is one JSON list sorted by `batch_row_idx`. Each request object has
exactly these fields in this order:

| Field | Type | Meaning |
| --- | --- | --- |
| `batch_row_idx` | integer | Zero-based row in this submitted batch |
| `dataset_idx` | integer | Same physical dataset row as `requests.csv.dataset_idx` |
| `input_len` | integer | `len(input_ids)`; equals `requests.csv.prompt_len` |
| `output_len` | integer | `len(output_ids)`; equals `requests.csv.resp_len` |
| `input_ids` | array of integers | Exact tokenized request sent to the verifier |
| `input_text` | string | Rendered text after applying the configured chat template |
| `output_ids` | array of integers | Exact final output token IDs |
| `output_text` | string | Final text returned by the verifier |

List position, `batch_row_idx`, and `requests.csv.batch_row_index` must agree.

## Cross-File Invariants

For a successful run:

```text
resolved batch.size
  = requests.csv row count
  = content.json item count
```

For every batch row:

```text
requests.batch_row_index = content.batch_row_idx
requests.dataset_idx = content.dataset_idx
requests.prompt_len = content.input_len = len(content.input_ids)
requests.resp_len = content.output_len = len(content.output_ids)
requests.valid_draft_len
  = sum(requests.spec_num_proposed_drafts_by_position) / spec_verify_ct
requests.acc_len = content.output_len / spec_verify_ct
```

The producer validates the field set, ordering, types, formulas, and cardinality
before writing these files. Built-in consumers read this fixed schema, and unit
tests freeze the producer contract. There is no separate centralized artifact
audit; do not change the contract unless the user explicitly requests it.
