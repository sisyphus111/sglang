# Streaming Batch Contract

## Request

The client sends one `POST /generate` payload:

```text
rid:             one request ID per batch member
input_ids:       List[List[int]] with exactly batch.size entries
sampling_params: one mapping per batch member
stream:          true
```

Every per-request sampling mapping receives `max_new_tokens` from that
request's configured output length. Other generation settings come from the
resolved client config.

## Response

Each SSE JSON event must contain an integer `index` in
`[0, batch.size)`. The client keeps the last event for each index and records
token-increase timestamps. The stream must end with `[DONE]`; missing indices,
invalid indices, incomplete frames, server errors, or absent `[DONE]` are hard
failures.

## Timing

- `requests.csv.e2e_latency_s`: formal batch start to the last token-increase
  event for that request, in seconds.
- `batch.json.batch_elapsed_latency_s`: maximum request E2E latency.
- `batch.json.batch_thpt`: total output tokens divided by that maximum E2E
  latency, in tokens/s.

The full SSE stream bounds remain in `observer/bench_timeline.json` as
`client_started_wall_time` and `client_finished_wall_time`, but they are not a
business-result field. The fixed contract has no TTFT or TPOT.
E2E latency includes server processing, transport, buffering, and client
receive/parse effects; it is not a pure GPU or Decoupled-Spec IPC duration.
