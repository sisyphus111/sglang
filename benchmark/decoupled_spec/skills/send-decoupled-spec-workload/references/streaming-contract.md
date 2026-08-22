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

- TTFT: formal batch start to the first event that increases completion tokens
  for that request.
- TPOT: `(last_token_time - first_token_time) / (completion_tokens - 1)` when
  at least two completion tokens are observed.
- E2E latency: formal batch start to the last token-increase event for that
  request.
- `batch_elapsed_s`: formal batch start until the complete SSE stream ends.
- `output_tokens_per_s`: total completion tokens divided by batch elapsed time.

These are client-observed HTTP streaming timings. They include server-side
work, transport, buffering, and client receive/parse effects; they are not pure
GPU or decoupled-IPC durations.
