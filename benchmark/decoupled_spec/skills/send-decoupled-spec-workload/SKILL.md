---
name: send-decoupled-spec-workload
description: Prepare and submit one client-side tokenized, streaming batch to an already-running decoupled-spec verifier, then collect request-level results. Use for dataset, tokenizer, chat-template, batch, generation, or SSE-client work; it does not launch model servers.
---

# Send a Decoupled-Spec Workload

Turn one client YAML plus named overrides into exactly one reproducible formal
batch and save both its inputs and streaming results.

## Prepare the Batch

Read [references/datasets.md](references/datasets.md) for the selected dataset
format and [references/chat-template-and-tokenization.md](references/chat-template-and-tokenization.md)
for prompt processing.

1. Run `client.py --check` with the exact intended CLI overrides.
2. Run `scripts/inspect_workload.py` with the same config and overrides. Treat
   tokenizer loading, dataset parsing, selected row count, and token lengths as
   preflight—not as a benchmark result.
3. Verify that `target_tokenizer.model_path` matches the target/verifier model
   family and that the resolved batch, prompt, output, template, and thinking
   settings match the requested case.

## Submit the Formal Request

Read [references/streaming-contract.md](references/streaming-contract.md).

- Start only after both servers are ready and the collector has successful
  baseline samples.
- Invoke `client-side/client.py` once with the same config, overrides, and
  shared `RUN_DIR`.
- Require `batch.size` prepared requests and one streaming `/generate` call
  containing `input_ids: List[List[int]]`.
- Require a final response for every batch index and a `[DONE]` SSE marker.

Do not send one HTTP request per sample and do not periodically call
`/generate`. Do not retokenize on the verifier for this benchmark path.

## Output Contract

Return the resolved client tuple, selected prompt-length distribution, formal
window, request/completion counts, output throughput, TTFT/TPOT/E2E summaries,
and speculative counters. Preserve all files under `RUN_DIR/client/`; on
failure, identify the first dataset, tokenizer, HTTP, SSE, or response-contract
error and do not fabricate missing metrics.
