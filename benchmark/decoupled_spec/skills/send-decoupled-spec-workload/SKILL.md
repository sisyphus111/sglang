---
name: send-decoupled-spec-workload
description: Prepare and submit one client tokenized, streaming batch to a selected verifier from a ready decoupled-spec server manifest, then collect request-level results. Use for dataset, tokenizer, chat-template, batch, generation, or SSE-client work; it does not launch model servers.
---

# Send a Decoupled-Spec Workload

Turn one client YAML plus named overrides into exactly one reproducible formal
batch and save both its inputs and streaming results.

## Prepare the Batch

Read [references/client-artifact-contract.md](references/client-artifact-contract.md)
before writing or interpreting client results. This is the fixed human-facing
schema and must not drift unless the user explicitly asks to change it.

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
4. Pass `--server-manifest <RUNTIME_DIR>/server/manifest.json` and an explicit
   `--verifier-rank`; record the selected verifier `engine_id` and HTTP URL.

## Submit the Formal Request

Read [references/streaming-contract.md](references/streaming-contract.md).

- Start only after every manifest engine is ready and the observer has a
  successful zero-waiting baseline sample for every engine.
- In a standard run, let `runner.py` invoke the Client once with the same
  config, overrides, and shared `RUN_DIR`. Use `client/client.py` directly only
  when the task is explicitly limited to Client behavior.
- Require `batch.size` prepared requests and one streaming `/generate` call
  containing `input_ids: List[List[int]]`.
- Require a final response for every batch index and a `[DONE]` SSE marker.

Do not send one HTTP request per sample and do not periodically call
`/generate`. Do not retokenize on the verifier for this benchmark path.

## Output Contract

Return the resolved client tuple, selected prompt-length distribution, formal
window, fixed request rows, batch output tokens/elapsed latency/throughput,
content rows, and speculative metrics. A successful run writes exactly
`client/{requests.csv,batch.json,content.json}` according to the fixed artifact
contract and adds the effective client config to root `config.json`. The Client
returns its request boundary to the caller; the standard Runner persists it in
`observer/bench_timeline.json` together with the Observer boundaries.
On failure, identify the first dataset, tokenizer, HTTP, SSE, or
response-contract error and do not fabricate missing metrics.
