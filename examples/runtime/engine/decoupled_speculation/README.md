# Decoupled Speculation Example

This directory has two CLI entrypoints for decoupled speculative decoding:

- `multi-node.py`: run the Ray-backed multi-node benchmark from either `--prompt` or `--dataset-path`.
- `single-node.py`: run the local-process single-node benchmark without Ray.
- `common/`: shared helpers split by function: runtime/Ray topology, prompt loading, metrics/output, and shared types.

Decoupled-spec engines use a two-stage runtime rendezvous. Verifier and drafter
instances start first, bind free ZMQ endpoints locally, and publish the bound
endpoints through engine init metadata. The helper then builds the full mesh and
configures peers before generation starts: every verifier connects to every
drafter control endpoint, and every drafter connects to every verifier result
endpoint. Users no longer need to reserve decoupled result/control ports in
advance; for multi-node runs, make sure each node's automatically detected
local IP is mutually reachable, or override it with `SGLANG_HOST_IP`/`HOST_IP`.
When multiple verifier replicas are used,
`--batch-size` must be divisible by the verifier replica count; each verifier
receives one equal contiguous slice of the batch.

For verifier DP attention in either script, set `--target-dp-size` and
`--target-enable-dp-attention` (or `--enable-dp-attention`). In that mode,
`--target-tp-size` is interpreted as the attention TP size per DP lane, and the
SGLang engine `tp_size` becomes `target_tp_size * target_dp_size`. For example,
`--target-tp-size 8 --target-dp-size 4 --enable-dp-attention --target-ep-size
32` launches a 32-GPU verifier engine with DP attention.
DP attention uses TCP for internal SGLang IPC. Without an available port pool,
the legacy path derives those TCP ports from `dist-init`, so manually supplied
`--dist-init-port` or `--reserved-ports` values should leave a contiguous
6-port block for each verifier or baseline engine instance. On platforms that
expose usable ports as `PORT1`, `PORT2`, and so on, pass `--target-use-env-ports`;
the launcher reads those non-contiguous
ports from each verifier rank-0 node and passes them to SGLang as the target
engine's available port pool. Ports already bound by the local Ray runtime or
reserved for dist-init are skipped before the pool is sliced for each engine.

Common modes:

```bash
# Single prompt, compare decoupled speculation against normal decode.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --prompt "Write a short haiku about distributed systems." \
  --target-model-path Qwen/Qwen3-32B \
  --draft-model-path Qwen/Qwen3-0.6B \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 128

# Single prompt, compare against SGLang builtin colocated MTP/EAGLE.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --prompt "Write a short haiku about distributed systems." \
  --baseline mtp \
  --target-model-path Qwen/Qwen3-32B \
  --draft-model-path Qwen/Qwen3-0.6B \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 128

# Dataset batch, decoupled speculation only.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --dataset-path /path/to/prompts.parquet \
  --batch-size 16 \
  --baseline none \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 8 \
  --draft-tp-size 1 \
  --max-new-tokens 1024

# Multi-node mesh. Here 16 GPUs are reserved for verifier replicas and 4 GPUs
# are reserved for drafter replicas; the remaining GPUs stay idle.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --dataset-path /path/to/prompts.parquet \
  --batch-size 64 \
  --baseline none \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --nnodes 4 \
  --n-gpu-per-node 8 \
  --target-tp-size 8 \
  --draft-tp-size 1 \
  --verify-ngpus 16 \
  --draft-ngpus 4 \
  --dist-init-port 30000 \
  --max-new-tokens 1024

# Print responses and write per-mode CSV/JSON outputs.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --prompt "Explain speculative decoding." \
  --show-responses \
  --output-dir ./decoupled_spec_outputs \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 256

# Write decoupled/decode/MTP tracer CSV files under one directory.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --prompt "Explain speculative decoding." \
  --baseline mtp \
  --spec-trace-dir ./spec_traces \
  --output-dir ./decoupled_spec_outputs \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 256
```
