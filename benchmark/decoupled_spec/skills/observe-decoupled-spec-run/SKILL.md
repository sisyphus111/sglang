---
name: observe-decoupled-spec-run
description: Collect and validate periodic verifier and drafter service telemetry for a decoupled-spec benchmark run. Use when the user asks to monitor a live run or inspect its service-level time series; it does not generate traffic or attribute microsecond CUDA operations.
---

# Observe a Decoupled-Spec Run

Create a service-level time series that covers the formal client request while
leaving the benchmark traffic path unchanged.

## Collect

Read [references/timing-boundaries.md](references/timing-boundaries.md), then:

1. Validate the observability config with
   `benchmark/decoupled_spec/common/collector.py --check`.
2. Start the collector only after both servers are HTTP-ready and before the
   formal client request.
3. Require at least one successful `/v1/loads` sample from both verifier and
   drafter before starting the client.
4. After the client finishes, retain at least one successful trailing sample
   from both roles, then terminate the collector gracefully.
5. Require collector status `completed`, inspect its error count, and run
   `scripts/validate_samples.py`.
6. Generate the derived overview explicitly with
   `benchmark/decoupled_spec/plot/plot_observability.py`.
7. When `decode_metrics_windows` are present, require
   `observability/plots/decode_metrics.{svg,png}` and inspect all five aligned
   time series: scheduler cycle, mean batch size, mean context length, valid
   draft length, and accept length.

The collector may query `/model_info`, `/server_info`, and `/v1/loads`; it must
not call `/generate`.

## Interpret

Read [references/loads-schema.md](references/loads-schema.md) when interpreting
payload fields. Preserve raw `samples.jsonl`; the collector summary and
separately generated plot are derived views.

Use the time series for second-scale queue, request, throughput, token-usage,
and speculative trends. Do not infer CUDA stream wait time, IPC copy latency,
C6 duration, or GraphExec spacing from the collector. Those require a profiler
trace with a narrower timing boundary.

## Output Contract

Return the sampling interval, formal-window duration, per-role successful/error
sample counts, coverage before/inside/after the formal window, maximum sample
gap, number of unique decode windows, and plot paths. Surface missing coverage
or a missing decode-metrics plot as an observability failure even when the
client request itself succeeded.
