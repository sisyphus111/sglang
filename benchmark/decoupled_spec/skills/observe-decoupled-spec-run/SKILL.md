---
name: observe-decoupled-spec-run
description: Collect and validate periodic verifier and drafter service telemetry for a decoupled-spec benchmark run. Use when the user asks to monitor a live run or inspect its service-level time series; it does not generate traffic or attribute microsecond CUDA operations.
---

# Observe a Decoupled-Spec Run

Create a service-level time series that covers the formal client request while
leaving the benchmark traffic path unchanged.

## Collect

Read [references/timing-boundaries.md](references/timing-boundaries.md), then:

1. Validate the observability config with `collector.py --check`, passing the
   ready server manifest.
2. Start the collector with `--server-manifest` only after every engine is
   HTTP-ready and before the formal client request.
3. Require at least one successful `/v1/loads` sample with
   `num_waiting_reqs == 0` from every verifier and drafter engine.
4. After the client finishes, retain at least one successful trailing sample
   from both roles, then terminate the collector gracefully.
5. Require collector status `completed`, inspect its error count, and run
   `scripts/validate_samples.py`.
6. Generate the derived overview explicitly with
   `benchmark/decoupled_spec/plot/plot_observability.py`.
7. Require the derived observability plots to show:
   - `iteration latency` from `decode_metrics_windows`, with separate verifier
     and drafter lines;
   - verifier `valid draft tail length` and `accept length` from the same decode
     windows; and
   - `running batch size over time` from each role's sampled
     `num_running_reqs`.

   Use ordinary linear y-axes starting at zero for iteration latency and
   running batch size. Do not use a logarithmic axis. A derived iteration
   latency view may remove clearly abnormal large values, but it must preserve
   the raw windows and record the exclusion rule, count, window IDs, times, and
   values in the plot manifest or report.

The collector may query `/model_info`, `/server_info`, and `/v1/loads`; it must
not call `/generate`.

Every successful formal-window sample must report `num_waiting_reqs == 0` for
every manifest engine. A positive value means the requested batch was not
served concurrently and invalidates the performance run. The artifact audit
also scans role logs so a logged transient queue cannot be hidden by the
collector's sampling interval.

Require `observability/summary.json.decode_metrics_by_target` to report every
`engine_id` independently, plus verifier/drafter aggregates under
`decode_metrics`. Window identity is `(target_id, dp_rank, window_id)`; two
replicas may legitimately reuse the same local window ID. The persisted
compatibility key may remain
`scheduler_cycle_ms`, but figures and human-facing reports must call the metric
`iteration latency`.

Treat sampled `num_running_reqs` as the server's running batch size at that
collector instant. Plot verifier and drafter separately on one time axis. Do
not substitute decode-window `mean_batch_size`, fill a missing HTTP sample with
zero, or interpolate across a collection failure.

## Interpret

Read [references/loads-schema.md](references/loads-schema.md) when interpreting
payload fields. Preserve raw `samples.jsonl`; the collector summary and
separately generated plot are derived views.

Use the time series for second-scale queue, request, throughput, token-usage,
and speculative trends. Do not infer CUDA stream wait time, IPC copy latency,
C6 duration, or GraphExec spacing from the collector. Those require a profiler
trace with a narrower timing boundary.

## Output Contract

Return the sampling interval, formal-window duration, per-engine
successful/error sample counts, coverage before/inside/after the formal window,
maximum sample gap, maximum formal-window waiting requests, per-engine unique
decode-window counts and iteration latency statistics, and paths for the running batch size,
valid draft tail length, accept length, and iteration latency plots. Surface
any nonzero waiting queue, missing coverage, or a missing required plot as an
observability failure even when the client request itself succeeded.
