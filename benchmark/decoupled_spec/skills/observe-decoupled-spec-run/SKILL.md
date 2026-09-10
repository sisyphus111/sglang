---
name: observe-decoupled-spec-run
description: Collect and validate periodic verifier and drafter service telemetry for a decoupled-spec benchmark run. Use when the user asks to monitor a live run or inspect its service-level time series; it does not generate traffic or reconstruct per-frame traces.
---

# Observe a Decoupled-Spec Run

Create a service-level time series that covers the formal client request while
leaving the benchmark traffic path unchanged.

## Collect

Read [references/timing-boundaries.md](references/timing-boundaries.md), then:

1. Validate the observability config with `client/observer.py --check`, passing the
   ready server manifest.
2. In a standard benchmark, let `runner.py` start the Observer subprocess with
   `--server-manifest` after every engine is HTTP-ready and before the formal
   client request.
3. Require at least one successful `/v1/loads` sample with
   `num_waiting_reqs == 0` from every verifier and drafter engine.
4. After the client finishes, retain at least one successful trailing sample
   from both roles, then terminate the observer gracefully.
5. Stop the observer cleanly, inspect its stdout summary, and run
   `scripts/validate_samples.py` against the two saved observer files.
6. Generate the derived overview explicitly with
   `benchmark/decoupled_spec/plot/plot_observability.py`.
7. Require the derived observability plots to show:
   - `iteration latency` from `decode_metrics_windows`, with separate verifier
     and drafter lines;
   - verifier `valid draft tail length` and `accept length` from the same decode
     windows; and
   - `running batch size over time` from each role's sampled
     `num_running_reqs`.

   Use ordinary linear y-axes starting at zero for all non-negative metrics,
   including latency, running batch size, tail length, queue depth, and
   transport timing. Signed effects must include a visible zero baseline. Do
   not use logarithmic, broken, or truncated y-axes. Exclude clearly abnormal
   presentation outliers using a deterministic rule (default: `Q3 + 3 * IQR`),
   preserve the raw windows, and record the rule, threshold, count, window IDs,
   times, and values in the report.
8. When decode windows expose `decoupled_spec`, require
   `plots/decoupled_spec_metrics.png`. It must use `Batch runtime (s)` for a
   benchmark formal window (`Server runtime (s)` otherwise) and show:
   - verifier selector reasons as a stacked row share;
   - raw, consumable, and selected draft-tail length over time;
   - logical delta, drafter-arrival sequence freshness, and pending-prefix
     fast-forward recovery over time;
   - independent exact-mean panels for drafter send-queue latency, calibrated
     send-to-receive latency from healthy peers, and verifier
     receive-to-GPU-publish-enqueue latency; and
   - sampled GPU publish-completion latency when it is available.

   Assign each deduplicated engine window to the first observer sample that
   contains it. For each `(target_id, dp_rank, sample_id)`, merge all newly
   observed in-boundary windows by summing their raw histogram `sum_us` and
   `count`, then plot the weighted exact mean `sum(sum_us) / sum(count)` at that
   sample's `collected_wall_time`. A poll with no new window or a merged count
   of zero remains a gap. Do not reconsume repeated bounded-history windows or
   average per-window means. Plot each calibrated cross-host histogram when its
   own count is positive, label it as record-time calibrated, and report
   drain-time valid/invalid peer ranges; current all-peer validity is not a plot
   gate. Plot pending-prefix recovery as events per select row on its own
   unbounded axis; it is not a selector-row percentage.

The observer may query `/model_info`, `/server_info`, and `/v1/loads`; it must
not call `/generate`.

Every successful formal-window sample must report `num_waiting_reqs == 0` for
every manifest engine. A positive value means the requested batch was not
served concurrently and invalidates the performance run. Optional server logs
may provide additional queue evidence, but their absence is not an artifact
failure.

Derive every per-engine and per-role summary from `observer/samples.jsonl`.
Window identity is `(target_id, dp_rank, window_id)`; two replicas may
legitimately reuse the same local window ID. Figures and human-facing reports
must call `iter_latency_ms` “iteration latency”.

For formal-request decode-window summaries, apply the first-window boundary
rule from [references/timing-boundaries.md](references/timing-boundaries.md)
before aggregating any window field. Never use the earliest in-boundary window
as request-local evidence merely because its `end_time` falls inside the Client
boundary.

Treat sampled `num_running_reqs` as the server's running batch size at that
observer instant. Plot verifier and drafter separately on one time axis. Do
not substitute decode-window `mean_batch_size`, fill a missing HTTP sample with
zero, or interpolate across a collection failure.

## Interpret

Read [references/loads-schema.md](references/loads-schema.md) when interpreting
payload fields. Preserve raw `samples.jsonl`; summaries and plots are derived
views and are not persisted as additional observer artifacts.

Use the time series for second-scale queue, request, throughput, token-usage,
and speculative trends. Engine-frozen transport histograms additionally expose
the measured stage distributions described in the loads schema; they do not
recover per-frame causality. Do not infer uninstrumented CUDA stream waits, C6
duration, or GraphExec spacing from the observer. Those require a profiler
trace with a narrower timing boundary.

## Output Contract

Return the sampling interval, formal-window duration, per-engine
successful/error sample counts, coverage before/inside/after the formal window,
maximum sample gap, maximum formal-window waiting requests, per-engine unique
decode-window counts and iteration latency statistics, and paths for the running batch size,
valid draft tail length, accept length, and iteration latency plots. Surface
any nonzero waiting queue, missing coverage, or a missing required plot as an
observability failure even when the client request itself succeeded. When the
new decoupled-spec window section is present, also return raw merged selector
and transport counters/histograms plus the decoupled-spec metrics figure path.

The complete persisted observability directory is exactly
`observer/{samples.jsonl,bench_timeline.json}`. Observer owns `samples.jsonl`;
Runner owns the cross-component `bench_timeline.json`, which contains only the
four Observer/Client wall-time boundaries and `observer_elapsed_s`.
