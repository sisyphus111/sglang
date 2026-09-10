# Timing Boundaries

## Required ordering

```text
both servers HTTP-ready
        │
observer/bench_timeline.json.observer_started_wall_time
        │ successful zero-waiting baseline sample from every target
        ▼
observer/bench_timeline.json.client_started_wall_time
        │ one streaming batch is active
        ▼
observer/bench_timeline.json.client_finished_wall_time
        │ successful trailing sample from every target
        ▼
observer/bench_timeline.json.observer_finished_wall_time
        │ Runner completes bench_timeline.json
        │ plot_observability.py reads stable artifacts
        ▼
observer plots exist under RUN_DIR/plots
```

Runner waits until a complete Observer round starts at or after the Client
finish boundary. It then stops Observer gracefully and runs plotting only after
`observer/samples.jsonl` is stable.

The completed `bench_timeline.json` contains exactly the four wall-time
boundaries shown above plus `observer_elapsed_s`, defined as
`observer_finished_wall_time - observer_started_wall_time`. Baseline and
trailing sample IDs remain internal Runner barriers and are not persisted in
the timeline.

## Clock semantics

The observer and client use wall time for cross-process alignment. Client
token timing and batch elapsed time use a monotonic clock internally. Do not
subtract monotonic timestamps from different processes.

`collected_wall_time` is taken before the HTTP request and `latency_ms` measures
that request. A sample is not an instantaneous GPU snapshot; its payload arrives
after the recorded wall time by approximately the collection latency.

## Resolution boundary

With `interval_s=1`, the time series resolves service behavior over seconds.
Each successful `/v1/loads` response contributes one instantaneous
`num_running_reqs` point per role for the “Running batch size over time” plot.
Do not reinterpret it as a decode-window average or fill a missed poll with
zero.

The engine's bounded `decode_metrics_windows` history preserves completed
`decode_log_interval`-step windows even when several finish between two HTTP
polls. It can show iteration latency, valid draft tail length, and accept length
trends at that fixed iteration granularity. The engine may additionally freeze
raw decoupled-spec selector counters and transport latency histograms into the
same windows. Those histograms preserve a microsecond latency distribution. The
communication plot merges all engine windows first observed at one poll using
their raw sums and counts, so it intentionally resolves their exact mean at
observer polling granularity. It does not reconstruct an individual frame
timeline or causally align a frame with one selector result.

## First decode-window boundary

The current `decode_metrics_windows` schema records `end_time`, but not a
trustworthy window start time. The first window whose end falls inside one
formal Client interval may therefore have started before
`client_started_wall_time`. It can span the preceding request, cache flush, or
idle gap. Treating it as a fully in-request window can severely inflate
`iter_latency_ms` and can mix earlier-request counters into the formal result.

For every per-request aggregate or comparison derived from decode windows:

1. Deduplicate windows by `(target_id, dp_rank, window_id)`.
2. Keep windows whose `end_time` is within the formal Client boundary.
3. Group by `(target_id, dp_rank)` and order by `(end_time, window_id)`.
4. Exclude exactly the earliest in-boundary window from each group before
   weighting, averaging, plotting, or applying a statistical outlier rule.
5. Record the excluded target, DP rank, window ID, end time, and relevant metric
   values. Preserve the raw Observer artifact unchanged.

This boundary exclusion applies to formal-request views of every field carried
by that decode window, including iteration latency, window-level acceptance,
selector counters, and transport histograms. It does not remove Client-owned
request metrics such as `batch.json.acclen` or `mean_valid_draft_len`, and it
does not alter instantaneous Observer samples such as `num_running_reqs` or
`num_waiting_reqs`. If no complete window remains after the exclusion, report
the decode-window metric as unavailable rather than restoring the boundary
window.

This is a boundary-validity rule, not an outlier rule. Apply it before IQR or
other presentation filtering. A future schema may retain the first window only
when a recorded, trustworthy start time proves the entire window began at or
after `client_started_wall_time`.

For fixed-BS queue validity, additionally retain the contiguous full-batch
verifier decode segment, using `num_decode_rows == batch.size * num_decode_iters`.
The first full window supplies the start boundary; only subsequent complete
windows are eligible. Bound the end by the earliest Client request completion
and stop at the first non-full window. Prefill/batch-fill and drain queues are
outside this measurement interval. The validator records these bounds as
`decode_queue_window` and requires queue samples from both roles inside it.
Missing evidence is invalid, not an implicit zero queue.

Within one process, transport stages use a local monotonic clock. Cross-host
send-to-receive and result-ready-to-receive samples are recorded only when the
wire timestamp belongs to a calibrated peer epoch with a finite error bound.
`clock_sync_valid=false` may mean that another configured peer is invalid or
that a new calibration is in progress at window-drain time; it does not erase
samples admitted under an earlier valid epoch in that window. The accompanying
error bound covers those retained samples, while valid/invalid peer counts are
current drain-time gauges. Never subtract uncalibrated monotonic timestamps
from two hosts. A missing peer calibration is a missing sample, not a
zero-latency observation.
