# Timing Boundaries

## Required ordering

```text
both servers HTTP-ready
        │
collector starts
        │ successful baseline sample from each role
        ▼
client formal_window.started_wall_time
        │ one streaming batch is active
        ▼
client formal_window.finished_wall_time
        │ successful trailing sample from each role
        ▼
collector stops and writes summary
        │ plot_observability.py reads stable artifacts
        ▼
observability overview and manifest exist
```

Wait at least one configured sampling interval after the client completes when
necessary to obtain the trailing sample. Stop the collector gracefully so it
writes `summary.json` and final status. Run plotting only after the raw sample
stream is stable.

## Clock semantics

The collector and client use wall time for cross-process alignment. Client
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
trends at that fixed iteration granularity. It still cannot decompose a
speculative round, CUDA callback, IPC copy, or event wait that lasts
microseconds or milliseconds.
