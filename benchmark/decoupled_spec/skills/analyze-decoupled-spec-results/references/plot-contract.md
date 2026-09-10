# Single-Run Plot Contract

Run these independent scripts for one `RUN_DIR`:

- `plot_latency.py` reads `client/requests.csv` and writes
  `plots/request_latency.png`. The fixed request contract exposes
  E2E latency in seconds; TTFT and TPOT are not result fields.
- `plot_speculative.py` reads `client/requests.csv` and plots request-level
  accept length and valid draft tail length. It writes
  `plots/request_speculative.png`; missing fixed fields fail closed.
- `plot_observability.py` reads `observer/samples.jsonl` and
  `observer/bench_timeline.json`, then writes `plots/overview.png`. Its required service-level
  view is “Running batch size over time” from sampled `num_running_reqs`, with
  separate verifier and drafter lines. When fixed decode windows are present,
  it also writes `plots/decode_metrics.png` with iteration
  latency for both roles and verifier-only valid draft tail length and accept
  length on aligned time axes. When decoupled-spec transport histograms are
  present, its communication panels assign every engine window to its first
  observer poll and plot `sum(sum_us) / sum(count)` across all new windows at
  that poll; repeated history and zero-count polls remain gaps.
- `generate_report.py` reads `config.json`, the fixed client results, raw
  observer samples, and generated PNGs, then writes `plots/run_report.md`.

Use this order:

```bash
python benchmark/decoupled_spec/plot/plot_latency.py --run-dir <RUN_DIR>
python benchmark/decoupled_spec/plot/plot_speculative.py --run-dir <RUN_DIR>
python benchmark/decoupled_spec/plot/plot_observability.py --run-dir <RUN_DIR>
python benchmark/decoupled_spec/plot/generate_report.py --run-dir <RUN_DIR>
```

Regenerating a plot is valid because the saved Client and Observer files remain
authoritative; do not edit a source result merely to improve a figure. This
directory intentionally provides no cross-run comparison script.

## User-facing delivery

Generate one high-resolution PNG per figure and inspect it at report scale and
full resolution before delivery.

For every non-negative user-facing metric, use an ordinary linear y-axis whose
lower bound is zero. A signed delta/effect plot must include a visible zero
baseline and still use a linear scale. Do not use logarithmic, broken, or
truncated y-axes.

Clearly abnormal outliers must be excluded from the presentation series so
they do not destroy the useful scale. Use a deterministic robust rule, with
`value > Q3 + 3 * IQR` as the default for a sufficiently populated series.
Leave raw windows untouched and record the rule, threshold, excluded count,
window IDs, times, and values in the report. Do not zero-fill
or interpolate missing samples. `num_running_reqs` is an instantaneous sampled
value; it is not decode-window `mean_batch_size`.

Before plotting any formal-request field from `decode_metrics_windows`, apply
the [first decode-window boundary rule](../../observe-decoupled-spec-run/references/timing-boundaries.md#first-decode-window-boundary).
Exclude the earliest in-boundary window per `(target_id, dp_rank)` before
outlier filtering because its missing start boundary makes it potentially
cross-request. Record this exclusion independently of statistical outliers.
