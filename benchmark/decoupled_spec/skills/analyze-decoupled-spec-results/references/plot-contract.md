# Single-Run Plot Contract

Run these independent scripts for one `RUN_DIR`:

- `plot_latency.py` reads `client/request_metrics.csv` and writes
  `plots/request_latency.{svg,png}` plus
  `plots/request_latency_manifest.json`.
- `plot_speculative.py` reads `client/request_metrics.csv` and plots request-level
  accept rate, accept length, and valid draft tail length (the saved
  `proposed_draft_length`, excluding the bonus token). It writes
  `plots/request_speculative.{svg,png}` when speculative values are available,
  plus `plots/request_speculative_manifest.json` in all cases.
- `plot_observability.py` reads `observability/samples.jsonl` and the optional
  client formal window, then writes `observability/plots/overview.{svg,png}`
  plus `observability/plots/plot_manifest.json`. Its required service-level
  view is “Running batch size over time” from sampled `num_running_reqs`, with
  separate verifier and drafter lines. When fixed decode windows are present,
  it also writes `observability/plots/decode_metrics.{svg,png}` with iteration
  latency for both roles and verifier-only valid draft tail length and accept
  length on aligned time axes.
- `generate_report.py` reads the saved summary/config/provenance files and the
  preceding manifests, then writes `plots/run_report.md` plus
  `plots/run_report_manifest.json`.

Use this order:

```bash
python benchmark/decoupled_spec/plot/plot_latency.py --run-dir <RUN_DIR>
python benchmark/decoupled_spec/plot/plot_speculative.py --run-dir <RUN_DIR>
python benchmark/decoupled_spec/plot/plot_observability.py --run-dir <RUN_DIR>
python benchmark/decoupled_spec/plot/generate_report.py --run-dir <RUN_DIR>
```

Each manifest records source paths and SHA-256 values for exactly one derived
artifact. Regenerating a plot is valid only when the saved source artifacts
remain authoritative; do not edit a source result merely to improve a figure.
This directory intentionally provides no cross-run comparison script.

For iteration latency and running batch size, use ordinary linear y-axes
starting at zero; never use a logarithmic scale. If clearly abnormal large
iteration latency values are hidden in a presentation plot, leave the raw
windows untouched and record the exact filter, excluded count, window IDs,
times, and values in the manifest or report. Do not zero-fill or interpolate a
missing running-BS sample. `num_running_reqs` is an instantaneous sampled value;
it is not decode-window `mean_batch_size`.
