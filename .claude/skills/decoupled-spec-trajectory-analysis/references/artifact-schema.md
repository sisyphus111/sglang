# Artifact Schema

The scripts use one result directory per experiment matrix:

```text
<run-dir>/
  config.toml
  manifest.json
  status.jsonl
  logs/<case>.log
  runs/<case>/summary.json
  analysis/
    decode_points.csv
    decode_points_filtered.csv
    decode_points_smooth.csv
    decode_points_trajectory.csv
    decode_points_trajectory_smooth.csv
    controller_switches.csv
    profile_costs.csv
    case_summary.csv
    case_summary.json
    step_occupancy.csv
    e2e_summary.csv
    e2e_summary.json
    speedup_summary.csv
    report.md
    trajectory_<case>.{svg,png}
    static_ap{0,1}_latency_profile_gap_{raw,smooth}.png
    dynamic_ap{0,1}_observed_vs_modeled_{raw,smooth}.png
    analysis_metadata.json
```

`decode_points.csv` is the normalized source of truth. Important columns are:

- case identity: `label`, `allow_partial`, `dynamic`, `max_step`
- runtime state: `point_index`, `elapsed_s`, `batch_size`, `ctx_per_req`
- observation: `observed_itl_ms`, `observed_throughput_tok_s`, `accept_len`
- model: `modeled_itl_ms`, `modeled_throughput_tok_s`, `model_source`
- controller model: `controller_ema_expected_tokens`,
  `controller_ema_modeled_throughput_tok_s`
- profile match: `profile_cost_ms`, `runtime_cpu_overhead_ms`,
  `matched_profile_batch_size`, `matched_profile_ctx_len`

For dynamic logs, `model_source=scheduler_modeled_throughput` means the model values
come from the selected runtime state printed by scheduler INFO and already contain
the runtime CPU overhead. For static logs, `model_source=profile_lookup_plus_overhead`
means the analyzer performed controller-aligned profile lookup and added the configured
overhead.

`controller_ema_expected_tokens` is the one controller state value that cannot be
recovered from BS/step/ctx/profile fields. The analyzer derives
`controller_ema_modeled_throughput_tok_s` as
`batch_size * controller_ema_expected_tokens * 1000 / modeled_cost_ms`. Older logs
without this scalar retain empty controller-model columns. The controller EMA does
not define a separate iteration latency: it uses the same selected-tier modeled cost
already represented by `modeled_itl_ms`.

`trajectory_<case>.svg` (with a report-scale PNG companion) is the direct runtime view: raw and centered-smoothed
throughput, the controller EMA throughput model when available, raw and
centered-smoothed acceptance length, and active draft length on one reconstructed
decode-time axis. Its source CSVs remove only latency
outliers, retaining small-batch tail points; full-batch filtering applies only
to the separate profile-fit figures. If the scheduler stream has no
`accept_len`, that panel is explicitly marked unavailable; the analyzer does not
replace it with zero or with a normal-decoding assumption.

`controller_switches.csv` preserves the complete score string and also extracts each
candidate into `candidate_scores_json`, including expected tokens, costs, profile
costs, legacy acceptance rates or current supply/conditional-accept rates, and the
selected candidate.

`e2e_summary.csv` reads each completed `runs/<case>/summary.json` and records output
throughput, generation time, generated tokens, and aggregate acceptance metrics. It is
kept separate from scheduler-point means because those quantities have different
weighting and should not be compared as if they were identical.

`speedup_summary.csv` compares each dynamic case with the static case having the
same maximum step and with the best measured static case under the same
`allow_partial` setting. `report.md` is a compact index over these structured files
and figures; CSV/JSON remain authoritative.
