# Campaign Input Contract

Campaign input is a YAML mapping with these required top-level fields:

```yaml
schema_version: 1
campaign:
  name: example-campaign
  run_name_prefix: example-run

configs:
  server_by_mode:
    nonoverlap: /path/to/unified-nonoverlap.yaml
    overlap: /path/to/unified-overlap.yaml
  client: /path/to/client.yaml
  observer: /path/to/observer.yaml
  workload_inspection: /path/to/inspection.json

axes:
  # Optional; defaults to verifier for backward compatibility.
  mode_target: verifier
  modes: [nonoverlap, overlap]
  ignore_eos: [true]
  batch_sizes: [1, 8, 16]
  output_lengths: [4096]

expected:
  target_model_path: /path/to/target
  target_tp_size: 4
  drafter_model_path: /path/to/drafter
  drafter_tp_size: 1
  random_seed: 42
  dataset_format: dapo_math_17k
  dataset_path: /path/to/data.parquet
  speculative_num_steps: 3
  speculative_eagle_topk: 1
  speculative_num_draft_tokens: 4
  verifier_replayssm_flag: enable_linear_replayssm_spec
  verifier_mamba_slots_per_request: 5
  drafter_mamba_slots_per_request: 8
  max_total_tokens: 300000
  cuda_graph_bs_decode: [1, 8, 16]
  chat_template_mode: tokenizer
  enable_thinking: true
  temperature: 1
  ignore_eos_values: [true]
  capacity_reference:
    prompt_len_min: 1
    prompt_len_max: 200
    prompt_len_sum: 1200
    verify_reserve_per_request: 4
    required_tokens: 34000
    inspection_sha256: <sha256>

execution:
  ordering: [batch_size, mode]
  results_root: /path/to/results
  max_concurrent_deployments: 1
  prerequisites: []

stop_gates:
  hard:
    require_completed_count_equals_batch_size: true
    require_completion_tokens_respect_ignore_eos: true
    require_zero_waiting_reqs: true
    min_verifier_max_total_num_tokens: 34000
    min_drafter_max_total_num_tokens: 34000
    min_verifier_mamba_cache_size: 16
    min_drafter_mamba_cache_size: 16
    min_spec_verify_ct: 1
    min_spec_num_proposed_drafts: 0
  diagnostic:
    require_nonzero_spec_proposals: true
    min_spec_accept_rate: 0.5
    min_spec_draft_occupancy_rate: 0.25
    max_overlap_accept_rate_drop_vs_nonoverlap: 0.1
```

Both `server_by_mode` files are complete unified fleet configurations. The
campaign scripts snapshot every referenced file, validate the static tuple, and
generate one `server`, one `observer`, and one `client` command for each case.
Command templates distinguish final `<RUN_DIR>` from disposable
`<RUNTIME_DIR>`; the server manifest lives only under the latter.

`axes.mode_target` selects which role the `nonoverlap`/`overlap` labels control.
It is optional and defaults to `verifier`, preserving the original contract. In
`verifier` mode, the verifier's `disable_overlap_schedule` follows the case
label and drafter scheduling is left unchanged. In `drafter` mode, the
drafter's `disable_overlap_schedule` follows the case label and both server
configs must keep the verifier overlapped (`disable_overlap_schedule: false`).

Keep concrete experiment inputs untracked under a local directory such as
`benchmark/decoupled_spec/configs/experiments/`. The materialized campaign
manifest and ledger, not the mutable workspace file, are authoritative after
materialization.
