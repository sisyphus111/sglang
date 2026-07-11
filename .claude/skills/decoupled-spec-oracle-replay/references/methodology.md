# Fluid Oracle Replay Methodology

## Model

Let the reference trace provide token-progress chunks `w_i` and observable
states `x_i = (allow_partial, routed_batch_slot, context_bucket)`. Static runs
estimate the service rate of tier `s` in each state:

```text
rate(x, s) = sum(output_tokens) / sum(observed_iteration_time)
```

The oracle selects `argmax_s rate(x_i, s)` and replays the same work chunk:

```text
oracle_time_i = w_i / max_s rate(x_i, s)
T_oracle_reconstructed = sum(oracle_time_i)
```

Scale the reference benchmark E2E time by the reconstructed-time ratio so
startup and non-decode overhead remain represented:

```text
oracle_speedup = T_reference_reconstructed / T_oracle_reconstructed
T_oracle_e2e = T_reference_e2e / oracle_speedup
```

For desired capture `q`:

```text
T_target = T_static - q * (T_static - T_oracle)
```

## Scope And Limits

The replay captures measured tier-dependent acceptance, verifier latency,
partial-tail supply, and BS/context variation along a real long-response path.
It does not model strategy-induced request finish order, counterfactual draft
buffer supply, or states absent from the static matrix. It is an approximate
fluid oracle, not a mathematical upper bound. Request-level discrete-event
replay is required when finish-order feedback materially changes the state path.

## Decision-Grade Checks

- `candidate_coverage_token_share`: reference work with at least two tiers.
- `all_tier_coverage_token_share`: reference work with every static tier.
- `static_total_token_relative_range`: generated-token drift across candidates.
- `static_queued_point_count` and `static_max_queue_req`: queue evidence across
  the complete AP-matched static matrix.
- `state_coverage`, `incomplete_states`, and `fallback_states`: exact missing
  tier/state evidence.
- `dynamic_time_saving_capture`: primary online-policy quality metric.
- `speedup_gain_capture`: secondary multiplicative-speedup ratio.
