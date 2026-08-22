# Server Config and Topology

## Role config contract

Each server YAML has four top-level keys:

| Key | Meaning |
| --- | --- |
| `schema_version` | Must be `1` |
| `role` | Exactly `verifier` or `drafter` for the selected launcher |
| `runtime` | Process-local GPU visibility and environment |
| `server_args` | Fields passed to the current checkout's `ServerArgs` |

`runtime.cuda_visible_devices` is a non-empty list when present. The verifier
and drafter lists must not overlap. `runtime.env` is applied inside the role
process; do not modify the parent shell globally to approximate it.

## Pair invariants

The verifier uses `speculative_algorithm=DECOUPLED_VERIFY`; the drafter is a
plain decode engine whose algorithm is unset and whose overlap schedule is
disabled. The effective configs must agree on:

- `speculative_num_steps` (K)
- `speculative_eagle_topk` (phase one requires F=1)
- `speculative_num_draft_tokens` (for F=1 this is K+1)
- `SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND` after boolean normalization

Each role's `decoupled_spec_connect_endpoints` must include the other role's
bind endpoint. HTTP ports and decoupled transport endpoints are separate
namespaces and must both be conflict-free.

## Named overrides

Role launchers accept:

- `--model-path`
- `--tp-size`
- `--cuda-visible-devices`
- `--host`
- `--port`

The pair validator accepts the same axes with `--verifier-` and `--drafter-`
prefixes. Pass the exact launch overrides to validation. K/F/verify-window,
data-plane backend, and bind/connect topology remain paired YAML settings
unless their public CLI contract is deliberately extended.

## Preflight evidence

Before model loading, collect enough read-only evidence to answer:

- Which checkout and commit will run?
- Which physical GPUs are free, and does each TP size fit its visible set?
- Are requested HTTP and transport ports already bound?
- Are unrelated SGLang, Ray, profiler, or benchmark processes active?
- Do both model paths exist and refer to the intended target/draft models?

If a resource is occupied, stop and identify the owner. Do not kill it merely
because its command resembles SGLang.
