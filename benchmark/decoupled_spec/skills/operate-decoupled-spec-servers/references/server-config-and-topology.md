# Server Config and Topology

## Unified config contract

One YAML describes the complete Ray fleet:

| Key | Meaning |
| --- | --- |
| `schema_version` | Must be `1` |
| `ray` | Existing-cluster address, namespace, and lifecycle timeouts |
| `verifier` | Verifier replica count, process environment, and `ServerArgs` template |
| `drafter` | Drafter replica count, process environment, and `ServerArgs` template |

Each role has `replicas`, `runtime.env`, and `server_args`. The launcher owns
all placement-dependent values. Do not put any of these fields in a role's
`server_args`:

- `host`, `port`, `nccl_port`, `dist_init_addr`, `nnodes`, or `node_rank`
- `base_gpu_id`, `gpu_id_step`, or `use_ray`
- `decoupled_spec_role`, `decoupled_spec_rank`, or
  `decoupled_spec_bind_endpoint`
- `decoupled_spec_connect_endpoints` or `decoupled_spec_peer_configs`

Do not set `CUDA_VISIBLE_DEVICES` in `runtime.env`. Ray assigns exactly
`tp_size` GPUs to each engine actor.

## Pair invariants

The verifier uses `speculative_algorithm=DECOUPLED_VERIFY`; the drafter is a
plain decode engine whose algorithm is unset and whose overlap schedule is
disabled. The role templates must agree on:

- `speculative_num_steps` (K)
- `speculative_eagle_topk` (currently F=1)
- `speculative_num_draft_tokens` (K+1 for F=1)
- `SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND` after boolean normalization

The current drafter additionally requires TP1 and `page_size=1`. Both roles
require DP1 and PP1.

## Placement and ports

The launcher creates one joint Ray placement group with one bundle per engine.
Each bundle requests one CPU and `tp_size` GPUs, so every TP replica must fit on
one node; different replicas may run on different nodes.

After placement, each engine actor holds three real node-local TCP socket
leases for HTTP, decoupled transport, and NCCL initialization. The sockets are
released only immediately before that engine's HTTP subprocess starts. There
are no user-selected port ranges to coordinate across nodes.

## Sparse quota topology

The launcher ports the v0.5.14-dev equal-share quota graph. It partitions the
same integer interval by verifier and drafter rank and uses interval overlap as
the positive SWRR quota. Only intersecting verifier/drafter pairs become peers;
do not replace this with a dense full mesh.

Each resolved engine config receives
`decoupled_spec_peer_configs=[{rank, endpoint, quota}, ...]`. Peer ranks are in
the opposite role's rank space. The resulting `topology.quota_edges` and every
engine's effective peer list are saved in `server/manifest.json`.

The Ray job uploads the current checkout's `server-side` package and SGLang
Python package with `runtime_env.py_modules`. Remote actors and their HTTP
children therefore execute that uploaded source rather than a stale node-local
installation. Each manifest engine records the resolved `sglang_path`; model
and dataset files themselves must still be reachable from the selected node.

## CLI overrides

The unified launcher accepts only control-plane overrides:

- `--ray-address`
- `--ray-namespace`

Model, TP, replica count, speculative settings, and process environment remain
in the unified YAML so that one saved fleet config fully describes the intended
deployment.

## Preflight evidence

Before model loading, collect enough read-only evidence to answer:

- Which checkout and commit will run on every Ray node?
- Does the existing Ray cluster expose enough jointly schedulable GPUs?
- Can each role's `tp_size` fit on one node?
- Do both model paths exist on the nodes where their actors may be placed?
- Are unrelated SGLang, Ray, profiler, or benchmark processes active?

Run `server.py --check` before connecting to Ray. If a resource is occupied,
stop and identify its owner; do not kill it merely because its command resembles
SGLang.
