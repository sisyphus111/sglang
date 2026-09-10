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

Do not set `CUDA_VISIBLE_DEVICES` in `runtime.env`. Ray assigns one balanced
node-local GPU shard to each Engine actor; the actor pins its child processes
to exactly that allocation.

## Pair invariants

The verifier uses `speculative_algorithm=DECOUPLED_VERIFY`; the drafter is a
plain decode engine whose algorithm is unset. The drafter may use the legacy
non-overlap scheduler or the GPU-authoritative overlap path. The role templates
must agree on:

- `speculative_num_steps` (K)
- `speculative_eagle_topk` (currently F=1)
- `speculative_num_draft_tokens` (K+1 for F=1)

The current drafter additionally requires TP1 and `page_size=1`. Drafter
overlap requires the shared GPU backend, `disable_radix_cache=true`, mixed
chunked prefill disabled, and ReplaySSM disabled. Dense attention drafts restore
the committed prefix through KV positions; Mamba/GDN drafts additionally require
a routable recurrent-state checkpoint pool. Both roles require DP1 and PP1.

`SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND` selects native C++ threads/libzmq
(`1`) or Python threads/pyzmq (`0`). Both use the same native GPU backend and
wire codec, so mixed Python/C++ peers are supported. The Python transport
also requires the compiled GPU extension.

## Placement and ports

The launcher first reads available GPU capacity per Ray node and applies the
v0.5.14 joint placement objective: minimize verifier node count, then physical
nodes, then prefer verifier/drafter colocation. A verifier is split evenly:
TP16 on 8-GPU nodes becomes two actor bundles with 8 GPUs each. Each bundle
constructs a native SGLang Engine using the same `dist_init_addr` and exact
`nnodes/node_rank`; only node rank 0 launches HTTP. Drafters remain TP1 and use
one complete Engine actor.

After placement, each engine actor holds three real node-local TCP socket
leases for HTTP, decoupled transport, and NCCL initialization. The sockets are
allocated by first trying the node's numerically ordered `PORT{n}` environment
values and then falling back to an OS-assigned port. Invalid or duplicate
`PORT{n}` values fail fast. Transport and NCCL leases are released immediately
before engine startup; the HTTP listener is inherited by `role_server.py` and
handed directly to uvicorn, so slow weight loading cannot reopen a port-selection
race. There are no user-selected port ranges to coordinate across nodes.

## Sparse quota topology

The launcher ports the v0.5.14-dev equal-share quota graph. It partitions the
same integer interval by verifier and drafter rank and uses interval overlap as
the positive SWRR quota. Only intersecting verifier/drafter pairs become peers;
do not replace this with a dense full mesh.

Each resolved engine config receives
`decoupled_spec_peer_configs=[{rank, endpoint, quota}, ...]`. Peer ranks are in
the opposite role's rank space. The resulting `topology.quota_edges` and every
engine's effective peer list are saved in
`<RUNTIME_DIR>/server/manifest.json` for runtime discovery only.

The Ray job uploads the current checkout's `server` package and SGLang
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
- Can `sum(replicas * tp_size)` fit across the available nodes, including a
  separate drafter GPU?
- Do both model paths exist on the nodes where their actors may be placed?
- Are unrelated SGLang, Ray, profiler, or benchmark processes active?

Run `server.py --check` before connecting to Ray. If a resource is occupied,
stop and identify its owner; do not kill it merely because its command resembles
SGLang.
