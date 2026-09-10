# Readiness and Shutdown

Run commands from the SGLang repository root with `PYTHONPATH=python` so imports
come from the active checkout.

## Validation

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server/server.py \
  --config <server-fleet.yaml> --run-dir <RUN_DIR> --check
```

## Unified Ray launch

Connect to an existing Ray cluster and keep this one driver command in a
long-running session:

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server/server.py \
  --config <server-fleet.yaml> --run-dir <RUN_DIR> --runtime-dir <RUNTIME_DIR>
```

The launcher jointly reserves all replicas with one placement group, allocates
node-local HTTP/transport/NCCL ports, starts both roles, and writes
`<RUNTIME_DIR>/server/manifest.json`. Do not send formal traffic until that manifest is
`ready`. A verifier's `node_actors` and `rank_placements` describe all native
Engine shards; do not infer TP locality from the HTTP leader alone.

After the observer starts, require a successful baseline sample with
`num_waiting_reqs == 0` for both roles. Do not start measured traffic on top of
leftover queued work.

## Readiness gate

```bash
python benchmark/decoupled_spec/skills/operate-decoupled-spec-servers/scripts/wait_for_roles.py \
  --runtime-dir <RUNTIME_DIR> \
  --timeout-s 600
```

The helper reads every engine from the ready manifest and performs the
role-appropriate HTTP check: verifier uses `/health`, while drafter uses
read-only `/model_info`. A failed/stopped manifest fails immediately.

## Shutdown

After the client and observer finish, stop the deployment through its owning
environment and confirm no owned process remains. Node-local server logs are
optional diagnostics, not benchmark artifacts.

Do not use a repository-wide `pkill`, and do not stop the externally owned Ray
cluster.

### Submitted tasks

For a submitted task, stop the task after traffic and collection finish. The
task environment owns the Ray cluster and may terminate its launcher before
it writes a terminal manifest or copies node-local logs. Treat submitted-task
finality as the resource-release evidence. The manifest and logs remain temporary;
Client and Observer data under `RUN_DIR` are the benchmark evidence.
