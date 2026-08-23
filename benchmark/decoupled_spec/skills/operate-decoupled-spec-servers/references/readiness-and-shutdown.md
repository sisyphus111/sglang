# Readiness and Shutdown

Run commands from the SGLang repository root with `PYTHONPATH=python` so imports
come from the active checkout.

## Validation

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/server.py \
  --config <server-fleet.yaml> --run-dir <RUN_DIR> --check
```

## Unified Ray launch

Connect to an existing Ray cluster and keep this one driver command in a
long-running session:

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/server.py \
  --config <server-fleet.yaml> --run-dir <RUN_DIR>
```

The launcher jointly reserves all replicas with one placement group, allocates
node-local HTTP/transport/NCCL ports, starts both roles, and writes
`server/manifest.json`. Do not send formal traffic until that manifest is
`ready`. The launcher currently requires each TP replica to fit on one Ray node;
replicas may be distributed across many nodes.

After the collector starts, require a successful baseline sample with
`num_waiting_reqs == 0` for both roles. Do not start measured traffic on top of
leftover queued work.

## Readiness gate

```bash
python benchmark/decoupled_spec/skills/operate-decoupled-spec-servers/scripts/wait_for_roles.py \
  --run-dir <RUN_DIR> \
  --timeout-s 600
```

The helper reads every engine from the ready manifest and performs the
role-appropriate HTTP check: verifier uses `/health`, while drafter uses
read-only `/model_info`. A failed/stopped manifest fails immediately.

## Shutdown

After the client and collector finish:

1. Send graceful termination to the unified launcher PID/session.
2. Wait for it to stop every engine, collect remote logs/status/configs, kill
   owned actors, remove its placement group, and disconnect from Ray.
3. Inspect `server/manifest.json`, per-engine status, and log files.
4. Confirm no owned child process remains before pre-seal audit.

Do not use a repository-wide `pkill`, and do not stop the externally owned Ray
cluster.
