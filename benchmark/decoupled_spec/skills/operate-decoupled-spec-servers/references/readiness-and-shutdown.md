# Readiness and Shutdown

Run commands from the SGLang repository root with `PYTHONPATH=python` so imports
come from the active checkout.

## Validation

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/config.py validate \
  --verifier-config <verifier.yaml> \
  --drafter-config <drafter.yaml> \
  <role-prefixed effective overrides>

PYTHONPATH=python python benchmark/decoupled_spec/server-side/verifier_server.py \
  --config <verifier.yaml> --run-dir <RUN_DIR> <verifier overrides> --check

PYTHONPATH=python python benchmark/decoupled_spec/server-side/drafter_server.py \
  --config <drafter.yaml> --run-dir <RUN_DIR> <drafter overrides> --check
```

## Independent launch

Create `RUN_DIR/logs/` and start each command in its own long-running session.
Direct stdout/stderr to `logs/verifier.log` and `logs/drafter.log` while keeping
the sessions independently controllable.

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/verifier_server.py \
  --config <verifier.yaml> --run-dir <RUN_DIR> <verifier overrides>
```

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/drafter_server.py \
  --config <drafter.yaml> --run-dir <RUN_DIR> <drafter overrides>
```

There is no combined server launcher and no required launch order. Do not send
formal traffic until both roles are ready.

## Readiness gate

```bash
python benchmark/decoupled_spec/skills/operate-decoupled-spec-servers/scripts/wait_for_roles.py \
  --run-dir <RUN_DIR> \
  --timeout-s 600
```

The helper derives each HTTP URL from the saved status or resolved config and
requires both the file-state and a role-appropriate HTTP check: verifier uses
`/health`, while drafter uses read-only `/model_info` because the decoupled
drafter deliberately rejects generation-based health requests. A `failed` or
premature `exited` state fails immediately.

## Shutdown

After the client and collector finish:

1. Send graceful termination to the exact verifier and drafter sessions/PIDs.
2. Wait for both processes to exit and inspect their final logs/status files.
3. Escalate only the process that did not exit, and record the escalation.
4. Confirm no owned child process remains before pre-seal audit.

Do not use a repository-wide or command-pattern-wide `pkill`.
