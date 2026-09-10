---
name: operate-decoupled-spec-servers
description: Validate, launch, inspect, and stop a Ray-orchestrated fleet of decoupled-spec verifier and drafter HTTP servers from one unified config. Use for server setup or lifecycle work; it does not submit benchmark traffic or start the Observer.
---

# Operate Decoupled-Spec Servers

Bring up all verifier and drafter replicas through one Ray control plane while
keeping every engine independently observable over HTTP.

## Before GPU Work

Read [references/server-config-and-topology.md](references/server-config-and-topology.md),
then inspect the active checkout, visible GPUs, existing SGLang/Ray processes,
model paths, and Ray cluster health. Do not terminate or reuse resources owned by
another run.

Run unified `server.py --check` before connecting to Ray. The config must not
contain launcher-owned ranks, ports, endpoints, or CUDA device IDs. Confirm the
Ray cluster has enough jointly schedulable GPU resources for every replica.
If adaptive verifier mode is enabled, require a complete local profile and a
valid adaptive JSON before Ray placement. The launcher stages both byte-for-byte
to every verifier actor; verify their source paths, SHA256 values, and actor-local
  resolved paths in the temporary server manifest.

## Launch and Readiness

Read [references/readiness-and-shutdown.md](references/readiness-and-shutdown.md).

- Start exactly one long-running `server/server.py` process with the unified
  YAML, final `RUN_DIR`, and separate temporary `RUNTIME_DIR`.
- Treat `<RUNTIME_DIR>/server/manifest.json` as the machine-readable authority. Continue only
  after `state=ready`, every engine has passed its role-appropriate HTTP probe,
  and the launcher has printed the human-readable endpoint table plus the
  `DSPEC_SERVER_MANIFEST` JSON line.
- Distinguish each engine's `http_url` from its internal
  `transport_endpoint`; client and observer consume only HTTP URLs.
- Use `scripts/wait_for_roles.py` for the deterministic readiness gate.

## Shutdown

Stop the deployment through its owning environment and confirm that no owned
process or pod remains. For a local launcher, terminate only that launcher. For
a submitted task, stop the task itself; do not signal its Ray subprocess or call
`ray stop` inside the managed cluster.

## Output Contract

Return the temporary manifest path, quota edges, every engine's rank/node/GPU/HTTP URL,
every node actor and TP-rank placement, readiness result, launcher PID/session,
and final submitted-task process state. Per-node logs are temporary diagnostics and
must not be copied into `RUN_DIR`.
Report the first failed invariant rather than continuing with a partial pair.

## Submitted Task Remote Loop

For remote validation, the submitted task owns only the server fleet.
Run Client, Observer, validation, analysis, and campaign coordination from the
local checkout against the ready remote HTTP endpoints. Do not embed a complete
benchmark or campaign driver in the task entrypoint. Keep the entrypoint direct:
validate the unified config and launch `server/server.py`, with only the minimal
shell setup needed by the reference environment.

Keep this order strict:

1. Modify and locally validate the code.
2. Commit and push an exact branch commit.
3. Query owned submitted tasks and require zero `STARTED`/`PENDING`/`RUNNING` tasks
   for this campaign. Never exceed the authorized submitted-task concurrency.
4. Submit the task pinned to that commit. A TP16 verifier plus TP1
   drafter on 8-GPU nodes requires one head and two workers.
5. Wait for `DSPEC_SERVER_MANIFEST` in the head log, reconstruct a local
   manifest containing the advertised remote HTTP endpoints, then run local
   `runner.py` so the local Observer establishes a zero-waiting baseline before
   the local Client submits traffic.
6. After traffic and collection finish, stop the submitted task to release the Ray
   head and workers. Do not signal task-owned Ray subprocesses separately.
7. Require every task process to stop and zero active submitted tasks before
   changing code or submitting another.
