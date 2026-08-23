---
name: operate-decoupled-spec-servers
description: Validate, launch, inspect, and stop a Ray-orchestrated fleet of decoupled-spec verifier and drafter HTTP servers from one unified config. Use for server-side setup or lifecycle work; it does not submit benchmark traffic or start the observability collector.
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

## Launch and Readiness

Read [references/readiness-and-shutdown.md](references/readiness-and-shutdown.md).

- Start exactly one long-running `server-side/server.py` process with the
  unified YAML and shared `RUN_DIR`.
- Treat `server/manifest.json` as the machine-readable authority. Continue only
  after `state=ready`, every engine has passed its role-appropriate HTTP probe,
  and the launcher has printed the human-readable endpoint table plus the
  `DSPEC_SERVER_MANIFEST` JSON line.
- Distinguish each engine's `http_url` from its internal
  `transport_endpoint`; client and collector consume only HTTP URLs.
- Use `scripts/wait_for_roles.py` for the deterministic readiness gate.

## Shutdown

Terminate only the unified launcher process created for this run. It owns child
HTTP processes, Ray actors, placement group, and port leases; wait for its
idempotent cleanup before continuing. Do not call `ray stop` or shut down an
externally owned cluster.

## Output Contract

Return the manifest path, quota edges, every engine's rank/node/GPU/HTTP URL,
readiness result, launcher PID/session, per-engine log paths, and final cleanup
state.
Report the first failed invariant rather than continuing with a partial pair.
