---
name: operate-decoupled-spec-servers
description: Validate, launch, inspect, and stop the independent verifier and drafter HTTP servers for an SGLang decoupled-spec benchmark. Use for server-side setup or lifecycle work; it does not submit benchmark traffic or start the observability collector.
---

# Operate Decoupled-Spec Servers

Bring up the verifier and drafter as two independently observable processes
using one effective, pair-validated topology.

## Before GPU Work

Read [references/server-config-and-topology.md](references/server-config-and-topology.md),
then inspect the active checkout, visible GPUs, existing SGLang/Ray processes,
model paths, and requested ports. Do not terminate or reuse resources owned by
another run.

Run both role-local `--check` commands and the pair validator. When the launch
uses CLI overrides, apply the role-prefixed equivalents to the pair validator;
validating only the YAML baseline is insufficient.

## Launch and Readiness

Read [references/readiness-and-shutdown.md](references/readiness-and-shutdown.md).

- Start `verifier_server.py` and `drafter_server.py` in separate long-running
  sessions with the same `RUN_DIR`.
- Preserve each session/PID separately and capture each role's stdout/stderr.
- Do not continue on log text alone. Require both `status.json` state
  `http_ready` and the role-appropriate HTTP readiness probe.
- Use `scripts/wait_for_roles.py` for the deterministic readiness gate.

## Shutdown

Stop only the two server sessions created for this run. Prefer graceful
termination and wait for exit before escalating. Record any escalation. The
server status files and captured logs remain part of the run evidence.

## Output Contract

Return the effective verifier/drafter topology, assigned GPUs, URLs, role PIDs
or session identifiers, log paths, readiness result, and final exit states.
Report the first failed invariant rather than continuing with a partial pair.
