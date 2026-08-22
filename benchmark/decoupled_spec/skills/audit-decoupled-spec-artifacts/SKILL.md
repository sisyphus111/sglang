---
name: audit-decoupled-spec-artifacts
description: Verify completeness, cross-file consistency, process finality, plot provenance, and checksums for a decoupled-spec benchmark RUN_DIR. Use when deciding whether a run is complete, sealable, reproducible, or safe to share; it does not repair or invent missing results.
---

# Audit Decoupled-Spec Artifacts

Apply a deterministic acceptance gate to one run directory.

## Before Sealing

Read [references/acceptance-gates.md](references/acceptance-gates.md), then run:

```bash
python benchmark/decoupled_spec/skills/audit-decoupled-spec-artifacts/scripts/audit_run.py \
  --run-dir <RUN_DIR> \
  --phase pre-seal \
  --output <RUN_DIR>/audit/pre_seal.json
```

Seal only when this exits zero. The audit checks required files, role states,
live PIDs, batch cardinality, request IDs, formal-window validity,
observability coverage, and plot source hashes.

## After Sealing

Read [references/artifact-schema.md](references/artifact-schema.md) when a file
or relationship is unclear, then run:

```bash
python benchmark/decoupled_spec/common/artifacts.py seal --run-dir <RUN_DIR>

python benchmark/decoupled_spec/skills/audit-decoupled-spec-artifacts/scripts/audit_run.py \
  --run-dir <RUN_DIR> \
  --phase sealed
```

Sealed mode verifies `run_manifest.json`, every checksum entry, and the exact
set of files covered by `SHA256SUMS`. It is read-only unless `--output` points
outside `RUN_DIR`.

## Failure Policy

Do not fix an audit by deleting evidence, changing metrics, editing resolved
configs, or resealing over an unexplained mismatch. Report exact paths and
invariants. If the run is genuinely incomplete, preserve it unsealed and run a
new experiment after correcting the cause.

## Output Contract

Return PASS/FAIL, phase, checked run path, all errors, warnings, client
cardinality results, observability validation, plot provenance checks, and
sealed checksum coverage. A warning remains visible but does not replace a
failed invariant.
