# Acceptance Gates

## Pre-seal PASS

A successful run must satisfy all of the following:

- provenance, effective role/client/collector configs, status files, raw client
  artifacts, observability artifacts, and single-run plot artifacts exist
- client and observability statuses are `completed`
- verifier and drafter are no longer live; neither has a recorded failed state
- effective client batch size equals formal-window batch size, summary
  cardinality, sampled requests, final responses, raw batch responses, and CSV
  rows
- request IDs are unique and agree across sampled requests, responses, and CSV
- the formal window is completed, ordered, and has positive elapsed time
- verifier and drafter observability samples cover baseline, formal, and
  trailing periods with internally consistent summary counts
- every source hash and output path in the latency, speculative,
  observability, and report manifests verifies
- the explicitly generated observability overview plots exist
- when collected load samples contain fixed decode windows, both
  `decode_metrics.svg` and `decode_metrics.png` exist and are declared by the
  observability plot manifest

Captured verifier and drafter logs are expected from the Agent workflow. Their
absence is reported as a warning because older valid run directories may not
contain them.

## Sealed PASS

Sealed mode includes every pre-seal gate and additionally requires:

- valid `run_manifest.json` with a seal timestamp
- valid `SHA256SUMS` syntax
- every listed file exists and matches its SHA-256
- every current regular file except `SHA256SUMS` is listed exactly once
- no unlisted file has been added after sealing

## Process check

The audit checks PIDs saved in role status files. A live owned verifier or
drafter is a hard failure before sealing. A dead process whose last status is
still `http_ready` is accepted with a warning because external termination may
prevent the final status update; its log should be inspected.

## Result boundary

Audit success means the artifact relationships are internally consistent. It
does not establish model correctness, benchmark representativeness, or causal
performance attribution.
