# Acceptance Gates

## Pre-seal PASS

A successful run must satisfy all of the following:

- provenance, unified fleet/per-engine/client/collector configs, status files,
  per-engine logs, raw client artifacts, observability artifacts, and
  single-run plot artifacts exist
- client and observability statuses are `completed`
- the unified launcher is no longer live, its final status is `exited`, the
  server manifest is `stopped`, and every engine status is `exited`
- effective client batch size equals formal-window batch size, summary
  cardinality, sampled requests, final responses, raw batch responses, and CSV
  rows
- request IDs are unique and agree across sampled requests, responses, and CSV
- the formal window is completed, ordered, and has positive elapsed time
- every manifest engine's observability samples cover baseline, formal, and
  trailing periods with internally consistent per-target and per-role counts
- every engine reports zero waiting requests in every successful formal-window
  load sample, and no engine log contains a positive
  `#queue-req`; queueing means the requested batch was not processed fully in
  parallel and is a hard invalid-run condition
- every source hash and output path in the latency, speculative,
  observability, and report manifests verifies
- the explicitly generated observability overview plots exist
- the observability plots visibly include verifier and drafter running batch
  size over time from `num_running_reqs`, on a linear y-axis starting at zero
- when collected load samples contain fixed decode windows, both
  `decode_metrics.svg` and `decode_metrics.png` exist and are declared by the
  observability plot manifest; their user-facing labels use iteration latency,
  and the iteration latency axis is linear and starts at zero

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

The audit checks local PIDs saved in role status files. A live unified launcher
is a hard failure before sealing. Ray-node child PIDs are not interpreted from
the driver host; their copied status must instead be `exited`. Legacy singular
server runs retain the old local-PID check and may accept stale `http_ready`
with a warning when the process is already dead.

## Result boundary

Audit success means the artifact relationships are internally consistent. It
does not establish model correctness, benchmark representativeness, or causal
performance attribution.
