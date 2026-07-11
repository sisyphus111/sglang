# Raw Experiment Artifact Contract

The copied `config.toml` is the user-facing experiment intent; the first
invocation freezes it as `config.lock.toml`. `manifest.json`
lists the expected case identity and raw paths; `status.jsonl` is an append-only
lifecycle ledger that records checkout commit, dirty state, Python executable,
and config hash for every accepted invocation. Dirty checkouts fail before the
run directory is mutated. Each case owns exactly one scheduler log and one output
directory.

The runner rejects config drift and a dirty source checkout. Profile provenance
includes the content hash and runtime-compatible fingerprint. `manifest.json` is
written before the first case so an interruption still leaves the expected case
ledger. `summary.json`
is the resume boundary because it is written only after the
official example completes benchmark aggregation. A log alone is not completion
evidence. `--force` reruns every selected case, removes stale summaries before
launch, and replaces logs, so preserve the old run directory when provenance matters.

Downstream consumers must not rewrite raw files. The trajectory skill writes to
`<run-dir>/analysis/`; oracle replay writes below that analysis directory.
