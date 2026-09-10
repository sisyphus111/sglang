# Decoupled Verifier Profile Contract

The JSON has `kind=sglang_decoupled_verify_scheduler_cycle_profile`, schema and
profile ABI versions, `status`, a runtime fingerprint, request/grid metadata,
coverage, provenance, and one point per `(step, batch_size, context_len)`.

`cost_ms` is the 10%-trimmed mean of gaps between consecutive completed decode
rounds. The completion timestamp is taken only after normal batch-result
processing and the verifier-local `VerifyCommit` enqueue. It is Scheduler cycle
ITL, not target-forward CUDA time and not a host-corrected estimate.

The forward stream launches a mock GPU-tail selector at the exact production
snapshot call site. It reads an immutable GPU buffer with the production row
stride and writes the same compact snapshot/debug layout before normal TP
broadcast and VerifyInput construction. The verifier daemon consumes and drops
profile controls; it never injects tokens or mutates the production tail state.
A profile round is valid only when the full requested BS is decoding, active K
matches, selected draft length equals K for every row, acceptance is full, no
row finishes/retracts, and CUDA Graph replay is reported.

Contexts are anchored at the measurement midpoint. The input prompt is shorter
than the requested context by the prefill bonus and expected warmup/measurement
growth. The point records requested context, input context, and measured mean
context so the anchor remains auditable.

Production rejects partial profiles and any fingerprint mismatch. The
fingerprint covers the target model, dtype/topology, hardware, attention/CUDA
Graph settings, overlap mode, page size, ReplaySSM/Mamba settings, and Kmax.
