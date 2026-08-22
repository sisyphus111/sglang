---
name: analyze-decoupled-spec-results
description: Summarize and plot one saved SGLang decoupled-spec benchmark run directory with source provenance. Use when the user asks what an existing run shows or wants its derived figures and report; it does not launch servers, rerun traffic, or compare different runs.
---

# Analyze Decoupled-Spec Results

Turn existing run artifacts into a concise, source-traceable result without
changing the measured workload.

## Single Run

1. Confirm the run contains `client/summary.json` and
   `client/request_metrics.csv`.
2. Read [references/metrics.md](references/metrics.md) before interpreting the
   values.
3. Read [references/plot-contract.md](references/plot-contract.md), then run the
   latency, speculative, observability, and report scripts in that order.
4. Inspect request-level points and service time series as well as aggregates. Report missing or null
   metrics explicitly.
5. Retain every script-specific manifest. Each manifest hashes exactly the
   saved inputs used for its own outputs.

## Evidence Boundary

Client latency is HTTP streaming latency. Observability samples are
second-scale service state. Neither source alone decomposes GPU compute, CUDA
IPC, callback, event wait, or network time. Use an Nsys/NVTX or torch-profiler
artifact for those questions and keep it linked to the same run tuple.

## Output Contract

Return the exact input run directory and tuple, the main metrics, plot/report
paths, manifest paths, strongest supported conclusion, and any measurement
limitation. Never rerun an equivalent
costly benchmark only to recreate a plot from valid saved inputs.
