---
name: analyze-decoupled-spec-results
description: Summarize and plot one saved SGLang decoupled-spec benchmark run directory with source provenance. Use when the user asks what an existing run shows or wants its derived figures and report; it does not launch servers, rerun traffic, or compare different runs.
---

# Analyze Decoupled-Spec Results

Turn existing run artifacts into a concise, source-traceable result without
changing the measured workload.

## Single Run

1. Read the
   [fixed client artifact contract](../send-decoupled-spec-workload/references/client-artifact-contract.md).
   Confirm the run contains exactly
   `client/{requests.csv,batch.json,content.json}` and no legacy client
   result file.
2. Read [references/metrics.md](references/metrics.md) before interpreting the
   values.
3. Read [references/plot-contract.md](references/plot-contract.md), then run the
   latency, speculative, observability, and report scripts in that order.
4. Inspect request-level points and service time series as well as aggregates. Report missing or null
   metrics explicitly.
5. Keep the final PNG figures and Markdown report under `plots/`; do not create
   plot manifests or duplicate observer summaries.

## Evidence Boundary

Client latency is HTTP streaming latency. Observability samples are
second-scale service state. Neither source alone decomposes GPU compute, CUDA
IPC, callback, event wait, or network time. Use an Nsys/NVTX or torch-profiler
artifact for those questions and keep it linked to the same run tuple.

## Output Contract

Return the exact input run directory and tuple, the main metrics, plot/report
paths, strongest supported conclusion, and any measurement limitation. The
single-run plot contract writes clear, high-resolution PNG figures. Every
non-negative user-facing metric plot must use an
ordinary linear y-axis starting at zero; signed effect plots must include a
visible zero baseline. Exclude clearly abnormal presentation outliers with a
deterministic recorded rule while preserving the raw artifact and recording
every excluded window/time/value in the report. Never rerun an equivalent
costly benchmark only to recreate a plot from valid saved inputs.
