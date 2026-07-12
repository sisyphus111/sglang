---
name: decoupled-spec-development
description: Preserve SGLang package, runtime, launcher, and experiment ownership boundaries when adding, moving, refactoring, or reviewing decoupled-speculation code under python/sglang/srt/speculative or examples/runtime/engine/decoupled_speculation.
---

# Decoupled Spec Development

Keep production and reusable runtime logic inside the installable `sglang`
package. Keep `examples` limited to user-facing entrypoints, argument handling,
and workload-specific orchestration.

## Package Boundary

- Place shared runtime types, actor helpers, topology construction, transport,
  profiling primitives, metrics, and reusable prompt/loading helpers under
  `python/sglang/srt/speculative/`.
- Place a cohesive launcher-support package under
  `python/sglang/srt/speculative/decoupled_speculation/`, not under `examples`.
- Import those helpers through `sglang.srt.speculative...`; do not rely on the
  example directory being on `sys.path` or insert it into `sys.path`.
- Keep `examples/runtime/engine/decoupled_speculation/` for executable wrappers
  such as `single-node.py`, `multi-node.py`, and narrowly scoped utilities.

This boundary matters because the SGLang distribution is built from `python/`.
Python files moved into `examples/` are no longer part of the `sglang` package or
installed wheel, so downstream users and installed launchers cannot import them.
An `__init__.py` under `examples` does not restore membership in `sglang`.

## Change Workflow

1. Classify each changed module as reusable runtime logic or example-only
   orchestration before moving it.
2. Search both package and example call sites and keep imports package-qualified.
3. Reject a move from `python/sglang` to `examples` when installed SGLang code,
   downstream callers, tests, or more than one launcher may reuse the module.
4. After a move, verify imports from outside the repository root or from a built
   wheel when packaging behavior is material.
5. Run focused tests plus `py_compile` for changed Python entrypoints and
   `git diff --check`.
