# Decoupled-Spec Benchmark Skill Rules

- The canonical, editable source of every decoupled-spec benchmark skill must
  live at `benchmark/decoupled_spec/skills/<skill-name>/`.
- `.claude/skills/<skill-name>` must be a relative symlink to that
  benchmark-local package. Do not maintain a second copy under `.claude/skills`.
- `.codex/skills` exposes `.claude/skills`; do not add another Codex-specific
  copy of a benchmark skill.
- Put deterministic repeated operations in the skill's `scripts/` directory and
  conditional contracts in `references/`. Keep `SKILL.md` focused on routing,
  invariants, and the operational workflow.
- After adding or changing a skill, run the Skill Creator `quick_validate.py`
  check and verify that the `.claude/skills` symlink resolves inside the current
  repository.
