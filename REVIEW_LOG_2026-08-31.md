# Review log — 2026-08-31 agent-ergonomics pass (fleet lane: evojump)

Cold-start audit, entering with only README.md + AGENTS.md as a fresh agent:

- (a) Determine current project status — PARTIAL FAIL before fixes. README's
  "Project Status & Achievements" made unverifiable marketing-grade claims
  (per-module coverage %, test counts, "8 Core Modules" vs 7 modules in
  `src/evojump/`), and there was no dated, command-backed status surface.
- (b) Find what to do next — FAIL before fixes. No backlog/TODO file existed
  anywhere in the repo; no pointer to next actions.
- (c) Find how to run primary verification — PARTIAL FAIL. Root README had
  correct `.venv/bin/python -m pytest` guidance, but `AGENTS.md` (the agent
  entry doc) still documented removed `run_all_tests.py` flags and a 95%
  coverage floor contradicting pyproject's `--cov-fail-under=68`;
  `tests/README.md` pointed to `uv run pytest` (documented stall risk).

Sweep findings and dispositions (all verified 2026-08-31):

1. AGENTS.md: 95% coverage claims; removed `run_all_tests.py` flags
   (`--all`, `--coverage`, `--benchmark`, `--lint`, `--docs`) — script is a
   thin forward-to-pytest wrapper (15 lines, verified). FIXED: 68% floor with
   measurement command; wrapper contract documented.
2. AGENTS.md stale pyproject pytest config block (claimed 95% floor) —
   superseded section replaced with pointer to pyproject (canonical home).
3. README: "15 test files" vs 16 on disk; table missing
   `test_methods_lane_postaudit.py`. FIXED both.
4. README: no orientation ladder. FIXED: "Status at a glance" section added
   with verified claims + verification commands.
5. No TODO.md. CREATED (single authoritative backlog; Medium items fixed this
   pass, Minor/Major deferred with reasons).
6. tests/README.md: two-line stub with stale guidance. REWRITTEN.
7. tests/AGENTS.md: 2026-08-29 roster missing 6 newer modules; `uv run`
   snippet. FIXED.
8. src/AGENTS.md: `uv run` verify snippet. FIXED.
9. examples/README.md: listed nonexistent `basic_usage_fixed.py` (verified
   absent); "(coming soon)" on an existing, tested case study. FIXED.
10. Relative-link check across README.md, AGENTS.md, docs/README.md,
    docs/AGENTS.md, tests/README.md, tests/AGENTS.md, paper/README.md: 0
    broken (script-checked 2026-08-31).
11. `import evojump` verified OK under the venv (cold I/O ~45 min on this
    external-drive checkout; run in background). Full pytest suite NOT run —
    ~30+ min under fleet load per paper/paper_verification_report.md;
    recorded as deferred Major in TODO.md. No gate pass claimed.
12. coverage.xml (2026-08-31 00:20) measures 65.9% vs the 68% floor —
    recorded in README status section and TODO.md Major; not a doc-pass fix.
