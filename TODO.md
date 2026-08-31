# TODO — EvoJump backlog

Single authoritative backlog. Entries: one line + path(s). Completed items get
moved to the "Done (verified)" section with date. Agent-ergonomics pass
2026-08-31: log in `REVIEW_LOG_2026-08-31.md`.

## Minor

- [ ] examples/README.md lists `basic_usage_fixed.py` which does not exist on disk (verified 2026-08-31: `ls examples/basic_usage_fixed.py` → No such file) — remove or restore the file (`examples/README.md`).
- [ ] examples/README.md marks `drosophila_case_study.py` "(coming soon)" but the file exists and has a passing test (`tests/test_drosophila_case_study.py`) — drop the "(coming soon)" (`examples/README.md`).
- [ ] tests/README.md says `uv run pytest tests/` — known to stall under heavy load on this machine (README/CHANGELOG v0.2.0 note); point to `.venv/bin/python -m pytest tests/ -q --no-cov` (`tests/README.md`).
- [ ] tests/AGENTS.md file roster (verified 2026-08-29, 9 files) predates the six newer test modules; refresh to the live set via `ls tests/test_*.py` (`tests/AGENTS.md`).
- [ ] src/AGENTS.md and tests/AGENTS.md "Verify" snippets use `uv run` despite the documented stall — align with root README commands (`src/AGENTS.md`, `tests/AGENTS.md`).
- [ ] README "Technical Specifications" per-module coverage/test-count table (84%/83%/78%... "24 tests" etc.) is prose truth with no stated measurement date; either re-measure and date it, or link to `coverage.xml` + `htmlcov/` as the executable truth (`README.md`).
- [ ] README "8 Core Modules" claim vs 7 modules in `src/evojump/` (analytics_engine, cli, datacore, evolution_sampler, jumprope, laserplane, trajectory_visualizer — verified 2026-08-31); recount or state what the 8th is (`README.md`).

## Medium

- [x] README test-files table missing `test_methods_lane_postaudit.py` (16 test files on disk; table listed 15) — FIXED 2026-08-31.
- [x] README "15 test files" count stale (16 on disk) — FIXED 2026-08-31.
- [x] AGENTS.md claims 95%+ coverage floor; pyproject enforces `--cov-fail-under=68` and last coverage.xml measured 65.9% — FIXED 2026-08-31 (AGENTS.md now states 68% floor + measurement command).
- [x] AGENTS.md still documents removed `run_all_tests.py` flags (`--all`, `--coverage`, `--lint`, `--docs`, `--benchmark`); the script is a thin forward-to-pytest wrapper — FIXED 2026-08-31 (AGENTS.md rewritten to match wrapper contract).
- [x] tests/README.md two-line stub gave wrong/stale run guidance — FIXED 2026-08-31.
- [x] No orientation ladder (status / next-actions / verification) in README — FIXED 2026-08-31 ("Status at a glance" section, verified claims only).
- [x] No root TODO/backlog file — FIXED 2026-08-31 (this file).

## Major

- [ ] Full-suite pytest re-run on this external-drive checkout is ~30+ min under fleet load (see `paper/paper_verification_report.md` measurements of 2026-08-30) and could not be completed in the 2026-08-31 doc pass; the coverage.xml (65.9%) is also below the 68% floor. Defer to an idle-machine session: run `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q --no-cov` then a coverage run, and reconcile README's coverage table with measured output. NOT done here — slow-drive constraint, no gate claimed.
- [ ] Author-identity discrepancy between `paper/README.md` (Daniel Ari Friedman) and `paper/paper.md` ("EvoJump Development Team") already flagged in `docs/manuscript/MANUSCRIPT_STATUS.md` — needs owner decision, not a doc-pass edit.

## Done (verified)

- [x] 2026-08-31: agent-ergonomics pass — see `REVIEW_LOG_2026-08-31.md` for the cold-start audit and every change made.
