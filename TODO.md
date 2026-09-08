# TODO — EvoJump backlog

Single authoritative backlog. Entries: one line + path(s). Completed items get
moved to the "Done (verified)" section with date. Agent-ergonomics pass
2026-08-31: log in `REVIEW_LOG_2026-08-31.md`.

## Minor

- [x] examples/README.md listed `basic_usage_fixed.py` (absent on disk) plus two thin_orchestrator examples (also absent) — FIXED 2026-08-31: run-snippet now points to `working_demo.py`; stale Architecture Examples section removed.
- [x] examples/README.md `drosophila_case_study.py` "(coming soon)" — STALE 2026-08-31: the phrase is already absent from the file (fixed in a prior pass); no edit needed.
- [x] tests/README.md `uv run pytest` guidance — STALE 2026-08-31: file already routes to `.venv/bin/python -m pytest` and warns against `uv run`; no edit needed.
- [x] tests/AGENTS.md file roster — STALE 2026-08-31: file already states the live roster (16 files verified 2026-08-31, lane modules named); no edit needed.
- [x] src/AGENTS.md and tests/AGENTS.md "Verify" snippets using `uv run` — STALE 2026-08-31: neither file contains `uv run` in its Verify section; src/AGENTS.md already uses `.venv/bin/python -m pytest ../tests/`; no edit needed.
- [x] README "Technical Specifications" table undated prose coverage — FIXED 2026-08-31: provenance note added pointing to `coverage.xml` (line-rate 0.6588, 2026-08-31) and `htmlcov/` as executable truth; re-measure deferred per Major item.
- [x] README "8 Core Modules" claim vs 7 modules in `src/evojump/` — FIXED 2026-08-31: changed to 7 with verification command.

## Medium

- [x] README test-files table missing `test_methods_lane_postaudit.py` (16 test files on disk; table listed 15) — FIXED 2026-08-31.
- [x] README "15 test files" count stale (16 on disk) — FIXED 2026-08-31.
- [x] AGENTS.md claims 95%+ coverage floor; pyproject enforces `--cov-fail-under=68` and last coverage.xml measured 65.9% — FIXED 2026-08-31 (AGENTS.md now states 68% floor + measurement command).
- [x] AGENTS.md still documents removed `run_all_tests.py` flags (`--all`, `--coverage`, `--lint`, `--docs`, `--benchmark`); the script is a thin forward-to-pytest wrapper — FIXED 2026-08-31 (AGENTS.md rewritten to match wrapper contract).
- [x] tests/README.md two-line stub gave wrong/stale run guidance — FIXED 2026-08-31.
- [x] No orientation ladder (status / next-actions / verification) in README — FIXED 2026-08-31 ("Status at a glance" section, verified claims only).
- [x] No root TODO/backlog file — FIXED 2026-08-31 (this file).

## Done (verified)

- [x] 2026-09-08 (v0.5.0 release pass): Full-suite gate completed on this
  checkout — `MPLBACKEND=Agg .venv/bin/coverage run --source=src/evojump -m
  pytest tests/ -q` + `.venv/bin/coverage report --fail-under=95`:
  **667 passed / 99% coverage** (floor 95%). README coverage prose reconciled
  to the measured values (previous historical note: 70.94% / 280 tests /
  68% floor).
- [x] 2026-09-08 (v0.5.0 release pass): Author identity resolved by owner
  decision — **Daniel Ari Friedman** (Active Inference Institute, ORCID
  0000-0001-6232-9096); applied in `paper/paper.md` YAML, `.zenodo.json`,
  `CITATION.cff`, and README citation blocks (verified in all three files).
- [x] 2026-08-31 (second doc pass): Minor items 1 fixed; items 2-5 found already-fixed/stale and closed with evidence; items 6-7 fixed.

- [x] 2026-08-31: agent-ergonomics pass — see `REVIEW_LOG_2026-08-31.md` for the cold-start audit and every change made.
