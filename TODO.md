# TODO — EvoJump backlog

Single authoritative backlog. Entries: one line + path(s). Completed items get
moved to the "Done (verified)" section with date. Agent-ergonomics pass
2026-08-31 (its log file was not committed to the repo).

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
  **684 passed / 99% coverage** (floor 95%). README coverage prose reconciled
  to the measured values (previous historical note: 70.94% / 280 tests /
  68% floor).
- [x] 2026-09-08 (v0.5.0 release pass): Author identity resolved by owner
  decision — **Daniel Ari Friedman** (Active Inference Institute, ORCID
  0000-0001-6232-9096); applied in `paper/paper.md` YAML, `.zenodo.json`,
  `CITATION.cff`, and README citation blocks (verified in all three files).
- [x] 2026-08-31 (second doc pass): Minor items 1 fixed; items 2-5 found already-fixed/stale and closed with evidence; items 6-7 fixed.

- [x] 2026-08-31: agent-ergonomics pass — cold-start audit (its log file was
  not committed to the repo).

- [x] 2026-10-25 (maintenance pass): `run_tests.py --coverage` NameError
  (undefined `COVERAGE_RUN`) FIXED; py3.9 PEP 604 breakage in `run_tests.py`
  / `run_all_examples.py` FIXED; stale `uv.lock` project version (v0.3.0)
  FIXED via `uv lock`; CI actions bumped (`actions/checkout` v4→v7,
  `actions/setup-python` v5→v7); dead links (Read the Docs, Discussions,
  support@evojump.org) and unpublished-PyPI/`requirements.txt` install
  instructions removed across README/docs/CONTRIBUTING; paper build DOI/date
  corrected; `docs/advanced_usage.rst` + `docs/troubleshooting.rst` examples
  rewritten against real signatures. Dependency floors intentionally
  unchanged (unverified bumps not committed). See `CHANGELOG.md` [Unreleased].

## Workflowz review — 2026-11-10

- [ ] **WJ-01 (P1)** — Stop returning an identity matrix as `genetic_correlations`. Observable outcome: `EvolutionSampler` population statistics no longer fabricate perfect self-correlation / zero cross-correlation; unidentifiable genetic correlations are reported as `np.nan` (or an explicit `'available': False` marker) consistent with the module contract. Evidence: `src/evojump/evolution_sampler.py:796-797` (`# Genetic correlations (placeholder)` / `genetic_correlations = np.eye(n_phenotypes)`), contradicting `src/evojump/evolution_sampler.py:7-9` and `:838-839`. Acceptance probe (repo root): add a small two-phenotype population regression through the real `EvolutionSampler` statistics path and run `.venv/bin/python -m pytest tests/test_evolution_sampler.py -q --no-cov`; expected: `genetic_correlations` is unavailable/`np.nan` under the existing schema, never the placeholder identity matrix. Dependencies/constraints: none beyond the existing venv; follow the established `np.nan` + `'available'` marker convention already used for heritability in the same file — do not invent a new result schema.

- [ ] **WJ-02 (P2)** — Reconcile the three disagreeing version strings. Observable outcome: package metadata, runtime `__version__`, and README status agree on one released version. Evidence: `pyproject.toml:8` (`version = "0.6.0"`), `src/evojump/__init__.py:37` (`__version__ = "0.5.2"`), `README.md:16` (`Current state: v0.5.2`). Acceptance probe (repo root): `.venv/bin/python -c "import evojump, importlib.metadata as m; assert evojump.__version__ == m.version('evojump'), (evojump.__version__, m.version('evojump'))"` after a fresh editable install, and confirm the README "Status at a glance" line names the same version. Dependencies/constraints: the correct target version is a maintainer/release decision (0.6.0 in pyproject may be an unreleased bump) — do not blindly bump; align `__init__.py` and README to the version the maintainer confirms, and note that `uv.lock` may also need regeneration only if the version there is stale (verify before changing).

- [ ] **WJ-03 (P2)** — Eliminate the silent `0.0` skewness fallback in the visualizer. Observable outcome: when `scipy.stats.skew` fails or returns a non-finite value, the skewness metadata is `np.nan` (or the key carries an explicit error marker) instead of a fabricated `0.0` that reads as "perfectly symmetric". Evidence: `src/evojump/trajectory_visualizer.py:1174-1178` (`except:` / `return 0.0`); check the adjacent `_compute_kurtosis` at `src/evojump/trajectory_visualizer.py:1180-1182` for the same pattern before editing. Acceptance probe (repo root): add a constant-array regression through the real skewness helper and run `.venv/bin/python -m pytest tests/test_trajectory_visualizer.py tests/test_viz_lane_kde.py -q --no-cov`; expected: undefined skew is `np.nan` or explicitly error-marked rather than `0.0`, with ordinary finite skew unchanged. Dependencies/constraints: scipy is a hard dependency (`pyproject.toml:29`), so the `except` should catch numerical/non-finite results specifically rather than ImportError; keep the change inside the existing return-type contract (float or nan) so downstream plot annotations do not break.
