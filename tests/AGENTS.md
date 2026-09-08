# AGENTS.md — `EvoJump/tests/`

Live roster: run `ls tests/test_*.py` (17 files verified 2026-09-08). One test
module per source module plus `test_drosophila_case_study.py` (end-to-end case
study), `test_audit_regression_2026_08_30.py` (v0.2.0 regression pins), the
methods-lane modules (`test_methods_lane_changepoints.py`,
`test_methods_lane_laserplane.py`, `test_methods_lane_postaudit.py`), the
viz-lane modules (`test_viz_lane_animation.py`, `test_viz_lane_heatmap.py`,
`test_viz_lane_kde.py`), and `test_property_invariants.py`
(hypothesis-based property suite; shared builders live in `conftest.py`).
See root `README.md` "Test Files Overview" for the annotated table.

## Conventions
- Real data / real computation; no mocks (root-repo policy).
- Run from the EvoJump root:
```bash
MPLBACKEND=Agg .venv/bin/coverage run --source=src/evojump -m pytest tests/ -q
.venv/bin/coverage report --fail-under=95
```
(`uv run` stalls under heavy load — see root README.)
- `__pycache__/` is generated; never document inside it.
Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.