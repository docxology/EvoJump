# AGENTS.md — `EvoJump/tests/`

Live roster: run `ls tests/test_*.py` (16 files verified 2026-08-31). One test
module per source module plus `test_drosophila_case_study.py` (end-to-end case
study), `test_audit_regression_2026_08_30.py` (v0.2.0 regression pins), the
methods-lane modules (`test_methods_lane_changepoints.py`,
`test_methods_lane_laserplane.py`, `test_methods_lane_postaudit.py`), and the
viz-lane modules (`test_viz_lane_animation.py`, `test_viz_lane_heatmap.py`,
`test_viz_lane_kde.py`). See root `README.md` "Test Files Overview" for the
annotated table.

## Conventions
- Real data / real computation; no mocks (root-repo policy).
- Run from the EvoJump root:
```bash
.venv/bin/python -m pytest tests/ -q --no-cov
```
(`uv run` stalls under heavy load — see root README.)
- `__pycache__/` is generated; never document inside it.
Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.