# EvoJump — agent test suite

Pytest suite; one test module per source module in `src/evojump/` plus
case-study, regression, lane, and property-invariant modules. Full roster:
run `ls tests/test_*.py` (17 files, verified 2026-09-08).

## Conventions
- Real data / real computation; no mocks (repo-wide policy).
- Configuration (strict markers) lives in `pyproject.toml`
  (`[tool.pytest.ini_options]`); the 95% coverage floor is enforced by
  `coverage report --fail-under=95` (direct `coverage run`, not pytest-cov —
  pytest-cov's plugin init collides with the numpy >= 2.5 module guard).
  Shared synthetic-data builders live in `conftest.py`
  (`make_growth_frame`, `make_population_frame`).

## Run

From the EvoJump root:

```bash
# Fast feedback (no coverage)
.venv/bin/python -m pytest tests/ -q

# Full suite with the 95% coverage gate
MPLBACKEND=Agg .venv/bin/coverage run --source=src/evojump -m pytest tests/ -q
.venv/bin/coverage report --fail-under=95

# One module
.venv/bin/python -m pytest tests/test_datacore.py -q

# Parallel (pytest-xdist)
.venv/bin/python -m pytest tests/ -n auto
```

Do NOT route through `uv run` — it stalls under heavy machine load on this
checkout (documented in root README and CHANGELOG v0.2.0 notes).

## Layout
- `test_*.py` — the suite. `__pycache__/` is generated; never document inside it.

Repo-wide policy: see root `AGENTS.md`.
