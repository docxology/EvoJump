# EvoJump — agent test suite

Pytest suite; one test module per source module in `src/evojump/` plus case-study,
regression, and lane modules. Full roster: run `ls tests/test_*.py` (16 files,
verified 2026-08-31).

## Conventions
- Real data / real computation; no mocks (repo-wide policy).
- Configuration (coverage floor, strict markers) lives in `pyproject.toml`
  (`[tool.pytest.ini_options]`).

## Run

From the EvoJump root:

```bash
# Fast feedback (no coverage)
.venv/bin/python -m pytest tests/ -q --no-cov

# Full suite with coverage floor enforcement (floor lives in pyproject.toml)
.venv/bin/python -m pytest tests/

# One module
.venv/bin/python -m pytest tests/test_datacore.py -q --no-cov
```

Do NOT route through `uv run` — it stalls under heavy machine load on this
checkout (documented in root README and CHANGELOG v0.2.0 notes).

## Layout
- `test_*.py` — the suite. `__pycache__/` is generated; never document inside it.

Repo-wide policy: see root `AGENTS.md`.
