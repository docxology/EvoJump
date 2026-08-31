# AGENTS.md — `EvoJump/src/`

Package source root. Contains one package: `evojump/` (modules: `datacore.py`,
`jumprope.py`, `laserplane.py`, `analytics_engine.py`, `evolution_sampler.py`,
`trajectory_visualizer.py`, `cli.py`, `__init__.py`) plus `__pycache__/`
(generated, undocumented by design) and `evojump.egg-info` metadata when built.

## Invariants
- Thin-orchestrator style: business logic lives here; `examples/` and root
  `run_*.py` scripts orchestrate it.
- Tests for every module live in `../tests/` (one test file per module).
- No cross-project imports; EvoJump is standalone.

## Verify
```bash
.venv/bin/python -m pytest ../tests/ -q   # from EvoJump root: .venv/bin/python -m pytest tests/ -q --no-cov
```
Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.