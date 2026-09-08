# AGENTS.md — `EvoJump/src/evojump/`

The `evojump` package. Module map (verified on disk 2026-08-29):

- `datacore.py` — `TimeSeriesData`, `DataCore`, `MetadataManager`: ingestion,
  QC, interpolation, normalization (z-score/min-max/robust).
- `jumprope.py` — the sweeping "jumprope" distribution model.
- `laserplane.py` — fixed analytical plane / cross-section computation.
- `analytics_engine.py` — statistical analysis over cross-sections.
- `evolution_sampler.py` — evolutionary sampling strategies.
- `trajectory_visualizer.py` — plotting (heatmaps, ridges, violins, phase portraits).
- `cli.py` — command-line entry point.
- `__init__.py` — package exports.

## Gotchas
- `__pycache__/` is generated; ignore it.
- Keep public API in sync with `docs/api_reference.rst` and root README.
Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.