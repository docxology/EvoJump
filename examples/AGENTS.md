# AGENTS.md — `EvoJump/examples/`

Demo/orchestration scripts (verified 2026-08-29; 14 files):
advanced_features_demo.py, animation_demo.py, comprehensive_advanced_analytics_demo.py, comprehensive_animation_demo.py, comprehensive_demo.py, drosophila_case_study.py, enhanced_animation_demo.py, performance_benchmarks.py, simple_animation_demo.py, simple_orchestrator.py, working_demo.py (superseded scripts live in `archive/`: basic_usage_fixed.py, thin_orchestrator_examples.py, thin_orchestrator_working.py) plus `README.md`. Thin orchestrators over `src/evojump/` —
business logic stays in the package. Several scripts are superseded variants
(`working_demo.py` top-level; `basic_usage_fixed.py` and `thin_orchestrator_*.py` in `archive/`); prefer
`comprehensive_demo.py` and `drosophila_case_study.py`. `__pycache__/` is
generated.

## Gotchas
- All demos write CWD-relative paths (the repo root when run via
  `run_all_examples.py`): `demo_outputs/`, `evojump_outputs/`,
  `comprehensive_analytics_outputs/`, `comprehensive_animation_outputs/`,
  `enhanced_animations/`, `simple_animation_outputs/`,
  `simple_orchestrator_outputs/`, `animation_outputs/`, `outputs/figures/`,
  `outputs/benchmarks/`, and `drosophila_case_study_outputs/`. Several demos
  also drop scratch CSVs into the CWD (`demo_data.csv`,
  `comprehensive_sample_data.csv`, `animation_data.csv`,
  `multi_condition_data.csv`, `animation_rich_data.csv`,
  `orchestrator_data.csv`, ...).
Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.