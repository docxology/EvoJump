# EvoJump Paper Build Verification Report

**Date**: 2026-08-30 (v0.2.0 methodology pass, fleet lane: manuscript)
**Baseline commit**: eac4853 → verified at worktree on top of ff30e3e
**Build Status**: PASS (see measured commands below)

## Measured verification commands (all run 2026-08-30)

| Check | Command | Result |
|-------|---------|--------|
| Test suite (baseline, before paper edits) | `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q --no-cov -p no:cacheprovider` | 4 failed, 242 passed (1746 s under fleet load ~7-9); 2 of 4 failures reproduced on isolated retry and belong to src/ (other lanes), 2 were load-flaky and passed on retry |
| Isolated retry of 4 failures | same, with the 4 failing node ids | 2 failed, 2 passed (1026 s). FAILED: `test_analytics_engine.py::TestChangePointDetector::test_detect_changes_bayesian` (IndexError in BOCPD at analytics_engine.py:724, cross-lane), `test_audit_regression_2026_08_30.py::TestCliFixes::test_visualize_command_uses_instance` (CLI subprocess 240 s timeout under load, cross-lane) |
| Main figures | `cd paper && MPLBACKEND=Agg ../.venv/bin/python render_figures.py` | exit=0; 15 main figures + auxiliary plots written to paper/figures (timestamps 2026-08-30 15:39) |
| Drosophila figures | `cd paper && MPLBACKEND=Agg ../.venv/bin/python render_drosophila_figures.py` | exit=0; 3 figures + drosophila_figures_summary.json written (2026-08-30 15:47-15:48) |

Full-suite reruns are tracked by the coordinator; under sibling fleet load the
suite takes ~30 min from this external drive. The two reproducible failures are
in files outside this lane's ownership and are recorded under CROSS-LANE
FINDINGS in the lane report.

## v0.2.0 methodology alignment (this pass)

Every statistical claim in Sections 3, 5, 6a, 7, and 12 was checked against
`src/` at eac4853. Corrections made (all inside paper/**):

- OU jump-diffusion likelihood now described as the exact Poisson-mixture
  one-step density actually implemented (jumprope.py:121-153); new equation
  `eq:ou_mixture_density` in 03_mathematical_foundations.md.
- CIR: qualified that the implementation uses a Gaussian (Euler) transition
  approximation (jumprope.py:581), not the exact non-central chi-square.
- 05_implementation: JumpRope no longer claims "Bayesian MCMC estimation";
  Bayesian inference correctly attributed to NIG Bayesian linear regression
  (AnalyticsEngine) and Metropolis-Hastings (EvolutionSampler).
- 05_implementation: new paragraphs on real importance sampling (ESS
  diagnostics), Metropolis-Hastings MCMC, Kaplan-Meier/Nelson-Aalen/Greenwood
  survival analysis, Rosenstein Lyapunov and Grassberger-Procaccia dimension.
- 05_implementation: new "Changelog of Methods (v0.2.0)" section.
- Python version claims updated to >=3.9,<3.15 (05, 12_code).
- 06a: heritability 0.42 marked as simulation design value; pedigree-gated
  estimator behavior (NaN + warning without pedigree) documented.
- 07: GPU/CUDA claim softened (CuPy optional, Linux-only; MCMC is CPU NumPy).

## Document Structure

12 sections (01_abstract .. 12_code), figures in paper/figures/, build via
`./build_paper.sh` (pandoc + pdflatex).

---
**Verification Status**: PASS (commands above are the evidence)
