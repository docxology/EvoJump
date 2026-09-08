# Changelog

All notable changes to the EvoJump project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-09-08 (documentation accuracy & Zenodo paper deposit)

### Fixed
- **Docs accuracy audit** (4 independent auditors, 38 findings, all verified
  and fixed): docs/changelog.rst fully re-synced (0.3.0-0.5.0 were missing);
  broken `:doc:` cross-reference in quickstart; sphinx config referencing
  nonexistent `_static/custom.css` and `cover.png`; stale "(v0.2.0)" version
  labels in advanced_usage/installation; README: stale 667→684 test counts,
  gallery captions corrected against the actual PNGs (copula τ 0.440 / ρ 0.643,
  sweep 50% crossing gen 19 / final 0.960, CV plateau ~0.5), broken bibtex
  brace, `uv sync --group dev` → `--extra dev`, `cd evojump` → `cd EvoJump`,
  six missing runtime deps in Requirements, ornstein-uhlenbeck added to the
  process enumeration; AGENTS/CONTRIBUTING/.cursorrules residual 68%-floor
  claims → the real 95% coverage gate; dead absolute repo-wide-policy links
  in four subdirectory AGENTS.md files repointed to the repo root; paper:
  data-availability statement pointed at the real render scripts, build
  commands/paths fixed, stale output stats corrected, superseded banner on
  the v0.2.0-era verification report, phantom `Analyzer (ABC)` root removed,
  actual synthetic-data generator documented in 10_figures (was described as
  N(10,1) sampling).
- CLI: removed the dead, never-consumed `--config` argparse flag.
- run_tests.py: `--coverage` reworked to direct `coverage run` +
  `coverage report --fail-under=95` (the old pytest-cov injection crashes
  under numpy ≥ 2.5); floor comment corrected; venv python pinned.

### Added
- README Citation section: Zenodo block linking the concept DOI (all
  versions) and noting the record archives the source snapshot + compiled
  paper PDF.
- Zenodo release records now include the compiled paper PDF
  (`evojump_paper.pdf`) alongside the source archive.

## [0.5.0] - 2026-09-08 (docs, manuscript & visualization polish)

### Added
- README Gallery: five annotated figures (comprehensive nine-panel,
  FBM density heatmap with dominant-timescale annotation, Gaussian copula
  with τ/ρ box, Drosophila selective sweep with s/w/50%-crossing stats box,
  20-marker network with threshold/edge counts).
- Docs: new sections for spectral coherence (`coherence_column`), spatial
  analysis (`spatial_weights`/`weights_kind`), robust M-estimators, proper
  CCA, cross-section distribution comparison (`rng=`), evolutionary genetic
  parameters (`available` marker, per-trait dicts), comprehensive-report
  column selection, and the reproducibility (seed/rng) surface; every code
  snippet in docs/ ast-parsed and execution-verified; api_reference gained
  all previously undocumented public classes.
- Visualization: centralized rcParams style helper (120 dpi, constrained
  layout, desplined axes, subtle grid); fitted-parameter annotations on
  model-comparison panels; AICc-winner notes on distribution panels;
  legends on all multi-line cross-section/violin/ridge/animation panels.
- Paper figures: per-model fitted-parameter legend labels, (a)-(i) panel
  captions with units, colorbar units + dominant-timescale annotation,
  Gaussian copula ρ box, Drosophila sweep/network stats boxes — all
  regenerated deterministically (seed 42).

### Changed
- Author identity resolved to Daniel Ari Friedman (Active Inference
  Institute, ORCID 0000-0001-6232-9096) across paper.md YAML, README,
  pyproject, CITATION.cff, .zenodo.json; manuscript date September 2026;
  methods changelog in 05_implementation.md extended through v0.4.0;
  12_code.md listings updated to v0.5.0 API (seed=, copula options,
  genetic_parameters 'available'); TODO.md Major items closed.
- PlotConfig.dpi default 100 → 120.

### Fixed
- docs/quickstart + examples snippets: two code blocks passed DataFrames to
  `DataCore.load_from_csv` (crash); broken code fence in 12_code.md;
  short RST underlines; stale example roster in examples/AGENTS.md.

## [0.4.0] - 2026-09-08 (test-suite hardening & release pass)

### Changed
- **Coverage gate**: enforcement moved from pytest-cov to direct
  `coverage run` + `coverage report --fail-under=95` — pytest-cov's
  plugin-time imports collide with the numpy >= 2.5
  "cannot load module more than once per process" guard on this stack
  (plain `coverage run -m pytest` is unaffected). Floor raised 68% → **95%**.
- **Test suite grown and refactored**: 412 → ~600 tests across 17 files;
  per-module line coverage now datacore **100%**, cli **100%**,
  laserplane **100%**, trajectory_visualizer **99.5%**, analytics_engine
  **99.4%**, jumprope **99%**, evolution_sampler **98%** (overall ≥ 95%).
- Shared synthetic-data builders centralized in `tests/conftest.py`
  (`make_growth_frame`, `make_population_frame`); duplicated per-file
  builders deleted; repeated scenario loops parametrized across all modules;
  every stochastic test seeded; weak assertions (`fig is not None`,
  bare `pytest.raises(Exception)`, tautologies) replaced with observable
  contracts.
- New hypothesis property-invariant suite (`tests/test_property_invariants.py`,
  19 properties): interpolation row/order/idempotence, outlier-mask
  order-independence, KM bounds/monotonicity/CI bracketing, FBM
  `dt**(2H)` variance scaling, log-likelihood dominance over misspecified
  parameter boxes, selection gradient == standardized regression slope,
  robust-estimator contamination bounds, order-statistic median-CI coverage.
- `test`/`dev` extras now include `hypothesis` and `pytest-xdist`
  (uv.lock updated); `pytest tests/ -n auto` parallel runs documented.

### Fixed
- laserplane: `DistributionComparer._ad_ksample_statistic` implemented an
  inverted Scholz-Stephens statistic (larger under the null than under
  separation), making the `anderson` permutation fallback's p-values
  meaningless; replaced with the correct eq. 7 midrank formula, verified
  exactly equal to scipy's implementation across randomized trials incl. ties.
- analytics_engine: removed the `verbose` kwarg from
  `grangercausalitytests` (removed in statsmodels 0.15) — the entire Granger
  causality path had been dead code returning error dicts on every call.

## [0.3.0] - 2026-09-08 (comprehensive review & release pass)

A full-repo review pass: every module, test file, docs page, example, and the
manuscript audited by independent reviewers; findings verified and fixed.

### Fixed
- **JumpRope**: geometric jump-diffusion one-jump log-likelihood now includes
  the drift shift and diffusion variance (previously omitted, biasing fits);
  OU log-likelihood is the full Poisson-Gaussian mixture (k up to 20);
  `fit()` evaluates objectives on parameter copies so optimizer failure can no
  longer leak mid-optimization parameters; FBM diffusion standardized to
  std-deviation units across all processes; Levy alpha estimated via an
  empirical characteristic-function slope (recoverable below 2, previously
  structurally biased to 2.0); jump-time detection uses a robust MAD threshold
  instead of a per-path 95th percentile that flagged ~5% of any diffusion;
  FBM Hurst regression lag alignment fixed.
- **LaserPlane**: `compare_distributions` now returns populated
  test_statistics/p_values/effect_sizes (Cohen's d) — previously empty dicts;
  beta fits compute likelihood/AIC/BIC/Vuong on the scaled fit data with a
  change-of-variables Jacobian (the beta branch was previously dead code that
  never fit); lognormal/gamma information criteria computed on the positive
  subset used for fitting; `median_ci` is a true order-statistic confidence
  interval for the median (previously the central 95% of the data); `rng`
  threads through public comparison/bootstrap APIs for reproducible p-values.
- **AnalyticsEngine**: CCA solves the proper generalized eigenproblem
  (`cov11^{-1} cov12 cov22^{-1} cov21`) and returns both coefficient sets;
  survival analysis drops time/event pairs jointly and validates 0/1 events;
  seasonality auto-detects the period per column (previously leaked across
  columns); copula/bayesian analyses drop NaN pairs jointly; Frank copula
  parameter solved from the exact Kendall-tau relation; student copula
  implemented; Huber/Tukey/Rousseeuw-Croux Sn robust estimators are real
  M-estimators (previously median placeholders); variance changepoint
  detection gated by a Bonferroni-corrected F-test; 'information' changepoint
  method is a real BIC segmentation; spectral coherence (MSC) available via
  `coherence_column`; spatial analysis computes true Moran's I from
  `spatial_weights`; regime switching guards zero-variance features and
  computes transitions from the full label sequence; seeded Bayesian
  regression (`seed=`).
- **DataCore**: interpolation is positionally stable with duplicate index
  labels (previously expanded rows via `.loc` cross-product); outlier removal
  applies one combined order-independent mask, never treats NaN as an outlier,
  and refuses to empty a dataset; NaN time values raise instead of being
  backfilled; HDF5 load/save round-trip (including the save layout, string
  columns, group flattening, and clear unequal-length errors); aggregation
  unions phenotype columns across datasets; a single `TimeSeriesData` is
  accepted and `append()` added; quality metrics always include
  `temporal_consistency` plus a per-column outlier breakdown; empty metadata
  files raise a clear error; unknown phenotype columns raise in
  `filter_by_phenotype_range`.
- **EvolutionSampler**: genetic-parameter and selection placeholders replaced
  with real estimates (or NaN with an `available` marker — previously
  fabricated zeros); phylogenetic covariance is a double-centered (Gower)
  kernel with PSD validation (previously the raw distance matrix); selection
  gradient is the standardized regression slope (documented); Moran's I is
  computed only with a compatible distance matrix; sampling diagnostics no
  longer mutate the caller's dict; MCMC acceptance rate excludes burn-in.
- **CLI**: a global `--output` before the subcommand now survives subparser
  defaults (previously silently dropped); `visualize --interactive` persists
  the Plotly figure as HTML (previously exited 0 writing nothing); `-v/-vv`
  actually change log levels; exit codes consistently 0/1/2;
  `--time-column` added to fit/sample with input validation everywhere;
  `analysis_results.json` reports real trajectory counts.
- **TrajectoryVisualizer**: `networkx` import restored so network plots render
  instead of silently falling back; `tick_labels=` for matplotlib>=3.9;
  Agg backend only set when no backend configured; consistent CI band
  labeling (95% CI of the mean vs ±1 SD); heatmaps ignore NaN instead of
  imputing zeros; deterministic phase-portrait subsampling; `close=` figure
  ownership parameter and `_save` helper; animation fps derived from the frame
  interval; distribution quantiles stored in animation frames.
- **Examples**: repaired three example scripts that shipped with corrupted
  string literals (`animation_demo.py`, `comprehensive_animation_demo.py`,
  `comprehensive_demo.py`) — `run_all_examples.py` failed 3/11 before; the
  Drosophila case study now emits strict JSON (no NaN tokens) with real
  PCA/network/Bayesian statistics and a per-marker correlation network figure
  (previously plotted DataFrame columns and placeholder zeros).

### Changed
- License changed from MIT to Apache License 2.0.
- Python floor aligned at 3.9 across `requires-python`, classifiers, black,
  and mypy (previously mixed 3.8/3.9 metadata).
- `selection_differential`/`selection_response` in EvolutionSampler results
  are now per-trait dicts (previously scalar placeholders).
- `TimeSeriesAnalyzer.detect_change_points('cusum')` delegates to
  ChangePointDetector (result keys `change_magnitude`/`z_score`).

### Added
- GitHub Actions CI workflow running the suite on Python 3.9-3.12.
- `.zenodo.json` and `CITATION.cff` release metadata.
- `.gitignore`; build/coverage artifacts (`coverage.xml`, `.coverage`,
  `__pycache__`, `.DS_Store`, `paper/output/`) removed from version control.
- Parameter-recovery, seeding-reproducibility, and log-likelihood tests across
  the stochastic processes; per-method visualization lane tests.

### Removed
- `demo_testing.py` (advertised pytest flags that do not exist) and
  `run_all_tests.py` (pure alias of `run_tests.py`).
- `paper/latex_template.tex` (dead template referencing a nonexistent
  bibliography) and ten unreferenced legacy figure PNGs.

### Docs (2026-08-30 documentation deep pass)
- README: removed fictional `run_all_tests.py` flags (`--all`, `--benchmark`,
  `--profile`, `--memory`, `--lint`, `--docs` all fail with pytest
  "unrecognized arguments"); documented the wrapper's real forward-to-pytest
  contract and canonical `.venv/bin/python -m pytest` invocations.
- README: badge block and all repo URLs corrected from the nonexistent
  `github.com/evojump/evojump` to `github.com/docxology/EvoJump`; unverifiable
  CI/coveralls/PyPI/RTD badges removed; Python 3.9+ packaging bound stated.
- README: Quick Start verified by execution (runs verbatim under the venv).
- docs: API reference now lists FBM/CIR/Levy process classes and the
  `shortest_path_analysis`, `wavelet_analysis`, `copula_analysis`,
  `extreme_value_analysis`, `regime_switching_analysis` methods (all verified
  present in source); Sphinx version bumped to 0.2.0.
- docs/troubleshooting: added verified v0.2.0 gotchas — SciPy >= 1.15
  two-sample `kstest` breakage (use frozen CDFs), `uv run` stalls under heavy
  load (invoke `.venv/bin/python` directly), pandas 3.x numeric-dtype
  selection in DataCore.
- docs: Python 3.9+ (not 3.8+) across installation/architecture/contributing/
  api_reference; JumpRope documented as seven stochastic processes.

## [0.2.0] - 2026-08-30 (audit & hardening pass)

### Fixed
- **Packaging**: `requires-python` bounded to `>=3.9,<3.15`; `cupy-cuda12` extra
  marker restricted to `python_full_version < 3.14` on linux x86_64 and uv
  `environments` limited to darwin/linux so resolution succeeds; setuptools
  package discovery fixed (`where = ["src"]`) — installs were broken before.
- **JumpRope**: OU and geometric jump-diffusion log-likelihoods replaced with
  exact Poisson-jump Gaussian mixtures (previously ignored jump probability
  mass); compound-Poisson likelihood now exact (previously only `-lambda*dt`);
  all processes accept a seeded `Generator` — `JumpRope.fit(seed=...)` and
  `generate_trajectories(seed=...)` are reproducible; unused `numba`/`cuda`
  import removed (no GPU code exists despite the extra).
- **EvolutionSampler**: importance sampling now uses real exponential-tilt
  weights with systematic resampling and records effective sample size;
  MCMC is a real Metropolis-Hastings chain and records acceptance rate
  (previously both silently aliased plain Monte Carlo); phylogenetic signal
  computed as Moran's I on the distance matrix (previously hardcoded 0.0);
  parent-offspring heritability refuses row-order pseudo-pedigrees and
  returns NaN with a warning unless explicit `parent`/`offspring` columns
  exist.
- **AnalyticsEngine**: Kaplan-Meier now produces Nelson-Aalen hazards,
  Greenwood CIs, and a true KM median (previously hazard = constant 0.1
  placeholder, median = plain median of times); largest Lyapunov exponent via
  the Rosenstein method and correlation dimension via Grassberger-Procaccia
  slopes (previously hardcoded placeholders); Bayesian linear regression uses
  the conjugate Normal-Inverse-Gamma posterior with split R-hat and a real
  log-evidence approximation (previously fake `r_hat=1.0`, evidence `0.0`).
- **CLI**: `visualize` subcommand called instance methods on the class and
  crashed with TypeError; now instantiates `TrajectoryVisualizer`, generates
  trajectories on demand, and forces the Agg backend headless; `fit`/`analyze`
  `--model-type` choices extended to all seven supported processes; missing
  `_validate_input_file` implemented (strict `.csv` suffix, FileNotFoundError
  -> exit 1); `--output` accepted after subcommands (analyze/fit/visualize/
  sample) with global fallback; `--samples` alias for `--n-samples`; analyze
  writes `analysis_results.json` + `data_summary.json`; sample output uses
  real phenotype column names in long format; `fit` seeds the model.
- **Drosophila case study**: `DrosophilaPopulation.generations` default
  restored to 10 (demo passes 100 explicitly); generated table now carries
  `genotype` (0/1), `phenotype`, and `allele_frequency` columns matching the
  documented analysis contract; `individual_id` kept as object dtype;
  boolean assessors return real Python bools (previously `np.bool_`).
- **DataCore/JumpRope edge cases**: `DataCore` accepts (and defers on) empty
  series — `JumpRope.fit` raises a clear error for empty data; unfitted
  `JumpRope.generate_trajectories` falls back to the process's own parameters;
  laserplane KS test uses frozen CDFs (scipy >= 1.15 name+args breakage).
- **Flake control**: seeded the stochastic Levy heavy-tail test.
- **DataCore**: missing-data interpolation is temporally ordered (sort by
  time, interpolate, restore order) so results are row-order independent;
  `dataset_id` in aggregation is stable instead of `id()`-based.
- **TrajectoryVisualizer**: animation axes fixed across frames (previously
  rescaled per frame); frame CI documented as the SEM band it computes.
- **README**: capability claims aligned with actual implementations (UMAP,
  deep learning, AutoML, real-time claims removed).

### Added
- `tests/test_audit_regression_2026_08_30.py`: 17 regression tests covering
  every fix above (real data/computation, no mocks).
- `NetworkAnalyzer.shortest_path_analysis()` (weighted + unweighted shortest
  paths; `construct_correlation_network` now persists the graph — previously
  it was computed and discarded).
- CLI input validation (`_validate_input_file`), structured
  `analysis_results.json` / `data_summary.json` outputs, subcommand-level
  `--output`/`--samples` aliases, on-demand trajectory generation in
  `visualize`, and headless-safe logging configuration (package logger, not
  root).

### Changed
- Coverage gate set to an honestly measured 68% floor (the previous 95% gate
  was aspirational and unmet by every recorded run; visualizer module is at
  34%).

## [0.1.0] - 2024-10-01

### Added

#### Core Framework
- **DataCore Module**: Complete data management system
  - Time series data ingestion and validation
  - Multiple data format support (CSV, HDF5, SQL)
  - Data preprocessing and quality control
  - Missing data interpolation methods
  - Outlier detection and removal
  - Normalization methods (z-score, min-max, robust)
  - Metadata management system

- **JumpRope Engine**: Stochastic process modeling
  - Jump-diffusion model implementation
  - Ornstein-Uhlenbeck process
  - Geometric jump-diffusion
  - Compound Poisson process
  - Fractional Brownian Motion (FBM)
  - Cox-Ingersoll-Ross (CIR) process
  - Lévy processes
  - Parameter estimation and model fitting
  - Trajectory generation and simulation
  - Cross-section computation

- **LaserPlane Analyzer**: Cross-sectional analysis
  - Distribution fitting (normal, lognormal, gamma, beta, etc.)
  - Statistical comparison methods (KS test, Mann-Whitney, etc.)
  - Moment analysis and confidence intervals
  - Bootstrap analysis
  - Goodness-of-fit assessment
  - Quantile estimation

- **TrajectoryVisualizer**: Advanced visualization system
  - Static trajectory plotting
  - Animated trajectory sequences
  - Cross-section visualizations
  - Density heatmaps
  - Violin plots for distribution evolution
  - Ridge plots (joyplots) for temporal distributions
  - Phase portraits
  - Landscape analysis plots
  - Model comparison visualizations
  - Publication-quality graphics export

- **AnalyticsEngine**: Statistical analysis suite
  - Time series analysis (trends, seasonality, change points)
  - ARIMA modeling
  - Multivariate analysis (PCA, CCA, cluster analysis)
  - Dimensionality reduction (t-SNE, UMAP)
  - Predictive modeling (random forest, cross-validation)
  - Wavelet analysis
  - Copula methods
  - Extreme value theory
  - Regime switching detection
  - Bayesian inference and model comparison
  - Network analysis and community detection
  - Causal inference methods

- **EvolutionSampler**: Population-level analysis
  - Population dynamics modeling
  - Heritability estimation
  - Selection gradient computation
  - Effective population size estimation
  - Monte Carlo and MCMC sampling
  - Phylogenetic comparative methods
  - Quantitative genetics approaches
  - Selective sweep detection

- **CLI Interface**: Command-line tools
  - Data analysis workflows
  - Model fitting automation
  - Visualization generation
  - Batch processing support

#### Testing Framework
- **173 test methods** across 8 comprehensive test suites
- **95%+ code coverage** requirement
- **Real data testing** - no mocks, biological/synthetic data only
- **Integration testing** - cross-module validation
- **Performance validation** - large dataset testing
- **Multiple testing modes**: quick, full, benchmark, CI/CD
- Test files:
  - `test_datacore.py` - 24 tests
  - `test_jumprope.py` - 22 tests
  - `test_laserplane.py` - 25 tests
  - `test_trajectory_visualizer.py` - 19 tests
  - `test_analytics_engine.py` - 39 tests
  - `test_evolution_sampler.py` - 21 tests
  - `test_advanced_features.py` - 23 tests
  - `test_cli.py` - 20 tests

#### Documentation
- Comprehensive README with installation, usage, and examples
- Testing framework documentation (AGENTS.md)
- API reference documentation
- User guides and tutorials
- Scientific methodology documentation
- Architecture documentation
- Contributing guidelines
- Troubleshooting guide

#### Examples
- Basic usage examples
- Advanced features demonstrations
- Animation examples
- Comprehensive analytics demos
- Drosophila case study
- Performance benchmarks
- Orchestrator examples

### Scientific Applications
- Developmental biology trajectory analysis
- Evolutionary biology population dynamics
- Quantitative genetics analysis
- Agricultural research optimization
- Medical research and biomarker discovery
- Systems biology complex trait modeling

### Key Features
- Novel "cross-sectional laser" metaphor for developmental analysis
- Six different stochastic process models
- Advanced statistical and machine learning methods
- Rich visualization capabilities (static, animated, interactive)
- Scientific rigor with comprehensive validation
- Extensible modular architecture
- High-performance computing support

### Performance
- Efficient vectorized operations using NumPy
- Optional GPU acceleration support
- Parallel processing capabilities
- Memory-efficient data structures
- Streaming algorithms for large datasets

### Dependencies
- Python 3.8+
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Pandas ≥ 1.3.0
- Matplotlib ≥ 3.5.0
- Plotly ≥ 5.0.0
- Scikit-learn ≥ 1.0.0
- PyWavelets ≥ 1.3.0
- NetworkX ≥ 2.6.0
- StatsModels ≥ 0.13.0
- Seaborn ≥ 0.11.0
- And more (see pyproject.toml)

## Release Notes

### Version 0.1.0 Highlights

This initial release represents a complete, production-ready framework for evolutionary ontogenetic analysis. The system has been developed using strict test-driven development (TDD) principles with comprehensive validation at every level.

**Core Innovations**:
1. Novel analytical metaphor connecting developmental and evolutionary biology
2. Multiple stochastic process implementations for biological modeling
3. Comprehensive statistical analysis suite adapted for biological data
4. Advanced visualization system with animation capabilities
5. Rigorous scientific validation with real data testing

**Quality Assurance**:
- All features developed with comprehensive test coverage
- 95%+ code coverage maintained across all modules
- CI/CD ready with automated testing workflows
- Professional code quality with Black, Flake8, MyPy
- Performance benchmarks and optimization validation

**Documentation Quality**:
- Module-level docstrings for all components
- Complete API reference with parameter descriptions
- Type annotations throughout codebase
- Usage examples in docstrings and dedicated examples directory
- Cross-references between related modules

**Scientific Impact**:
The framework provides researchers with novel tools for analyzing developmental and evolutionary processes, comprehensive modeling of complex biological systems, advanced statistical methods, rich visualization for scientific communication, and an extensible framework for custom analyses.

---

## Future Roadmap

### Planned Features
- Deep learning integration for trajectory prediction
- Real-time analysis dashboard
- Cloud-based distributed computing support
- Additional stochastic process models
- Enhanced phylogenetic methods
- Integration with genomic databases
- Interactive web application

### Under Consideration
- R language integration
- Julia language bindings
- GPU-accelerated algorithms
- Distributed data processing
- Real-time streaming analysis
- Mobile visualization apps

---

For complete details on any release, see the GitHub releases page: https://github.com/docxology/EvoJump/releases

For questions or issues, please visit: https://github.com/docxology/EvoJump/issues

