# Computational Implementation

The mathematical elegance of stochastic process theory must be matched by computational efficiency and software engineering rigor to be useful for biological research. This section describes EvoJump's architecture, algorithmic implementations, performance optimizations, and quality assurance practices that transform theoretical models into practical research tools.

## Software Architecture

### Design Principles

EvoJump's architecture balances mathematical rigor, computational efficiency, and accessibility through five principles:

1. **Modularity**: Independent, testable components (data, modeling, analysis, visualization) enable focused development and selective use
2. **Composability**: Well-defined interfaces enable complex analyses through simple combinations with consistent API patterns
3. **Extensibility**: Abstract base classes allow new models via subclassing without modifying core infrastructure
4. **Performance**: NumPy vectorization and intelligent caching optimize critical paths, with support for JIT compilation where beneficial
5. **Usability**: High-level APIs with extensive documentation and examples lower entry barriers while enabling expert customization

### Core Modules

**DataCore**: Data management and preprocessing with time series structures, quality validation, missing data handling, metadata management, and reproducible workflows.

**JumpRope**: Stochastic process modeling with base `StochasticProcess` class and implementations for OU with jumps, fBM, CIR, Lévy, compound Poisson, and geometric jump-diffusion processes. Supports maximum likelihood, method of moments, and Bayesian MCMC estimation.

**LaserPlane**: Cross-sectional analysis implementing the "laser plane" metaphor with distribution fitting, moment computation, goodness-of-fit testing, and bootstrap confidence intervals.

**AnalyticsEngine**: Advanced statistical methods including autocorrelation, spectral analysis, PCA, wavelet transforms, copula fitting, extreme value analysis, and regime-switching detection with publication-ready reporting.

**TrajectoryVisualizer**: Visualization framework with static (matplotlib) and interactive (Plotly) plots, animations, and journal-standard outputs (300+ DPI, colorblind-friendly palettes).

**EvolutionSampler**: Population-level analysis with Monte Carlo sampling, phylogenetic methods, quantitative genetics calculations, and selection analysis.

### Class Hierarchy

## Algorithmic Implementation

This section details key algorithms implementing the stochastic processes and statistical methods. We emphasize clarity and correctness, with performance optimizations applied after validation.

### Stochastic Process Simulation

**Euler-Maruyama Scheme** for SDEs: The Euler-Maruyama method is the stochastic analog of the Euler method for ODEs. It discretizes the continuous-time SDE by approximating integrals as finite sums. The implementation iterates through time steps, generating Wiener increments $dW \sim N(0, \sqrt{dt})$ and updating the trajectory via $X_{t+dt} = X_t + \mu(X_t, t)dt + \sigma(X_t, t)dW$. While simple, this scheme converges to the true solution as the time step decreases (strong convergence of order 0.5, weak convergence of order 1.0). The key insight: Brownian motion increments scale as $\sqrt{dt}$, not $dt$, reflecting their non-differentiable nature.

**Jump Component**: The compound Poisson process is simulated by determining the number of jumps in each time interval $n \sim \text{Poisson}(\lambda dt)$, then drawing jump magnitudes from the specified distribution and summing them. This captures the discrete, stochastic nature of developmental transitions like metamorphosis or environmental regime shifts.

### Parameter Estimation

**Maximum Likelihood via Numerical Optimization**: Parameters are estimated by minimizing the negative log-likelihood using L-BFGS-B optimization with parameter bounds. The log-likelihood is computed from the transition densities of the stochastic process, summed over all observed transitions. This approach provides asymptotically efficient estimates under standard regularity conditions.

**Moment Matching**: For processes with tractable moments, we match empirical moments (sample mean, variance, autocorrelation) to their theoretical expressions. The OU process equilibrium equals the sample mean, the reversion speed is estimated from lag-1 autocorrelation via $\hat{\kappa} = -\log(\hat{\rho})/\Delta t$, and the diffusion coefficient from the equilibrium variance relationship. This method is computationally efficient and provides good initial estimates for more sophisticated inference.

### Wavelet Transform Implementation

The continuous wavelet transform is computed using the PyWavelets library, which provides efficient implementations of multiple wavelet families. For a given signal and set of scales, the CWT computes wavelet coefficients by convolving the signal with scaled and translated versions of the mother wavelet. The power spectrum is obtained by squaring coefficient magnitudes, and the dominant temporal scale is identified as the scale with maximum mean power across all time points. This reveals which frequencies or periodicities dominate the developmental trajectory.

### Copula Fitting

Copula fitting proceeds in two steps: first, transform marginal data to uniform $[0,1]$ distributions using empirical ranks; second, fit the copula dependence structure to these uniform margins. For Gaussian copulas, we transform uniform margins to standard normal via the inverse normal CDF and estimate the correlation parameter. For Clayton copulas, we compute Kendall's $\tau$ and convert to the copula parameter via $\theta = 2\tau/(1-\tau)$. This approach separates marginal distributions from dependence structure, enabling flexible modeling of complex trait relationships.

## Performance Optimization

### Vectorization

Critical computational loops are vectorized using NumPy's array operations, replacing explicit Python loops with optimized C-level operations. This transformation typically provides 10-100x speedups for numerical operations. Vectorization applies array functions directly to entire arrays rather than iterating element-by-element, leveraging SIMD instructions and cache-friendly memory access patterns.

### Computational Efficiency

The framework is designed with performance in mind through NumPy vectorization of core operations. Critical loops use array operations that leverage optimized C-level implementations. The architecture supports JIT compilation via Numba for performance-critical paths when needed, and parallel processing via multiprocessing for independent trajectory generation, though these optimizations are applied selectively based on computational requirements.

### Memory Efficiency

Large datasets are processed in chunks to avoid memory overflow. Data are read, processed, and written in fixed-size blocks, with intermediate results accumulated incrementally. This streaming approach enables analysis of datasets exceeding available RAM, trading modest increases in computation time for dramatic reductions in memory footprint. Chunk size is tuned based on available memory and cache characteristics.

## Testing Framework

### Unit Tests

Each component has comprehensive unit tests verifying individual function correctness. Tests cover normal operation, edge cases, and error conditions. For stochastic processes, tests verify output dimensions, finite values, and basic statistical properties. The test suite uses pytest with fixtures for common test data, ensuring consistent test environments and facilitating debugging.

### Integration Tests

Integration tests verify correct interaction between modules through complete analysis pipelines. A typical test loads data via DataCore, fits a stochastic model with JumpRope, performs cross-sectional analysis with LaserPlaneAnalyzer, and generates visualizations with TrajectoryVisualizer. Assertions verify that data flow correctly between components and that final outputs meet quality criteria. These tests catch interface mismatches and ensure the system works as an integrated whole.

### Validation Tests

Validation tests compare numerical results against analytical solutions and established benchmarks. For the Ornstein-Uhlenbeck process, we simulate long trajectories and verify that empirical stationary moments match theoretical predictions within tolerance. Fractional Brownian motion tests verify correct Hurst parameter estimation. CIR process tests confirm non-negativity and appropriate stationary distributions. These tests establish confidence in implementation correctness and numerical accuracy.

## Documentation System

### Docstring Format

All public functions, classes, and methods include Google-style docstrings documenting parameters, return values, exceptions, and usage examples. This consistent format enables automatic API documentation generation and provides inline help for users. Parameter descriptions include types and semantics; return values specify structure and interpretation; examples demonstrate typical usage patterns. Docstrings serve as both user documentation and developer reference.

### Sphinx Documentation

Complete API documentation is generated automatically from source code docstrings using Sphinx. The documentation system includes module overviews, class hierarchies, function signatures, and cross-references. Mathematical notation in docstrings renders correctly in HTML and PDF outputs. The generated documentation provides searchable, hyperlinked reference material accessible to both novice and expert users.

### Tutorials and Examples

Comprehensive worked examples demonstrate all major features through realistic use cases. Examples progress from basic trajectory fitting to advanced multivariate analysis, copula modeling, and visualization. Each example includes clear objectives, complete working code, expected outputs, and interpretation guidance. Examples serve as both learning materials for new users and templates for researchers adapting EvoJump to their specific problems. All example code is tested as part of the continuous integration pipeline, ensuring examples remain functional as the codebase evolves.

**Note**: Complete code listings for all algorithms and implementations described in this section are provided in Section 12 (Complete Code Listings) for reference and reproducibility.

## Visualization Framework

### Advanced Visualization Types

EvoJump provides multiple innovative visualization methods for developmental trajectory analysis. These visualizations transform numerical results from stochastic process models into interpretable graphics that reveal patterns invisible in raw data. Below we present five key visualization types, each designed for specific analytical purposes.

**Figure 1** presents a comprehensive model comparison across three stochastic processes (Fractional Brownian Motion, Cox-Ingersoll-Ross, and Jump-Diffusion). This multi-panel visualization includes: (a) mean trajectories with confidence intervals comparing overall developmental trends, (b) final distribution comparisons showing endpoint variability, (c) jump pattern detection highlighting discontinuous changes, (d) statistical properties analysis (mean, standard deviation, coefficient of variation, skewness, kurtosis), (e) trajectory variability over time, (f) model parameter comparison, (g) trajectory clustering by final values, (h) performance metrics evaluation, and (i) summary statistics for each model type.

![Comprehensive model comparison across stochastic processes showing trajectory patterns, statistical properties, parameter estimates, and performance metrics for fBM, CIR, and Jump-Diffusion models.\label{fig:model_comparison}](figures/figure_1_comparison.png){ width=95% }

**Figure 2** provides a comprehensive trajectory analysis using the Fractional Brownian Motion model as an exemplar. This 9-panel figure includes: (a) individual trajectories with mean and standard deviation bands, (b) density heatmap showing temporal evolution, (c) cross-sectional distributions at key timepoints, (d) violin plots revealing distribution shapes, (e) ridge plot displaying temporal progression, (f) phase portrait analysis of phenotype dynamics, (g) statistical summary with mean trends and coefficient of variation, (h) model parameter diagnostics, and (i) evolutionary change analysis comparing initial vs. final phenotypes.

![Comprehensive trajectory analysis of fBM model showing individual trajectories, density evolution, distribution comparisons, phase space dynamics, and statistical summaries across developmental time.\label{fig:comprehensive}](figures/figure_2_comprehensive.png){ width=95% }

**Figure 3** presents detailed visualizations for each stochastic model type, with four panels per model: (a) trajectory density heatmap showing temporal evolution of phenotypic distributions, (b) violin plots revealing distribution shape evolution at discrete timepoints, (c) ridge plot (joyplot) displaying stacked distributions over time, and (d) phase portrait analysis showing phenotype values versus their rate of change.

![Individual model visualizations for fBM showing trajectory density heatmap.\label{fig:model_fbm}](figures/figure_3_fbm_heatmap.png){ width=85% }

![Individual model visualizations for CIR showing trajectory density heatmap.\label{fig:model_cir}](figures/figure_3_cir_heatmap.png){ width=85% }

![Individual model visualizations for Jump-Diffusion showing trajectory density heatmap.\label{fig:model_jump}](figures/figure_3_jump-diffusion_heatmap.png){ width=85% }

### Implementation Details

The visualization framework provides methods for generating trajectory density heatmaps (with adjustable time and phenotype resolution), violin plots at specified timepoints, ridge plots showing distribution evolution, and phase portraits computed via finite difference approximation. Each visualization method supports both static (matplotlib) and interactive (Plotly) output modes, enabling publication-quality graphics and exploratory analysis. The consistent API across visualization types simplifies generation of comprehensive figure panels. Complete code examples are provided in Section 12.

## Package Management with UV

### Project Configuration

EvoJump uses modern Python packaging standards with pyproject.toml configuration. Dependencies include NumPy (>=1.21.0) for numerical operations, SciPy (>=1.7.0) for statistical functions, pandas (>=1.3.0) for data management, matplotlib (>=3.5.0) and Plotly (>=5.0.0) for visualization, scikit-learn (>=1.0.0) for machine learning methods, PyWavelets (>=1.3.0) for wavelet analysis, NetworkX (>=2.6.0) for network analysis, statsmodels (>=0.13.0) for statistical modeling, and seaborn (>=0.11.0) for enhanced visualizations. Version constraints balance feature requirements with compatibility. The package requires Python >=3.8.

### Development Workflow

UV provides fast, reliable dependency resolution and environment management, and is the exclusive package manager for EvoJump. Development workflow includes: creating isolated virtual environments with `uv venv`, installing packages with `uv add`, syncing dependencies with `uv sync`, running the test suite via `uv run pytest`, and building documentation with `uv run sphinx-build`. UV's speed and reproducibility ensure consistent, reliable installations across all platforms.

### Reproducible Environments

Lock files generated from pyproject.toml ensure reproducible environments across different systems and time periods. The lock file pins exact versions of all dependencies and their transitive dependencies, preventing subtle bugs from version drift. Installation from lock files guarantees identical environments in development, testing, and production contexts, supporting reproducible research.
