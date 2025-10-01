# Figure Generation and Reproducibility

Reproducibility is a cornerstone of scientific research. This section provides complete technical details for regenerating all figures in this paper, ensuring that other researchers can validate our results, adapt our methods, and build upon our work.

## Technical Details

All figures in this paper were generated using EvoJump's visualization framework applied to synthetic developmental data with known parameters. No manual editing or post-processing was performed—figures represent direct outputs from the software, demonstrating the publication-readiness of automated visualizations.

### Data Generation Parameters

Synthetic developmental trajectories were generated with parameters chosen to mimic realistic biological variation:

- **Sample size**: 100 individuals × 100 timepoints (representing a moderately-sized developmental study with high temporal resolution)
- **Time span**: 0 to 10 time units (arbitrary units scalable to days, weeks, or developmental stages depending on organism)
- **Initial conditions**: Normal distribution with mean 10.0, standard deviation 1.0 (representing natural variation in starting phenotypes)
- **Model fitting**: Maximum likelihood estimation for all stochastic processes, using L-BFGS-B optimization with multiple random initializations to avoid local optima
- **Visualization engine**: Matplotlib 3.5+ for static publication-quality plots, Plotly 5.0+ for interactive versions (not shown in paper)
- **Image format**: PNG at 300 DPI for raster graphics, PDF for vector graphics where appropriate
- **Color schemes**: Colorblind-friendly palettes throughout (viridis for sequential data, plasma for diverging data) ensuring accessibility for readers with color vision deficiencies

### Figure Specifications

The figures presented throughout this paper demonstrate comprehensive multi-panel visualizations:

1. **Model Comparison** (Figure 1): Nine-panel comparison across three stochastic processes (fBM, CIR, Jump-Diffusion) including trajectory patterns, statistical properties, parameter estimates, clustering analysis, and performance metrics.

2. **Comprehensive Trajectory Analysis** (Figure 2): Nine-panel analysis of fBM model trajectories including individual trajectories, density heatmaps, cross-sectional distributions, violin plots, ridge plots, phase portraits, statistical summaries, model diagnostics, and evolutionary change analysis.

3. **Individual Model Visualizations** (Figure 3): Four-panel visualizations for each stochastic model (fBM, CIR, Jump-Diffusion) showing trajectory density heatmaps, violin plots, ridge plots, and phase portraits with detailed distribution evolution and dynamical systems perspectives.

4. **Copula Analysis** (Figure 4): Bivariate dependence analysis using rank-based transformations showing Kendall's $\tau$ correlation between early and late developmental phenotypes with statistical significance testing.

## Reproducibility

All figures can be reproduced using EvoJump's visualization framework. The general workflow involves: (1) generating synthetic developmental data with specified parameters, (2) creating a DataCore instance to manage the time series data, (3) fitting appropriate stochastic process models (fBM, CIR, or jump-diffusion) using JumpRope, and (4) generating visualizations via TrajectoryVisualizer methods. Figure 1 uses `plot_model_comparison()` to compare multiple models, Figure 2 uses `plot_comprehensive_trajectories()` for detailed single-model analysis, Figure 3 uses individual visualization methods (`plot_heatmap()`, `plot_violin()`, `plot_ridge()`, `plot_phase_portrait()`) for each model type, and Figure 4 uses custom copula analysis visualization. Complete working code for all figures is provided in Section 12 (Complete Code Listings).
