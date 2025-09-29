# EvoJump: A Comprehensive Framework for Evolutionary Ontogenetic Analysis

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/evojump/badge/?version=latest)](https://evojump.readthedocs.io/)

EvoJump represents a groundbreaking analytical framework that conceptualizes evolutionary and developmental biology through a novel "cross-sectional laser" metaphor. This system treats ontogenetic development as a temporal progression where a "jumprope-like" distribution sweeps across a fixed analytical plane (the laser), generating dynamic cross-sectional views of phenotypic distributions throughout an organism's developmental timeline.

## 🚀 Features

### Core Analytical Capabilities
- **Ontogenetic Trajectory Analysis**: Characterize complete developmental pathways from embryogenesis through adulthood
- **Cross-Sectional Distribution Analysis**: Advanced statistical methods for phenotypic distributions at specific timepoints
- **Jump Detection and Characterization**: Identify and quantify discrete developmental transitions
- **Evolutionary Pattern Recognition**: Machine learning approaches for identifying evolutionary patterns
- **Predictive Modeling**: Advanced modeling for predicting developmental outcomes

### Visualization and Interaction
- **Interactive Developmental Landscapes**: 3D visualization of phenotypic evolution over time
- **Temporal Animation Systems**: Animated visualization of developmental processes
- **Advanced Visualizations**: Heatmaps, violin plots, ridge plots, phase portraits
- **Comparative Visualization Tools**: Multi-condition, multi-genotype trajectory comparison
- **Real-Time Analysis Interface**: Interactive parameter adjustment with immediate feedback
- **Publication-Quality Graphics**: High-resolution exports for scientific publications

### Statistical and Analytical Methods
- **Time Series Analysis**: Trend analysis, seasonality detection, change point analysis, ARIMA modeling
- **Multivariate Analysis**: PCA, CCA, cluster analysis, TSNE, UMAP for complex phenotypic datasets
- **Stochastic Process Modeling**: Jump-diffusion, Lévy processes, Fractional Brownian Motion, Cox-Ingersoll-Ross
- **Advanced Analytics**: Wavelet analysis, copula methods, extreme value theory, regime switching detection
- **Machine Learning Integration**: Deep learning, ensemble methods, Gaussian processes, automated ML pipelines
- **Bayesian Methods**: Bayesian inference, posterior sampling, credible intervals
- **Network Analysis**: Graph theory, community detection, centrality measures

## 📦 Installation

### Requirements
- Python 3.8 or higher
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

### Quick Install
```bash
pip install evojump
```

### Development Install
```bash
git clone https://github.com/evojump/evojump.git
cd evojump
pip install -e .
```

## 🏁 Quick Start

```python
import evojump as ej
import pandas as pd
import numpy as np

# Load developmental data
data = pd.DataFrame({
    'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19],
    'phenotype2': [20, 22, 24, 26, 28, 21, 23, 25, 27, 29]
})

# Create DataCore instance
data_core = ej.DataCore.load_from_csv("data.csv", time_column='time')

# Fit jump-diffusion model
model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')

# Generate trajectories
trajectories = model.generate_trajectories(n_samples=100)

# Analyze cross-sections
analyzer = ej.LaserPlaneAnalyzer(model)
results = analyzer.analyze_cross_section(time_point=3.0)

# Create visualizations
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_trajectories(model)
visualizer.plot_cross_sections(model)
```

## 🎯 Advanced Features

### Advanced Stochastic Process Models

```python
# Fractional Brownian Motion (long-range dependence)
fbm_model = ej.JumpRope.fit(data_core, model_type='fractional-brownian', hurst=0.7)

# Cox-Ingersoll-Ross (mean-reverting, non-negative)
cir_model = ej.JumpRope.fit(data_core, model_type='cir', equilibrium=15.0)

# Levy Process (heavy-tailed distributions)
levy_model = ej.JumpRope.fit(data_core, model_type='levy', levy_alpha=1.5)
```

### Advanced Visualizations

```python
visualizer = ej.TrajectoryVisualizer()

# Trajectory density heatmap
visualizer.plot_heatmap(model, time_resolution=50, phenotype_resolution=50)

# Violin plots showing distribution evolution
visualizer.plot_violin(model, time_points=[1, 3, 5, 7, 9])

# Ridge plot (joyplot) for temporal distributions
visualizer.plot_ridge(model, n_distributions=10)

# Phase portrait (phenotype vs. rate of change)
visualizer.plot_phase_portrait(model, derivative_method='finite_difference')
```

### Advanced Statistical Methods

```python
analytics = ej.AnalyticsEngine(data, time_column='time')

# Wavelet analysis for time-frequency patterns
wavelet_result = analytics.wavelet_analysis('phenotype', wavelet='morl')

# Copula analysis for dependence structure
copula_result = analytics.copula_analysis('phenotype1', 'phenotype2')

# Extreme value analysis
extreme_result = analytics.extreme_value_analysis('phenotype')

# Regime switching detection
regime_result = analytics.regime_switching_analysis('phenotype', n_regimes=3)
```

## 📊 Command Line Interface

EvoJump provides a comprehensive command-line interface for batch processing and automation:

```bash
# Analyze developmental trajectories
evojump-cli analyze data.csv --output results/

# Fit stochastic process model
evojump-cli fit data.csv --model-type jump-diffusion --output model.pkl

# Visualize results
evojump-cli visualize model.pkl --plot-type trajectories --output plots/

# Perform evolutionary sampling
evojump-cli sample population.csv --samples 1000 --output samples.csv
```

## 🔬 Applications and Use Cases

### Developmental Biology Research
- **Ontogenetic Trajectory Analysis**: Characterize complete developmental pathways
- **Gene Expression Dynamics**: Temporal gene expression pattern analysis
- **Environmental Developmental Plasticity**: Environmental effects on development

### Evolutionary Biology Applications
- **Phylogenetic Developmental Analysis**: Comparative analysis across species
- **Quantitative Genetics**: Genetic contributions to developmental variation
- **Evolutionary Constraint Analysis**: Constraints on developmental pathways

### Agricultural and Applied Biology
- **Crop Development Modeling**: Agricultural optimization and yield prediction
- **Breeding Program Optimization**: Selection strategies based on developmental analysis
- **Pest and Disease Management**: Pest developmental responses to conditions

### Medical and Health Applications
- **Disease Progression Modeling**: Disease development as developmental processes
- **Therapeutic Development**: Drug effects on developmental processes
- **Biomarker Discovery**: Early developmental signatures of later outcomes

## 🏗️ Architecture

### Core Modules

#### DataCore Module
- Data ingestion, validation, and preprocessing
- Support for multiple data formats
- Robust data structures for longitudinal datasets
- Comprehensive metadata management

#### JumpRope Engine
- Jump-diffusion modeling for developmental trajectories
- Multiple stochastic process models (Ornstein-Uhlenbeck, geometric jump-diffusion, compound Poisson)
- Parameter estimation and model fitting
- Trajectory generation and simulation

#### LaserPlane Analyzer
- Cross-sectional analysis algorithms
- Distribution fitting and comparison
- Moment analysis and quantile estimation
- Goodness of fit assessment

#### Trajectory Visualizer
- Advanced visualization system
- Interactive plotting capabilities
- Animation sequences
- Comparative visualization tools

#### Evolution Sampler
- Population-level analysis
- Phylogenetic comparative methods
- Quantitative genetics approaches
- Population dynamics modeling

#### Analytics Engine
- Time series analysis
- Multivariate statistics
- Machine learning algorithms
- Predictive modeling

## 🧪 Testing

EvoJump follows test-driven development (TDD) with comprehensive test coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=evojump --cov-report=html

# Run specific test modules
pytest tests/test_datacore.py
pytest tests/test_jumprope.py
pytest tests/test_laserplane.py
```

## 📚 Documentation

- **User Guide**: Comprehensive tutorials and examples
- **API Reference**: Complete API documentation
- **Examples**: Working code examples for common use cases
- **Contributing Guide**: Guidelines for contributors

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/evojump/evojump.git
cd evojump
pip install -e ".[dev]"
```

### Key Development Principles
- Follow test-driven development (TDD)
- Maintain high test coverage (>95%)
- Use real methods and data in tests (no mocks)
- Write comprehensive documentation
- Follow scientific computing best practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Citation

If you use EvoJump in your research, please cite:

```bibtex
@software{evojump2024,
  title={EvoJump: A Comprehensive Framework for Evolutionary Ontogenetic Analysis},
  author={EvoJump Development Team},
  year={2024},
  url={https://github.com/evojump/evojump}
}
```

## 🔗 Links

- **Homepage**: https://github.com/evojump/evojump
- **Documentation**: https://evojump.readthedocs.io/
- **Issues**: https://github.com/evojump/evojump/issues
- **Discussions**: https://github.com/evojump/evojump/discussions

## 🙏 Acknowledgments

EvoJump builds upon decades of research in developmental biology, evolutionary theory, and statistical modeling. We acknowledge the contributions of the scientific community and the foundational work in:

- Stochastic processes in biology
- Developmental systems theory
- Quantitative genetics
- Statistical modeling of biological systems
- Scientific Python ecosystem

---

*EvoJump: Illuminating the dynamics of evolutionary development*

