# Complete Code Listings

This section contains all code examples and implementation details referenced throughout the paper. The code is organized by section and subsection for easy reference.

## Implementation Code

### Software Architecture

**Class Hierarchy**:

```python
StochasticProcess (ABC)
|-- OrnsteinUhlenbeckJump
|-- GeometricJumpDiffusion
|-- CompoundPoisson
|-- FractionalBrownianMotion
|-- CoxIngersollRoss
+-- LevyProcess

Analyzer (ABC)
|-- TimeSeriesAnalyzer
|-- MultivariateAnalyzer
|-- BayesianAnalyzer
|-- NetworkAnalyzer
+-- CausalInference
```

### Algorithmic Implementation

#### Stochastic Process Simulation

**Euler-Maruyama Scheme** for SDEs:

```python
def euler_maruyama(x0, t, drift, diffusion, dt):
    n_steps = len(t) - 1
    x = np.zeros(len(t))
    x[0] = x0

    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        x[i+1] = x[i] + drift(x[i], t[i])*dt + diffusion(x[i], t[i])*dW

    return x
```

**Jump Component**: Compound Poisson process

```python
def simulate_jumps(t, jump_intensity, jump_dist):
    dt = np.diff(t)
    jumps = np.zeros(len(t))

    for i, dti in enumerate(dt):
        n_jumps = np.random.poisson(jump_intensity * dti)
        if n_jumps > 0:
            jumps[i+1] = np.sum(jump_dist.rvs(n_jumps))

    return jumps
```

#### Parameter Estimation

**Maximum Likelihood via Numerical Optimization**:

```python
def estimate_parameters(data, dt, model_class):
    def negative_log_likelihood(params):
        model = model_class(params)
        return -model.log_likelihood(data, dt)

    result = minimize(
        negative_log_likelihood,
        x0=initial_guess,
        method='L-BFGS-B',
        bounds=param_bounds
    )

    return result.x
```

**Moment Matching**:

```python
def moment_matching(data, dt):
    # Empirical moments
    mean_x = np.mean(data)
    var_x = np.var(data)
    autocorr = np.corrcoef(data[:-1], data[1:])[0,1]

    # Match to theoretical moments
    theta = mean_x
    kappa = -np.log(autocorr) / dt
    sigma = np.sqrt(2 * kappa * var_x)

    return ModelParameters(
        equilibrium=theta,
        reversion_speed=kappa,
        diffusion=sigma
    )
```

#### Wavelet Transform Implementation

```python
def continuous_wavelet_transform(signal, scales, wavelet='morl'):
    """
    Compute CWT using PyWavelets.
    """
    import pywt

    coefficients, frequencies = pywt.cwt(
        signal,
        scales,
        wavelet
    )

    power = np.abs(coefficients) ** 2

    return {
        'coefficients': coefficients,
        'frequencies': frequencies,
        'power': power,
        'dominant_scale': scales[np.argmax(np.mean(power, axis=1))]
    }
```

#### Copula Fitting

```python
def fit_copula(data1, data2, copula_type='gaussian'):
    """
    Fit copula to bivariate data.
    """
    # Transform to uniform margins
    u1 = rankdata(data1) / (len(data1) + 1)
    u2 = rankdata(data2) / (len(data2) + 1)

    if copula_type == 'gaussian':
        # Gaussian copula parameter
        z1 = norm.ppf(u1)
        z2 = norm.ppf(u2)
        rho = np.corrcoef(z1, z2)[0, 1]
        return {'rho': rho}

    elif copula_type == 'clayton':
        # Clayton copula via Kendall's tau
        tau = stats.kendalltau(data1, data2)[0]
        theta = 2 * tau / (1 - tau)
        return {'theta': theta}
```

### Performance Optimization

#### Vectorization

Critical loops are vectorized using NumPy:

```python
# Scalar version (slow)
for i in range(n):
    result[i] = func(x[i])

# Vectorized version (fast)
result = func(x)
```

#### Performance Optimization Strategies

The framework leverages NumPy's optimized operations:

```python
# Vectorized operations for efficiency
def vectorized_trajectory_update(x, drift_fn, diffusion_fn, dt, dW):
    """Vectorized trajectory update."""
    drift = drift_fn(x)
    diffusion = diffusion_fn(x) * dW
    return x + drift * dt + diffusion

# Example: Multiple trajectories updated simultaneously
trajectories[:, i+1] = vectorized_trajectory_update(
    trajectories[:, i], 
    drift_fn, 
    diffusion_fn, 
    dt, 
    dW[:, i]
)
```

The architecture supports optional JIT compilation (via Numba) and parallel processing (via multiprocessing) for performance-critical applications when needed.

#### Memory Efficiency

Large datasets use chunked processing:

```python
def process_large_dataset(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        results.append(process_chunk(chunk))
    return concatenate_results(results)
```

### Testing Framework

#### Unit Tests

Each component has comprehensive unit tests:

```python
class TestFractionalBrownianMotion:
    def test_hurst_parameter(self):
        """Test Hurst parameter estimation."""
        fbm = FractionalBrownianMotion(params, hurst=0.7)
        t = np.linspace(0, 10, 100)
        paths = fbm.simulate(10.0, t, n_paths=50)

        assert paths.shape == (50, 100)
        assert np.all(np.isfinite(paths))
```

#### Integration Tests

Test component interactions:

```python
def test_full_pipeline():
    """Test complete analysis pipeline."""
    # Load data
    data_core = DataCore.load_from_csv(data_file)

    # Fit model
    model = JumpRope.fit(data_core, model_type='fractional-brownian')

    # Analyze
    analyzer = LaserPlaneAnalyzer(model)
    results = analyzer.analyze_cross_section(3.0)

    # Visualize
    visualizer = TrajectoryVisualizer()
    fig = visualizer.plot_trajectories(model)

    assert results is not None
    assert fig is not None
```

#### Validation Tests

Compare against known solutions:

```python
def test_ornstein_uhlenbeck_stationary():
    """Validate against analytical stationary distribution."""
    ou = OrnsteinUhlenbeckJump(params)

    # Simulate long trajectory
    t = np.linspace(0, 1000, 100000)
    x = ou.simulate(x0=0, t=t, n_paths=1)[0]

    # Check stationary moments
    theoretical_mean = params.theta
    theoretical_var = params.sigma**2 / (2*params.kappa)

    assert np.abs(np.mean(x[-10000:]) - theoretical_mean) < 0.1
    assert np.abs(np.var(x[-10000:]) - theoretical_var) < 0.2
```

### Documentation System

#### Docstring Format

Google-style docstrings throughout:

```python
def fit(self, data_core, model_type='jump-diffusion', **kwargs):
    """
    Fit stochastic process model to data.

    Parameters:
        data_core: DataCore instance with training data
        model_type: Type of stochastic process
        **kwargs: Additional model parameters

    Returns:
        Fitted JumpRope instance

    Examples:
        >>> model = JumpRope.fit(data, model_type='fractional-brownian')
        >>> trajectories = model.generate_trajectories(100)
    """
```

#### Sphinx Documentation

Complete API documentation generated via Sphinx:

```rst
.. automodule:: evojump.jumprope
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Tutorials and Examples

Comprehensive examples for all features:

```python
"""
Example: Advanced Stochastic Process Modeling

This example demonstrates the use of fractional Brownian
motion for modeling developmental trajectories with
long-range dependence.
"""

import evojump as ej

# Load data
data = ej.DataCore.load_from_csv('developmental_data.csv')

# Fit fBM model
model = ej.JumpRope.fit(data, model_type='fractional-brownian', hurst=0.7)

# Generate predictions
trajectories = model.generate_trajectories(n_samples=100, x0=10.0)

# Visualize
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_heatmap(model, output_dir='outputs/figures/')
```

### Visualization Framework

#### Implementation Details

```python
# Trajectory density heatmap
visualizer.plot_heatmap(model, time_resolution=50, phenotype_resolution=50)

# Violin plots at specific timepoints
visualizer.plot_violin(model, time_points=[1, 3, 5, 7, 9])

# Ridge plot for distribution evolution
visualizer.plot_ridge(model, n_distributions=10)

# Phase portrait analysis
visualizer.plot_phase_portrait(model, derivative_method='finite_difference')
```

Each visualization method supports both static (matplotlib) and interactive (Plotly) output modes, enabling publication-quality graphics and exploratory analysis.

### Package Management with UV

#### Project Configuration

```toml
[project]
name = "evojump"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "plotly>=5.0.0",
    "scikit-learn>=1.0.0",
    "PyWavelets>=1.3.0",
    "networkx>=2.6.0",
    "statsmodels>=0.13.0",
    "seaborn>=0.11.0"
]
```

#### Development Workflow

```bash
# Create virtual environment with UV
uv venv

# Sync all dependencies from pyproject.toml
uv sync

# Install in development mode
uv add -e .

# Run tests
uv run pytest

# Build documentation
uv run sphinx-build docs docs/_build
```

#### Reproducible Environments

```bash
# UV automatically generates uv.lock file
uv sync

# Install from lock file in new environment
uv sync --frozen
```

## Figure Generation Code

### Figure Generation Code Snippets

The following code snippets illustrate the core commands used to generate the figures. Full reproduction code is available in the EvoJump repository.

#### Comprehensive Model Comparison (Figure 1)

```python
import evojump as ej
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data for different stochastic processes
def create_synthetic_data(seed=42):
    # Generate synthetic developmental trajectories
    n_individuals, n_timepoints = 100, 100
    time_points = np.linspace(0, 10, n_timepoints)

    trajectories = []
    for i in range(n_individuals):
        # Base pattern with individual variation
        base = 10 + 3 * np.sin(time_points * 0.5) + time_points * 0.3
        noise = np.random.normal(0, 0.5, len(time_points))
        trajectory = base + noise
        trajectories.append(trajectory)

    return np.array(trajectories), time_points

# Generate data for each model type
fbm_trajectories, time_points = create_synthetic_data(seed=42)
cir_trajectories, _ = create_synthetic_data(seed=43)
jump_trajectories, _ = create_synthetic_data(seed=44)

# Create DataCore objects
fbm_data = ej.TimeSeriesData(fbm_trajectories, time_points, ['phenotype'])
cir_data = ej.TimeSeriesData(cir_trajectories, time_points, ['phenotype'])
jump_data = ej.TimeSeriesData(jump_trajectories, time_points, ['phenotype'])

# Fit stochastic models
fbm_model = ej.JumpRope.fit([fbm_data], model_type='fractional-brownian', hurst=0.7)
cir_model = ej.JumpRope.fit([cir_data], model_type='cox-ingersoll-ross')
jump_model = ej.JumpRope.fit([jump_data], model_type='jump-diffusion')

# Generate comprehensive model comparison (9 panels)
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_model_comparison(
    [fbm_model, cir_model, jump_model],
    model_names=['fBM', 'CIR', 'Jump-Diffusion'],
    output_path='figures/figure_1_comparison.png'
)
```

#### Comprehensive Trajectory Analysis (Figure 2)

```python
import evojump as ej
import numpy as np
import matplotlib.pyplot as plt

# Generate comprehensive 9-panel trajectory analysis
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_comprehensive_trajectories(
    fbm_model,
    output_path='figures/figure_2_comprehensive.png'
)

#### Individual Model Visualizations (Figure 3)

```python
import evojump as ej
# ... (load data_core and fit fbm_model as above) ...
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_heatmap(
    fbm_model,
    time_resolution=50,
    phenotype_resolution=50,
    output_path='figures/figure_2_heatmap.png'
)
```

#### Violin Plots (Figure 3)

```python
import evojump as ej
# ... (load data_core and fit cir_model as above) ...
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_violin(
    cir_model,
    time_points=[1, 3, 5, 7, 9],
    output_path='figures/figure_3_violin.png'
)
```

#### Ridge Plot (Figure 4)

```python
import evojump as ej
# ... (load data_core and fit levy_model as above) ...
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_ridge(
    levy_model,
    n_distributions=10,
    output_path='figures/figure_4_ridge.png'
)
```

#### Phase Portrait (Figure 5)

```python
import evojump as ej
# ... (load data_core and fit fbm_model as above) ...
visualizer = ej.TrajectoryVisualizer()
visualizer.plot_phase_portrait(
    fbm_model,
    derivative_method='finite_difference',
    output_path='figures/figure_5_phase_portrait.png'
)
```

#### Copula Analysis (Figure 7)

```python
import evojump as ej
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, kendalltau

def generate_copula_analysis(model, output_path):
    """Generate copula analysis showing trait dependence."""
    trajectories = model.trajectories

    # Select two time points for bivariate analysis
    time_indices = [len(model.time_points)//3, 2*len(model.time_points)//3]
    data_t1 = trajectories[:, time_indices[0]]
    data_t2 = trajectories[:, time_indices[1]]

    # Transform to uniform [0,1] scale using ranks
    u = rankdata(data_t1) / (len(data_t1) + 1)
    v = rankdata(data_t2) / (len(data_t2) + 1)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(u, v, alpha=0.6, s=50, c=range(len(u)), cmap='viridis')

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, linewidth=2, label='Perfect Dependence')

    # Calculate and display Kendall's tau
    tau, p_value = kendalltau(data_t1, data_t2)
    ax.text(0.05, 0.95, f"Kendall's $\\tau$ = {tau:.3f}\\n(p = {p_value:.3f})",
           transform=ax.transAxes, fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'Phenotype at t = {model.time_points[time_indices[0]]:.2f} (Rank)')
    ax.set_ylabel(f'Phenotype at t = {model.time_points[time_indices[1]]:.2f} (Rank)')
    ax.set_title('Copula Analysis: Temporal Dependence Structure')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Individual Index')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate copula figure
generate_copula_analysis(fbm_model, 'figures/figure_4_copula.png')
```

### Technical Details

- **Data Generation**: Synthetic developmental trajectories for 100 individuals over 100 timepoints.
- **Model Fitting**: Maximum likelihood estimation for all stochastic processes.
- **Visualization Libraries**: Matplotlib for static plots, Plotly for interactive versions.
- **Image Quality**: 300 DPI PNG format for publication.
- **Color Schemes**: Colorblind-friendly palettes used throughout.

### Software Requirements

- Python 3.8 or higher
- NumPy 1.21.0 or higher
- SciPy 1.7.0 or higher
- Matplotlib 3.5.0 or higher
- pandas 1.3.0 or higher
- EvoJump 0.1.0 or higher

### Installation

```bash
# Using UV
uv add evojump
```

## Drosophila Case Study Code

### Population Configuration

```python
# Initialize Drosophila population for selective sweep analysis
population_config = DrosophilaPopulation(
    population_size=100,
    generations=15,
    initial_red_eyed_proportion=0.1,
    advantageous_trait_fitness=1.2,  # 20% fitness advantage
    selection_coefficient=0.1
)
```

### Selection Simulation

```python
def _simulate_selection(self, current_red_eyed: int) -> int:
    """Simulate one generation of selection and reproduction."""
    current_freq = current_red_eyed / self.config.population_size

    # Selection differential
    mean_fitness = (current_freq * self.config.advantageous_trait_fitness +
                   (1 - current_freq) * 1.0)

    # New frequency after selection
    new_freq = (current_freq * self.config.advantageous_trait_fitness) / mean_fitness

    # Add genetic drift
    drift = np.random.normal(0, 0.01)
    new_freq = np.clip(new_freq + drift, 0.01, 0.99)

    return int(new_freq * self.config.population_size)
```

### Cross-Sectional Analysis

```python
# Analyze phenotypic distributions at key generations
analyzer = LaserPlaneAnalyzer(model)
stages = [2, 5, 8]  # Key generations

for stage in stages:
    result = analyzer.analyze_cross_section(time_point=float(stage))
    print(f"Stage {stage}: mean = {result.moments['mean']:.2f}, "
          f"std = {result.moments['std']:.2f}")
```

### Evolutionary Pattern Analysis

```python
# Population-level evolutionary pattern analysis
sampler = EvolutionSampler(population_data)
evolution_analysis = sampler.analyze_evolutionary_patterns()

pop_stats = evolution_analysis['population_statistics']
genetic_params = evolution_analysis['genetic_parameters']

print(f"Effective population size: {pop_stats.effective_population_size:.0f}")
print(f"Mean heritability: {np.mean(list(pop_stats.heritability_estimates.values())):.3f}")
```

### Network Analysis

```python
# Correlation network analysis for hitchhiking detection
network_results = analytics.network_analysis(correlation_threshold=0.6)
```

### Bayesian Analysis

```python
# Bayesian uncertainty quantification for evolutionary parameters
bayesian_results = analytics.bayesian_analysis('phenotype', 'fitness')
print(f"95% credible interval: {bayesian_results.credible_intervals['95%']}")
```
