Advanced Methods
================

This document provides comprehensive documentation for the advanced methods added to EvoJump, including cutting-edge stochastic process models, visualization techniques, and statistical analysis methods.

Advanced Stochastic Process Models
-----------------------------------

EvoJump now supports multiple advanced stochastic process models beyond the standard Ornstein-Uhlenbeck and jump-diffusion processes.

Fractional Brownian Motion (fBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fractional Brownian Motion is a stochastic process that exhibits long-range dependence, making it suitable for modeling developmental processes with memory effects.

**Key Features:**

- **Hurst Parameter** (H): Controls the degree of long-range dependence
  
  - H = 0.5: Standard Brownian motion (no memory)
  - H > 0.5: Persistent motion (positive correlation over time)
  - H < 0.5: Anti-persistent motion (negative correlation over time)

- **Applications**: Modeling developmental trajectories with temporal autocorrelation, phenotypic canalization, evolutionary constraints

**Usage:**

.. code-block:: python

    import evojump as ej
    
    # Fit fractional Brownian motion model
    model = ej.JumpRope.fit(
        data_core,
        model_type='fractional-brownian',
        hurst=0.7  # Persistent motion
    )
    
    # Generate trajectories
    trajectories = model.generate_trajectories(n_samples=100, x0=10.0)

**Parameters:**

- ``hurst`` (float): Hurst parameter, typically in range [0.1, 0.9]
- ``drift`` (float): Deterministic drift component
- ``diffusion`` (float): Scale parameter for stochastic component

Cox-Ingersoll-Ross (CIR) Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CIR process is a mean-reverting stochastic process that ensures non-negative values, ideal for modeling traits that cannot be negative (e.g., size, concentration, counts).

**Key Features:**

- **Mean Reversion**: Trajectories tend toward an equilibrium level
- **Non-negativity**: Ensures all values remain positive
- **State-dependent Volatility**: Variance increases with the level of the process

**Usage:**

.. code-block:: python

    # Fit CIR process
    model = ej.JumpRope.fit(
        data_core,
        model_type='cir',
        equilibrium=15.0,        # Long-term mean level
        reversion_speed=0.5,     # Speed of mean reversion
        diffusion=1.0            # Volatility parameter
    )

**Parameters:**

- ``equilibrium`` (float): Long-term mean level (theta)
- ``reversion_speed`` (float): Speed of mean reversion (kappa)
- ``diffusion`` (float): Volatility parameter (sigma)

**Mathematical Form:**

.. math::

    dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}dW_t

Levy Process
~~~~~~~~~~~~

Levy processes use stable distributions with heavy tails, suitable for modeling developmental processes with extreme events or jumps.

**Key Features:**

- **Heavy-tailed Distributions**: Captures rare extreme events
- **Infinite Divisibility**: Suitable for hierarchical developmental processes
- **Flexible Skewness**: Can model asymmetric distributions

**Usage:**

.. code-block:: python

    # Fit Levy process
    model = ej.JumpRope.fit(
        data_core,
        model_type='levy',
        levy_alpha=1.5,  # Stability parameter (tail heaviness)
        levy_beta=0.0    # Skewness parameter
    )

**Parameters:**

- ``levy_alpha`` (float): Stability parameter in (0, 2]. Lower values = heavier tails
- ``levy_beta`` (float): Skewness parameter in [-1, 1]. 0 = symmetric
- ``drift`` (float): Location parameter
- ``diffusion`` (float): Scale parameter

Advanced Visualization Methods
-------------------------------

Trajectory Density Heatmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizes the density of trajectories across time and phenotype space, revealing patterns of convergence, divergence, and distributional shifts.

**Usage:**

.. code-block:: python

    visualizer = ej.TrajectoryVisualizer()
    
    fig = visualizer.plot_heatmap(
        model,
        time_resolution=50,
        phenotype_resolution=50,
        interactive=False
    )

**Applications:**

- Identifying developmental bottlenecks
- Detecting critical transitions
- Visualizing population structure over time

Violin Plots
~~~~~~~~~~~~

Shows the full distribution of phenotypes at multiple time points, combining box plots with kernel density estimation.

**Usage:**

.. code-block:: python

    fig = visualizer.plot_violin(
        model,
        time_points=[1.0, 3.0, 5.0, 7.0, 9.0],
        output_dir=Path("outputs/")
    )

**Applications:**

- Comparing distributions across developmental stages
- Detecting multimodality
- Assessing distributional changes

Ridge Plots (Joyplots)
~~~~~~~~~~~~~~~~~~~~~~~

Displays stacked distributions over time, providing an intuitive view of how phenotypic distributions evolve.

**Usage:**

.. code-block:: python

    fig = visualizer.plot_ridge(
        model,
        n_distributions=10,
        output_dir=Path("outputs/")
    )

**Applications:**

- Temporal evolution visualization
- Publication-quality distribution comparisons
- Developmental trajectory overviews

Phase Portraits
~~~~~~~~~~~~~~~

Plots phenotype values against their rate of change, revealing dynamic attractors and developmental trajectories in phase space.

**Usage:**

.. code-block:: python

    fig = visualizer.plot_phase_portrait(
        model,
        derivative_method='finite_difference',
        interactive=True
    )

**Applications:**

- Identifying developmental attractors
- Detecting limit cycles or chaotic behavior
- Understanding developmental dynamics

Advanced Statistical Methods
-----------------------------

Wavelet Analysis
~~~~~~~~~~~~~~~~

Time-frequency analysis to identify periodic patterns and localized events in developmental trajectories.

**Usage:**

.. code-block:: python

    analytics = ej.AnalyticsEngine(data)
    
    result = analytics.wavelet_analysis(
        'phenotype',
        wavelet='morl',  # Morlet wavelet
        scales=np.arange(1, 128)
    )
    
    print(f"Dominant scale: {result['dominant_scale']}")
    print(f"Number of events: {result['n_events']}")

**Returns:**

- ``coefficients``: Wavelet coefficients matrix
- ``scales``: Scale values used
- ``power_spectrum``: Power spectrum across scales and time
- ``dominant_scale``: Most prominent scale
- ``n_events``: Number of significant events detected

**Applications:**

- Detecting developmental oscillations
- Identifying critical periods
- Multi-scale temporal analysis

Copula Analysis
~~~~~~~~~~~~~~~

Analyzes dependence structure between variables using copulas, capturing non-linear dependencies beyond correlation.

**Usage:**

.. code-block:: python

    result = analytics.copula_analysis(
        'phenotype1',
        'phenotype2',
        copula_type='gaussian'  # Options: 'gaussian', 'clayton', 'frank'
    )
    
    print(f"Kendall's tau: {result['kendall_tau']}")
    print(f"Tail dependence: {result['upper_tail_dependence']}")

**Returns:**

- ``copula_parameter``: Estimated copula parameter
- ``kendall_tau``: Kendall's tau correlation coefficient
- ``spearman_rho``: Spearman's rank correlation
- ``upper_tail_dependence``: Upper tail dependence coefficient
- ``lower_tail_dependence``: Lower tail dependence coefficient

**Applications:**

- Modeling complex trait dependencies
- Assessing co-development patterns
- Risk analysis for extreme phenotypes

Extreme Value Analysis
~~~~~~~~~~~~~~~~~~~~~~

Characterizes extreme phenotypes using extreme value theory, estimating return levels and tail behavior.

**Usage:**

.. code-block:: python

    result = analytics.extreme_value_analysis(
        'phenotype',
        threshold=None,  # Auto-select threshold
        block_size=None  # Auto-select block size
    )
    
    print(f"100-year return level: {result['pot_method']['return_levels']['100_year']}")
    print(f"Tail index: {result['tail_index']}")

**Methods:**

1. **Peaks-Over-Threshold (POT)**: Fits Generalized Pareto Distribution to exceedances
2. **Block Maxima**: Fits Generalized Extreme Value (GEV) distribution to block maxima
3. **Hill Estimator**: Estimates tail index for heavy-tailed distributions

**Returns:**

- ``pot_method``: POT analysis results with return levels
- ``block_maxima_method``: GEV analysis results
- ``hill_estimator``: Tail index estimate
- ``tail_index``: Inverse of Hill estimator

**Applications:**

- Predicting extreme phenotypes
- Assessing evolutionary constraints
- Risk assessment for rare developmental outcomes

Regime Switching Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Identifies discrete regimes in time series data and estimates transition probabilities between regimes.

**Usage:**

.. code-block:: python

    result = analytics.regime_switching_analysis(
        'phenotype',
        n_regimes=3
    )
    
    print(f"Number of switches: {result['n_switches']}")
    for stat in result['regime_statistics']:
        print(f"Regime {stat['regime_id']}: mean={stat['mean']:.2f}")

**Returns:**

- ``n_regimes``: Number of regimes identified
- ``regime_labels``: Regime assignment for each time point
- ``regime_statistics``: Mean, variance, and duration for each regime
- ``transition_matrix``: Count matrix of regime transitions
- ``transition_probabilities``: Probability matrix of regime transitions
- ``n_switches``: Total number of regime switches

**Applications:**

- Identifying developmental phases
- Detecting environmental regime shifts
- Modeling punctuated equilibrium

Best Practices
--------------

Model Selection
~~~~~~~~~~~~~~~

1. **Standard Brownian Motion**: Use for simple, memoryless diffusion
2. **Fractional Brownian Motion**: Use when temporal autocorrelation is expected
3. **Cox-Ingersoll-Ross**: Use for non-negative traits with mean reversion
4. **Levy Process**: Use when extreme events are important

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

1. **Wavelet Analysis**: Best for data with > 100 time points
2. **Copula Analysis**: Requires at least 50 paired observations
3. **Extreme Value Analysis**: Needs sufficient extreme observations (> 10 exceedances)
4. **Regime Switching**: Works best with clear developmental phases

Visualization
~~~~~~~~~~~~~

1. **Heatmaps**: Ideal for large trajectory datasets (> 50 trajectories)
2. **Violin Plots**: Best for comparing 3-10 time points
3. **Ridge Plots**: Optimal for showing 5-15 temporal distributions
4. **Phase Portraits**: Most informative with smooth, well-sampled trajectories

Performance Considerations
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

- **Fractional Brownian Motion**: O(n²) for covariance computation
- **Wavelet Analysis**: O(n log n) with FFT-based methods
- **Copula Analysis**: O(n log n) for empirical CDFs
- **Regime Switching**: O(k × n × d) where k = n_regimes, d = window features

Optimization Tips
~~~~~~~~~~~~~~~~~

1. Use ``n_samples`` parameter to limit trajectory generation
2. Reduce resolution parameters for faster visualization
3. Use ``interactive=False`` for batch processing
4. Consider downsampling for very large datasets (> 10,000 points)

References
----------

**Fractional Brownian Motion:**

- Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications. *SIAM Review*, 10(4), 422-437.

**Cox-Ingersoll-Ross Process:**

- Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A theory of the term structure of interest rates. *Econometrica*, 385-407.

**Levy Processes:**

- Sato, K. I. (1999). *Lévy processes and infinitely divisible distributions*. Cambridge University Press.

**Extreme Value Theory:**

- Coles, S. (2001). *An introduction to statistical modeling of extreme values*. Springer.

**Wavelet Analysis:**

- Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78.

**Copula Theory:**

- Nelsen, R. B. (2006). *An introduction to copulas*. Springer.
