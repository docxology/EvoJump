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
        copula_type='gaussian'  # Options: 'gaussian', 'clayton', 'frank', 'student'
    )
    
    print(f"Kendall's tau: {result['kendall_tau']}")
    print(f"Tail dependence: {result['upper_tail_dependence']}")

**Returns:**

- ``copula_type``: The fitted copula family
- ``copula_parameter``: Estimated copula parameter
- ``kendall_tau`` / ``kendall_tau_pvalue``: Kendall's tau and its p-value
- ``spearman_rho`` / ``spearman_rho_pvalue``: Spearman's rank correlation and its p-value
- ``upper_tail_dependence`` / ``lower_tail_dependence``: Empirical tail dependence coefficients
- ``degrees_of_freedom``: Present for the ``'student'`` copula only (method-of-moments estimate on excess kurtosis)
- ``dependence_class``: ``'positive'``, ``'negative'``, or ``'independent'``

**Copula families and estimation details:**

- ``'gaussian'``: parameter is the correlation of the inverse-normal-transformed ranks
- ``'student'``: rho from Kendall's tau via ``tau = (2/pi) arcsin(rho)``;
  degrees of freedom by the method of moments on excess kurtosis
- ``'clayton'``: method of moments ``2*tau/(1-tau)``; requires positive
  dependence (raises ``ValueError`` for ``tau <= 0``)
- ``'frank'``: **exact** inversion of the tau relation
  ``tau = 1 - 4/theta * (1 - Debye1(theta))`` (sign-flipped for negative tau),
  not an approximation

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

Spectral Analysis
~~~~~~~~~~~~~~~~~

Frequency-domain analysis using Welch's method, with optional
magnitude-squared coherence between two signals.

**Usage:**

.. code-block:: python

    result = analytics.spectral_analysis(
        'phenotype',
        sampling_frequency=1.0,
        coherence_column='phenotype2'  # optional second signal
    )

    print(f"Dominant frequencies: {result.dominant_frequencies}")
    print(f"Spectral entropy: {result.spectral_entropy:.3f}")

**Parameters:**

- ``signal_column`` (str): Column with signal data
- ``sampling_frequency`` (float): Sampling frequency (default ``1.0``)
- ``coherence_column`` (str, optional): Second column; when given, the
  magnitude-squared coherence between the two signals is stored in
  ``coherence_matrix`` as an ``(n_frequencies, 2)`` array of
  ``[frequency, coherence]``. Without it ``coherence_matrix`` is empty
  (coherence is a two-signal quantity).

**Returns** (``SpectralResult``):

- ``power_spectrum``: Welch power spectral density
- ``frequency_peaks``: Frequencies of spectral peaks (power above its 75th percentile)
- ``spectral_entropy``: Spectral entropy of the normalized power
- ``dominant_frequencies``: Peak frequencies
- ``coherence_matrix``: Coherence array (empty when no ``coherence_column``)

Spatial Analysis (Moran's I)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantifies spatial autocorrelation with Moran's I.

**Usage:**

.. code-block:: python

    result = analytics.spatial_analysis('phenotype')

    print(f"Moran's I: {result['morans_i']:.3f}")
    print(f"Autocorrelation: {result['spatial_autocorrelation']}")
    print(f"Weights: {result['weights_kind']}")

    # Custom n x n weights (rows/columns match the non-NaN observations)
    n = len(analytics.data)
    W = np.ones((n, n)) - np.eye(n)
    result = analytics.spatial_analysis('phenotype', spatial_weights=W)

**Parameters:**

- ``value_column`` (str): Column with values to analyze
- ``spatial_weights`` (ndarray, optional): n x n spatial weights matrix
  W. When given, Moran's I is ``I = (n / S0) * (z' W z) / (z' z)`` with
  z the centered values and ``S0 = sum(W)``; the shape must match the
  number of non-NaN observations of the column.

**Returns:**

- ``morans_i``: Moran's I statistic (``NaN`` when undefined)
- ``spatial_autocorrelation``: ``'Positive'`` (I > 0.1), ``'Negative'``
  (I < -0.1), or ``'None'``
- ``weights_kind``: ``'supplied'`` when an explicit weights matrix is
  passed; ``'linear_adjacency'`` for the default (``w_ij = 1`` for
  adjacent observations in sequence order)

Robust Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real M-estimators — not placeholders: Huber and Tukey biweight location
via IRLS, and the Rousseeuw-Croux Sn scale estimator.

**Usage:**

.. code-block:: python

    result = analytics.robust_statistical_analysis('phenotype')

    print(f"Huber estimate: {result['location_estimates']['huber_estimator']:.3f}")
    print(f"Tukey biweight: {result['location_estimates']['tukey_biweight']:.3f}")
    print(f"Sn scale: {result['scale_estimates']['sn_scale']:.3f}")

**Returns:**

- ``location_estimates``: ``median``, ``trimmed_mean``,
  ``huber_estimator``, ``tukey_biweight``
- ``scale_estimates``: ``mad``, ``mad_normalized`` (consistent with the
  normal distribution), ``iqr``, ``sn_scale``
- ``robust_location_preferred``: The trimmed mean
- ``robust_scale_preferred``: The MAD normalized to normal consistency

Canonical Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Relates two blocks of variables via CCA
(``MultivariateAnalyzer.canonical_correlation_analysis``), computing
canonical correlations from the generalized eigenproblem
``cov11^{-1/2} cov12 cov22^{-1} cov21 cov11^{-1/2}``.

**Usage:**

.. code-block:: python

    from evojump.analytics_engine import MultivariateAnalyzer

    mv = MultivariateAnalyzer(data_core.get_aggregated_data())
    cca = mv.canonical_correlation_analysis(block1, block2)

    print(f"Canonical correlations: {cca['canonical_correlations']}")

**Returns:**

- ``canonical_correlations``: Sorted canonical correlations
- ``canonical_variables_1``: X-side canonical coefficients
- ``canonical_variables_2``: Y-side canonical coefficients (scaled to unit canonical variates)
- ``eigenvalues``: Squared canonical correlations
- ``scaler1_mean`` / ``scaler1_scale`` / ``scaler2_mean`` / ``scaler2_scale``: Standardization used


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
- ``switch_timepoints``: Indices of the observations where the regime changed

**Applications:**

- Identifying developmental phases
- Detecting environmental regime shifts
- Modeling punctuated equilibrium


Cross-Section Distribution Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LaserPlaneAnalyzer.compare_distributions`` compares the model's
reference cross-section against one or more condition-specific samples
at a time point and returns a fully populated ``DistributionComparison``
(``test_statistics``, ``p_values``, ``effect_sizes`` and
``significant_differences`` are filled in, not left empty).

**Usage:**

.. code-block:: python

    import numpy as np

    model.generate_trajectories(n_samples=100, x0=10.0, seed=7)
    analyzer = ej.LaserPlaneAnalyzer(model)
    comparison = analyzer.compare_distributions(
        time_point=5.0,
        condition_data={
            'control': np.asarray([...]),
            'treated': np.asarray([...]),
        },
        test='auto',                     # or 'ks', 'anderson', 'cramer', 'mann_whitney', 't_test'
        rng=np.random.default_rng(42)    # seeds permutation p-values
    )

    print(comparison.test_statistics)
    print(comparison.p_values)
    print(comparison.effect_sizes)       # Cohen's d per condition
    print(comparison.significant_differences)

The module-level ``DistributionComparer.compare_distributions(data1,
data2, test='auto', rng=None)`` performs the same tests directly on two
samples; Anderson-Darling and Cramer-von Mises p-values are permutation
based and inherit reproducibility from ``rng``.

Evolutionary Genetic Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``EvolutionSampler.analyze_evolutionary_patterns`` reports real
quantities or explicit NaN — never fabricated placeholders:

- ``genetic_parameters['available']``: ``True`` only when a
  ``parent``/``offspring`` pedigree exists and at least one time point
  has replicated observations; ``additive_variance``,
  ``environmental_variance`` and ``narrow_sense_heritability`` are NaN
  when the flag is ``False``. ``dominance_variance``,
  ``epistatic_variance`` and ``broad_sense_heritability`` are not
  identifiable from phenotypes alone and are always NaN.
- ``selection_analysis['selection_differential']`` and
  ``selection_analysis['selection_response']``: **per-trait
  dictionaries** (trait name to value), not scalars.
- ``phylogenetic_signal``: computed via Moran's I only when a distance
  matrix has been supplied and each trait vector aligns with the matrix
  rows; otherwise the entry is left empty.

**Usage:**

.. code-block:: python

    sampler = ej.EvolutionSampler(population_data)
    results = sampler.analyze_evolutionary_patterns()
    genetics = results['genetic_parameters']
    if genetics.get('available'):
        print(f"Narrow-sense heritability: {genetics['narrow_sense_heritability']:.3f}")
    for trait, differential in results['selection_analysis']['selection_differential'].items():
        print(f"{trait}: S = {differential:.3f}")

Comprehensive Report Column Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AnalyticsEngine.comprehensive_analysis_report`` accepts explicit
column pairs for the sections that need a hypothesized direction:

.. code-block:: python

    report = analytics.comprehensive_analysis_report(
        bayesian_columns=('phenotype1', 'phenotype2'),   # (x, y) regression pair
        causal_columns=('temperature', 'growth_rate')    # (cause, effect) Granger pair
    )

Without an explicit pair the corresponding section is reported as
``{'not_analyzed': ...}`` — Bayesian regression and Granger causality on
arbitrary first-two columns produced uninterpretable results, so they
are skipped rather than fabricated.

Reproducibility (Seeding)
~~~~~~~~~~~~~~~~~~~~~~~~~

All stochastic entry points accept an explicit generator or seed:

.. code-block:: python

    import numpy as np

    model = ej.JumpRope.fit(data_core, model_type='jump-diffusion',
                            seed=42)                    # or rng=np.random.default_rng(42)
    trajectories = model.generate_trajectories(n_samples=100, x0=10.0, seed=7)

    sampler = ej.EvolutionSampler(population_data)

    sampler.seed(42)                                    # seeds the sampler's generator

    analyzer = ej.LaserPlaneAnalyzer(model)
    cross = analyzer.analyze_cross_section(5.0, n_bootstrap=1000,
                                          rng=np.random.default_rng(3))
    bayes = analytics.bayesian_analysis('phenotype1', 'phenotype2', seed=5)
    comparison = analyzer.compare_distributions(
        5.0, {'treated': treated}, rng=np.random.default_rng(11))

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
