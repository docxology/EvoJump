API Reference
=============

This section provides comprehensive documentation for all EvoJump modules, classes, and functions.

Core Modules
------------

DataCore Module
~~~~~~~~~~~~~~~

.. automodule:: evojump.datacore
   :members:
   :undoc-members:
   :show-inheritance:

JumpRope Engine
~~~~~~~~~~~~~~~

.. automodule:: evojump.jumprope
   :members:
   :undoc-members:
   :show-inheritance:

LaserPlane Analyzer
~~~~~~~~~~~~~~~~~~~

.. automodule:: evojump.laserplane
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Visualizer
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evojump.trajectory_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Evolution Sampler
~~~~~~~~~~~~~~~~~

.. automodule:: evojump.evolution_sampler
   :members:
   :undoc-members:
   :show-inheritance:

Analytics Engine
~~~~~~~~~~~~~~~~

.. automodule:: evojump.analytics_engine
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evojump.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main Classes
------------

DataCore
~~~~~~~~

.. autoclass:: evojump.DataCore
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: load_from_csv
   .. automethod:: load_from_hdf5
   .. automethod:: preprocess_data
   .. automethod:: validate_data_quality
   .. automethod:: save_processed_data

TimeSeriesData
~~~~~~~~~~~~~~

.. autoclass:: evojump.TimeSeriesData
   :members:
   :undoc-members:
   :show-inheritance:

MetadataManager
~~~~~~~~~~~~~~~

.. autoclass:: evojump.MetadataManager
   :members:
   :undoc-members:
   :show-inheritance:

JumpRope
~~~~~~~~

.. autoclass:: evojump.JumpRope
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: fit
   .. automethod:: generate_trajectories
   .. automethod:: compute_cross_sections
   .. automethod:: estimate_jump_times
   .. automethod:: save
   .. automethod:: load

ModelParameters
~~~~~~~~~~~~~~~

.. autoclass:: evojump.ModelParameters
   :members:
   :undoc-members:
   :show-inheritance:

StochasticProcess
~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.jumprope.StochasticProcess
   :members:
   :undoc-members:
   :show-inheritance:

OrnsteinUhlenbeckJump
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.jumprope.OrnsteinUhlenbeckJump
   :members:
   :undoc-members:
   :show-inheritance:

GeometricJumpDiffusion
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.jumprope.GeometricJumpDiffusion
   :members:
   :undoc-members:
   :show-inheritance:

CompoundPoisson
~~~~~~~~~~~~~~~

.. autoclass:: evojump.jumprope.CompoundPoisson
   :members:
   :undoc-members:
   :show-inheritance:

FractionalBrownianMotion
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.jumprope.FractionalBrownianMotion
   :members:
   :undoc-members:
   :show-inheritance:

CoxIngersollRoss
~~~~~~~~~~~~~~~~

.. autoclass:: evojump.jumprope.CoxIngersollRoss
   :members:
   :undoc-members:
   :show-inheritance:

LevyProcess
~~~~~~~~~~~

.. autoclass:: evojump.jumprope.LevyProcess
   :members:
   :undoc-members:
   :show-inheritance:

LaserPlaneAnalyzer
~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.LaserPlaneAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: analyze_cross_section
   .. automethod:: compare_distributions
   .. automethod:: generate_summary_report

CrossSectionResult
~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.laserplane.CrossSectionResult
   :members:
   :undoc-members:
   :show-inheritance:

DistributionComparison
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.laserplane.DistributionComparison
   :members:
   :undoc-members:
   :show-inheritance:

TrajectoryVisualizer
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.TrajectoryVisualizer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: plot_trajectories
   .. automethod:: plot_cross_sections
   .. automethod:: plot_landscapes
   .. automethod:: plot_heatmap
   .. automethod:: plot_violin
   .. automethod:: plot_ridge
   .. automethod:: plot_phase_portrait
   .. automethod:: create_animation
   .. automethod:: create_animation

PlotConfig
~~~~~~~~~~

.. autoclass:: evojump.trajectory_visualizer.PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:

EvolutionSampler
~~~~~~~~~~~~~~~~

.. autoclass:: evojump.EvolutionSampler
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: sample
   .. automethod:: analyze_evolutionary_patterns
   .. automethod:: cluster_individuals

PopulationStatistics
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.evolution_sampler.PopulationStatistics
   :members:
   :undoc-members:
   :show-inheritance:

AnalyticsEngine
~~~~~~~~~~~~~~~

.. autoclass:: evojump.AnalyticsEngine
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: analyze_time_series
   .. automethod:: analyze_multivariate
   .. automethod:: predictive_modeling
   .. automethod:: bayesian_analysis
   .. automethod:: network_analysis
   .. automethod:: shortest_path_analysis
   .. automethod:: causal_inference
   .. automethod:: advanced_dimensionality_reduction
   .. automethod:: wavelet_analysis
   .. automethod:: copula_analysis
   .. automethod:: extreme_value_analysis
   .. automethod:: regime_switching_analysis
   .. automethod:: spectral_analysis
   .. automethod:: nonlinear_dynamics_analysis
   .. automethod:: information_theory_analysis
   .. automethod:: robust_statistical_analysis
   .. automethod:: spatial_analysis
   .. automethod:: comprehensive_analysis_report

TimeSeriesResult
~~~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.TimeSeriesResult
   :members:
   :undoc-members:
   :show-inheritance:

PredictiveModelResult
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.PredictiveModelResult
   :members:
   :undoc-members:
   :show-inheritance:

BayesianResult
~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.BayesianResult
   :members:
   :undoc-members:
   :show-inheritance:

NetworkResult
~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.NetworkResult
   :members:
   :undoc-members:
   :show-inheritance:

CausalResult
~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.CausalResult
   :members:
   :undoc-members:
   :show-inheritance:

DimensionalityResult
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.DimensionalityResult
   :members:
   :undoc-members:
   :show-inheritance:

SurvivalResult
~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.SurvivalResult
   :members:
   :undoc-members:
   :show-inheritance:

SpectralResult
~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.SpectralResult
   :members:
   :undoc-members:
   :show-inheritance:

NonlinearResult
~~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.NonlinearResult
   :members:
   :undoc-members:
   :show-inheritance:

InformationResult
~~~~~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.InformationResult
   :members:
   :undoc-members:
   :show-inheritance:

RobustResult
~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.RobustResult
   :members:
   :undoc-members:
   :show-inheritance:

SpatialResult
~~~~~~~~~~~~~

.. autoclass:: evojump.analytics_engine.SpatialResult
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

All data structures used throughout EvoJump are documented below.

Configuration Objects
~~~~~~~~~~~~~~~~~~~~~

PlotConfig
   Configuration object for visualization settings.

   .. code-block:: python

      config = PlotConfig(
          figsize=(12, 8),
          dpi=150,
          style='default',
          alpha=0.7,
          show_confidence_intervals=True
      )

ModelParameters
   Parameters for stochastic process models.

   .. code-block:: python

      params = ModelParameters(
          drift=0.1,
          diffusion=1.0,
          jump_intensity=0.05,
          equilibrium=10.0,
          reversion_speed=0.5
      )

Result Objects
~~~~~~~~~~~~~~

All analysis results are returned as structured objects with consistent interfaces.

CrossSectionResult
   Results from cross-sectional analysis.

   .. code-block:: python

      result = analyzer.analyze_cross_section(time_point=5.0)
      print(f"Mean: {result.moments['mean']:.2f}")
      print(f"Distribution: {result.distribution_fit['distribution']}")

BayesianResult
   Results from Bayesian analysis.

   .. code-block:: python

      bayes_result = analytics.bayesian_analysis('x', 'y')
      print(f"Credible interval: {bayes_result.credible_intervals['95%']}")

NetworkResult
   Results from network analysis.

   .. code-block:: python

      network_result = analytics.network_analysis()
      print(f"Network nodes: {network_result.network_metrics['num_nodes']}")

Type Hints
----------

Performance Notes
-----------------

**Memory Usage**
  * Large datasets (>100k samples) may require chunked processing
  * Use ``dask`` for parallel processing of large datasets
  * There is no built-in caching layer; reuse fitted models and pass the
    same ``DataCore`` between analyzers instead of re-loading data

**Computational Complexity**
  * Time series analysis: O(n) for basic operations, O(n²) for some advanced methods
  * Network analysis: O(n²) for correlation networks
  * Bayesian analysis: O(n * samples) for MCMC methods
  * Dimensionality reduction: O(n²) to O(n³) depending on method

**Optimization Tips**
  * Use ``numba`` JIT compilation for performance-critical code
  * Enable parallel processing for independent operations
  * Cache intermediate results for repeated analyses
  * Use appropriate data types (float32 vs float64) for memory efficiency

**GPU Acceleration**
  * Enable GPU support with ``evojump[gpu]`` installation
  * GPU acceleration available for: trajectory simulation, matrix operations, neural networks
  * Automatic GPU detection and fallback to CPU

Version Compatibility
---------------------

**Python Versions**
  * Requires Python 3.9+ (``requires-python >=3.9,<3.15``; primary testing on 3.12)
  * Limited support: Python 3.7 (some features may not work)

**Dependency Versions**
  * NumPy: 1.21.0 - 1.26.0
  * SciPy: 1.7.0 - 1.12.0
  * Pandas: 1.3.0 - 2.1.0
  * Matplotlib: 3.5.0 - 3.8.0
  * Plotly: 5.0.0 - 5.17.0
  * Scikit-learn: 1.0.0 - 1.4.0

**Breaking Changes**
  * v0.1.0: Initial release
  * No breaking changes planned for v0.2.0
  * API stability guaranteed until v1.0.0

Migration Guide
~~~~~~~~~~~~~~~

From v0.0.x to v0.1.0:

.. code-block:: python

   # OLD (v0.0.x)
   from evojump.datacore import DataCore
   data = DataCore.load_from_csv("data.csv")

   # NEW (v0.1.0)
   import evojump as ej
   data = ej.DataCore.load_from_csv("data.csv")

Examples Index
--------------

**Basic Usage**
  * `examples/working_demo.py
    <https://github.com/docxology/EvoJump/blob/main/examples/working_demo.py>`_
  * Legacy: ``examples/archive/basic_usage_fixed.py`` (basic_usage example
    was archived; see ``examples/archive/README.md``)

**Advanced Usage**
  * `examples/comprehensive_demo.py
    <https://github.com/docxology/EvoJump/blob/main/examples/comprehensive_demo.py>`_
  * Legacy: ``examples/archive/thin_orchestrator_examples.py`` and
    ``examples/archive/thin_orchestrator_working.py``
  * `examples/comprehensive_advanced_analytics_demo.py
    <https://github.com/docxology/EvoJump/blob/main/examples/comprehensive_advanced_analytics_demo.py>`_

**Visualization**
  * `examples/animation_demo.py
    <https://github.com/docxology/EvoJump/blob/main/examples/animation_demo.py>`_
  * `examples/enhanced_animation_demo.py
    <https://github.com/docxology/EvoJump/blob/main/examples/enhanced_animation_demo.py>`_
  * `examples/comprehensive_animation_demo.py
    <https://github.com/docxology/EvoJump/blob/main/examples/comprehensive_animation_demo.py>`_

**Orchestration**
  * `examples/simple_orchestrator.py
    <https://github.com/docxology/EvoJump/blob/main/examples/simple_orchestrator.py>`_

API Stability
-------------

**Stable APIs** (guaranteed until v1.0.0)
  * ``DataCore`` class and methods
  * ``JumpRope`` class and methods
  * ``LaserPlaneAnalyzer`` class and methods
  * ``TrajectoryVisualizer`` class and methods
  * ``EvolutionSampler`` class and methods
  * ``AnalyticsEngine`` class and methods

**Experimental APIs** (subject to change)
  * Advanced analytics methods (``bayesian_analysis``, ``network_analysis``, etc.)
  * GPU acceleration features
  * Web interface components
  * R integration features

**Deprecated APIs** (will be removed in v1.0.0)
  * None currently deprecated

For the most up-to-date API documentation, see the source code docstrings and the examples directory.
