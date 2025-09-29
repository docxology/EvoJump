Examples Gallery
================

This section showcases comprehensive examples demonstrating EvoJump's capabilities across various research scenarios.

Getting Started Examples
------------------------

Basic Usage
~~~~~~~~~~~

The most fundamental example showing core EvoJump functionality:

.. literalinclude:: ../examples/basic_usage.py
   :language: python
   :caption: Basic usage example

This example demonstrates:
  * Loading and preprocessing developmental data
  * Fitting stochastic process models
  * Analyzing cross-sectional distributions
  * Creating basic visualizations

Working Demo
~~~~~~~~~~~~

A complete working demonstration:

.. literalinclude:: ../examples/working_demo.py
   :language: python
   :caption: Working demo example

Advanced Examples
-----------------

Comprehensive Demo
~~~~~~~~~~~~~~~~~~

Complete analysis workflow with all features:

.. literalinclude:: ../examples/comprehensive_demo.py
   :language: python
   :caption: Comprehensive demo example

This example shows:
  * Complete data processing pipeline
  * Multiple stochastic model types
  * Advanced statistical analysis
  * Comprehensive visualization
  * Report generation

Advanced Analytics Demo
~~~~~~~~~~~~~~~~~~~~~~~

Demonstrating advanced statistical and machine learning capabilities:

.. literalinclude:: ../examples/comprehensive_advanced_analytics_demo.py
   :language: python
   :caption: Advanced analytics demo

Features demonstrated:
  * Bayesian analysis with credible intervals
  * Network analysis and community detection
  * Causal inference with Granger causality
  * Advanced dimensionality reduction (FastICA, t-SNE)
  * Spectral analysis for frequency domain insights
  * Nonlinear dynamics analysis (Lyapunov exponents)
  * Information theory analysis (entropy measures)
  * Robust statistical methods

Visualization Examples
----------------------

Animation Demo
~~~~~~~~~~~~~~

Basic animation of developmental processes:

.. literalinclude:: ../examples/animation_demo.py
   :language: python
   :caption: Animation demo

Enhanced Animation Demo
~~~~~~~~~~~~~~~~~~~~~~~

Advanced animation with multiple visualization types:

.. literalinclude:: ../examples/enhanced_animation_demo.py
   :language: python
   :caption: Enhanced animation demo

Comprehensive Animation Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete animation suite with multiple conditions and models:

.. literalinclude:: ../examples/comprehensive_animation_demo.py
   :language: python
   :caption: Comprehensive animation demo

Orchestration Examples
----------------------

Simple Orchestrator
~~~~~~~~~~~~~~~~~~~

Basic orchestration pattern:

.. literalinclude:: ../examples/simple_orchestrator.py
   :language: python
   :caption: Simple orchestrator example

Thin Orchestrator Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced orchestration patterns for complex workflows:

.. literalinclude:: ../examples/thin_orchestrator_examples.py
   :language: python
   :caption: Thin orchestrator examples

Thin Orchestrator Working
~~~~~~~~~~~~~~~~~~~~~~~~~

Production-ready orchestration patterns:

.. literalinclude:: ../examples/thin_orchestrator_working.py
   :language: python
   :caption: Thin orchestrator working example

Specialized Use Cases
---------------------

Plant Development Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Example analyzing Arabidopsis thaliana development:

.. code-block:: python

   import evojump as ej
   import pandas as pd

   # Load plant developmental data
   plant_data = pd.read_csv("arabidopsis_development.csv")

   # Create DataCore with multiple phenotypes
   data_core = ej.DataCore.load_from_csv(
       plant_data,
       time_column='days_after_germination',
       phenotype_columns=['leaf_area', 'stem_height', 'root_length', 'chlorophyll_content']
   )

   # Preprocess with biological constraints
   data_core.preprocess_data(
       normalize=True,
       remove_outliers=True,
       interpolate_missing=True
   )

   # Analyze each phenotype
   for phenotype in ['leaf_area', 'stem_height', 'root_length', 'chlorophyll_content']:
       phenotype_data = plant_data[['days_after_germination', phenotype]]
       phenotype_data.columns = ['time', 'phenotype']

       temp_data = ej.DataCore.load_from_csv(
           pd.io.common.StringIO(phenotype_data.to_csv()),
           time_column='time'
       )

       model = ej.JumpRope.fit(temp_data, model_type='jump-diffusion')
       trajectories = model.generate_trajectories(n_samples=100)

       analyzer = ej.LaserPlaneAnalyzer(model)
       result = analyzer.analyze_cross_section(time_point=14.0)
       print(f"{phenotype}: mean = {result.moments['mean']:.2f}")

Evolutionary Genetics Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Population-level evolutionary analysis:

.. code-block:: python

   import evojump as ej

   # Load population genetic data
   population_data = ej.DataCore.load_from_csv("population_genetics.csv")

   # Evolutionary sampling
   sampler = ej.EvolutionSampler(population_data)
   samples = sampler.sample(n_samples=1000, method='monte-carlo')

   # Analyze evolutionary patterns
   evolution_results = sampler.analyze_evolutionary_patterns()
   pop_stats = evolution_results['population_statistics']

   print(f"Effective population size: {pop_stats.effective_population_size:.0f}")
   print(f"Mean heritability: {pop_stats.mean_heritability:.3f}")

   # Cluster individuals by developmental trajectories
   clusters = sampler.cluster_individuals(n_clusters=3)
   print(f"Identified {len(clusters['cluster_statistics'])} trajectory clusters")

Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

Advanced time series analysis of developmental data:

.. code-block:: python

   import evojump as ej

   # Load time series data
   time_series_data = ej.DataCore.load_from_csv("time_series_data.csv")

   # Advanced analytics
   analytics = ej.AnalyticsEngine(time_series_data)

   # Time series analysis
   ts_results = analytics.analyze_time_series()
   print(f"Change points: {len(ts_results.change_points)}")
   print(f"Seasonal patterns: {ts_results.seasonality_analysis}")

   # Forecasting
   forecasts = ts_results.forecasts
   for variable, forecast in forecasts.items():
       print(f"{variable} forecast: {forecast[:5]}...")

   # Spectral analysis
   spectral_result = analytics.spectral_analysis('phenotype1')
   print(f"Dominant frequencies: {spectral_result.dominant_frequencies}")

Network Analysis
~~~~~~~~~~~~~~~~

Gene regulatory network analysis:

.. code-block:: python

   import evojump as ej

   # Load gene expression network data
   network_data = ej.DataCore.load_from_csv("gene_network.csv")

   # Network analysis
   analytics = ej.AnalyticsEngine(network_data)
   network_result = analytics.network_analysis(correlation_threshold=0.7)

   print(f"Network nodes: {network_result.network_metrics['num_nodes']}")
   print(f"Network density: {network_result.network_metrics['density']:.3f}")

   # Community detection
   communities = network_result.community_structure
   print(f"Communities: {communities.get('num_communities', 'N/A')}")

   # Centrality analysis
   centrality = network_result.centrality_measures
   most_central = max(centrality['degree'].items(), key=lambda x: x[1])
   print(f"Most central gene: {most_central[0]} (degree: {most_central[1]:.3f})")

Batch Processing
~~~~~~~~~~~~~~~~

Processing multiple datasets in batch:

.. code-block:: python

   import glob
   from pathlib import Path
   import evojump as ej

   # Find all data files
   data_files = glob.glob("data/experiment_*.csv")

   results = []
   for file_path in data_files:
       experiment_name = Path(file_path).stem

       try:
           # Load and analyze
           data_core = ej.DataCore.load_from_csv(file_path)
           model = ej.JumpRope.fit(data_core)

           # Extract results
           result = {
               'experiment': experiment_name,
               'n_samples': len(data_core.time_series_data[0].data),
               'model_params': model.fitted_parameters,
               'status': 'success'
           }
           results.append(result)
           print(f"✓ Processed {experiment_name}")

       except Exception as e:
           result = {
               'experiment': experiment_name,
               'error': str(e),
               'status': 'failed'
           }
           results.append(result)
           print(f"✗ Failed {experiment_name}: {e}")

   # Save batch results
   import json
   with open('batch_results.json', 'w') as f:
       json.dump(results, f, indent=2, default=str)

Interactive Analysis
~~~~~~~~~~~~~~~~~~~~

Setting up interactive analysis (web interface):

.. code-block:: python

   import evojump as ej
   from flask import Flask, render_template, request, jsonify
   import json

   app = Flask(__name__)

   # Global data storage
   global_data = None
   global_model = None

   @app.route('/')
   def index():
       return render_template('analysis.html')

   @app.route('/load_data', methods=['POST'])
   def load_data():
       global global_data, global_model

       file_path = request.json['file_path']
       global_data = ej.DataCore.load_from_csv(file_path)

       return jsonify({'status': 'success', 'n_samples': len(global_data.time_series_data[0].data)})

   @app.route('/fit_model', methods=['POST'])
   def fit_model():
       global global_model

       model_type = request.json.get('model_type', 'jump-diffusion')
       global_model = ej.JumpRope.fit(global_data, model_type=model_type)

       params = global_model.fitted_parameters
       return jsonify({
           'status': 'success',
           'equilibrium': params.equilibrium,
           'reversion_speed': params.reversion_speed
       })

   @app.route('/analyze', methods=['POST'])
   def analyze():
       global global_model

       time_point = request.json['time_point']
       analyzer = ej.LaserPlaneAnalyzer(global_model)
       result = analyzer.analyze_cross_section(time_point)

       return jsonify({
           'time_point': result.time_point,
           'mean': result.moments['mean'],
           'std': result.moments['std'],
           'distribution': result.distribution_fit.get('distribution', 'unknown')
       })

   if __name__ == '__main__':
       app.run(debug=True)

Real-Time Analysis
~~~~~~~~~~~~~~~~~~

Real-time analysis for streaming data:

.. code-block:: python

   import evojump as ej
   import time
   from threading import Thread

   class RealTimeAnalyzer:
       def __init__(self):
           self.data_buffer = []
           self.model = None
           self.is_running = False

       def start_analysis(self, data_stream):
           """Start real-time analysis thread."""
           self.is_running = True
           thread = Thread(target=self._analysis_loop, args=(data_stream,))
           thread.daemon = True
           thread.start()

       def _analysis_loop(self, data_stream):
           """Main analysis loop."""
           while self.is_running:
               if len(self.data_buffer) > 50:  # Minimum data threshold
                   # Create DataCore from buffer
                   buffer_df = pd.DataFrame(self.data_buffer)
                   data_core = ej.DataCore.load_from_csv(
                       pd.io.common.StringIO(buffer_df.to_csv()),
                       time_column='timestamp'
                   )

                   # Update model
                   if self.model is None:
                       self.model = ej.JumpRope.fit(data_core)
                   else:
                       # Update existing model with new data
                       self.model = ej.JumpRope.fit(data_core)

                   # Generate real-time insights
                   analyzer = ej.LaserPlaneAnalyzer(self.model)
                   latest_result = analyzer.analyze_cross_section(time_point=-1)  # Latest time point

                   print(f"Real-time analysis: mean = {latest_result.moments['mean']:.2f}")

               time.sleep(1)  # Analysis interval

       def add_data_point(self, data_point):
           """Add new data point to buffer."""
           self.data_buffer.append(data_point)
           if len(self.data_buffer) > 1000:  # Buffer size limit
               self.data_buffer = self.data_buffer[-500:]  # Keep recent data

       def stop_analysis(self):
           """Stop real-time analysis."""
           self.is_running = False

   # Usage
   analyzer = RealTimeAnalyzer()

   # Simulate data stream
   for i in range(1000):
       data_point = {
           'timestamp': time.time(),
           'phenotype': 10 + np.sin(i * 0.1) + np.random.normal(0, 0.5)
       }
       analyzer.add_data_point(data_point)

       if i == 100:  # Start analysis after initial data
           analyzer.start_analysis(None)

   time.sleep(10)  # Let it run
   analyzer.stop_analysis()

Research Applications
---------------------

Developmental Biology
~~~~~~~~~~~~~~~~~~~~~

Analyzing organism development:

.. code-block:: python

   # Load developmental time series
   developmental_data = ej.DataCore.load_from_csv("embryo_development.csv")

   # Fit developmental model
   model = ej.JumpRope.fit(developmental_data, model_type='jump-diffusion')

   # Identify critical developmental transitions
   jump_times = model.estimate_jump_times()
   print(f"Critical developmental transitions at: {jump_times}")

   # Analyze developmental stages
   stages = [1.0, 5.0, 10.0, 15.0, 20.0]  # Hours post-fertilization
   for stage in stages:
       result = ej.LaserPlaneAnalyzer(model).analyze_cross_section(stage)
       print(f"Stage {stage}h: {result.distribution_fit['distribution']} "
             f"(mean={result.moments['mean']:.2f})")

Evolutionary Ecology
~~~~~~~~~~~~~~~~~~~

Population dynamics and adaptation:

.. code-block:: python

   # Load population data across environments
   population_data = ej.DataCore.load_from_csv("population_adaptation.csv")

   # Evolutionary analysis
   sampler = ej.EvolutionSampler(population_data)
   evolution_results = sampler.analyze_evolutionary_patterns()

   # Selection analysis
   selection = evolution_results['selection_analysis']
   print(f"Directional selection: {selection['directional_selection']:.3f}")
   print(f"Stabilizing selection: {selection['stabilizing_selection']:.3f}")

   # Heritability analysis
   genetics = evolution_results['genetic_parameters']
   print(f"Narrow-sense heritability: {genetics['narrow_sense_heritability']:.3f}")

Quantitative Genetics
~~~~~~~~~~~~~~~~~~~~~

Genetic architecture of complex traits:

.. code-block:: python

   # Load multi-trait data
   trait_data = ej.DataCore.load_from_csv("quantitative_traits.csv")

   # Multivariate analysis
   analytics = ej.AnalyticsEngine(trait_data)
   mv_results = analytics.analyze_multivariate()

   # Genetic correlations
   pca = mv_results['principal_components']
   print(f"PCA components: {len(pca['explained_variance_ratio'])}")
   print(f"Explained variance: {pca['explained_variance_ratio'][:3]}")

   # Trait clustering
   clusters = mv_results['cluster_analysis']
   print(f"Trait clusters: {len(clusters['cluster_statistics'])}")

Systems Biology
~~~~~~~~~~~~~~~

Network analysis of biological systems:

.. code-block:: python

   # Load interaction network data
   network_data = ej.DataCore.load_from_csv("biological_network.csv")

   # Network analysis
   analytics = ej.AnalyticsEngine(network_data)
   network_result = analytics.network_analysis(correlation_threshold=0.6)

   # Identify key regulators
   centrality = network_result.centrality_measures
   regulators = sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True)[:10]
   print("Top regulators:", [reg[0] for reg in regulators])

   # Community analysis
   communities = network_result.community_structure
   print(f"Biological modules: {communities.get('num_communities', 'N/A')}")

Best Practices
--------------

**Data Preparation**
  * Ensure consistent time points across all samples
  * Remove or interpolate missing data before analysis
  * Check for outliers that may affect model fitting
  * Normalize data when comparing across different scales

**Model Selection**
  * Use 'jump-diffusion' for most biological systems
  * Consider 'ornstein-uhlenbeck' for mean-reverting processes
  * Use 'geometric-jump-diffusion' for multiplicative growth processes
  * Choose 'compound-poisson' for pure jump processes

**Analysis Workflow**
  * Start with basic analysis (trajectories, cross-sections)
  * Use advanced analytics for deeper insights
  * Validate results with multiple methods
  * Create comprehensive visualizations

**Performance Optimization**
  * Use parallel processing for large datasets
  * Enable caching for repeated analyses
  * Consider GPU acceleration for intensive computations
  * Break large analyses into smaller chunks

**Reproducibility**
  * Set random seeds for reproducible results
  * Save complete analysis workflows
  * Document all parameters and settings
  * Version control all analysis code

Troubleshooting Common Issues
-----------------------------

**Model fitting fails**
  * Check data quality and remove outliers
  * Ensure sufficient data points (minimum 10-20 per time point)
  * Try different model types or parameter initialization

**Memory issues with large datasets**
  * Use chunked processing with ``dask``
  * Reduce number of trajectories generated
  * Enable memory-efficient algorithms

**Poor visualization quality**
  * Increase DPI for publication-quality plots
  * Use appropriate color schemes for accessibility
  * Consider interactive plots for complex data

**Statistical test warnings**
  * Check assumptions (normality, equal variance)
  * Use robust statistical methods when assumptions violated
  * Consider non-parametric alternatives

For more examples and detailed tutorials, see the ``examples/`` directory in the source code.

Running Examples
----------------

To run any example:

.. code-block:: bash

   # From the project root directory
   python examples/example_name.py

   # Or with specific Python path
   PYTHONPATH=src python examples/example_name.py

All examples are designed to be self-contained and will generate output files demonstrating the results.

Example Output Files
~~~~~~~~~~~~~~~~~~~~

Examples typically generate:

* **Plots**: ``*.png`` files with visualizations
* **Animations**: ``*.gif`` files with animated processes
* **Reports**: ``*.json`` files with analysis results
* **Models**: ``*.pkl`` files with fitted models
* **Data**: ``*.csv`` files with processed data

These files are saved in subdirectories named after the example (e.g., ``demo_outputs/``, ``animation_outputs/``).
