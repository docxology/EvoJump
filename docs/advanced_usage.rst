Advanced Usage Guide
=====================

This guide covers advanced features and sophisticated analysis techniques in EvoJump.

High-Performance Computing
--------------------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Enable parallel processing for large datasets:

.. code-block:: python

   import evojump as ej

   # Enable parallel processing
   ej.config.set_num_threads(8)

   # Use with dask for distributed computing
   import dask.dataframe as dd

   # Load large dataset with dask
   large_data = dd.read_csv("large_dataset.csv")

   # Process in parallel
   def process_chunk(chunk):
       data_core = ej.DataCore.load_from_csv(
           pd.io.common.StringIO(chunk.to_csv()),
           time_column='time'
       )
       model = ej.JumpRope.fit(data_core)
       return model.fitted_parameters

   # Process chunks in parallel
   results = large_data.map_partitions(process_chunk).compute()

GPU Acceleration
~~~~~~~~~~~~~~~~

Enable GPU acceleration for intensive computations:

.. code-block:: python

   import evojump as ej

   # Enable GPU acceleration (requires cupy)
   try:
       import cupy as cp
       ej.config.enable_gpu()
       print("GPU acceleration enabled")
   except ImportError:
       print("GPU acceleration not available")

   # GPU-accelerated trajectory simulation
   model = ej.JumpRope.fit(data_core)
   trajectories = model.generate_trajectories(n_samples=10000)  # Large number

Memory Optimization
~~~~~~~~~~~~~~~~~~~

Optimize memory usage for large datasets:

.. code-block:: python

   import evojump as ej

   # Set memory limits
   ej.config.set_memory_limit('8GB')

   # Use memory-efficient algorithms
   analytics = ej.AnalyticsEngine(data_core)

   # Process in chunks
   chunk_size = 10000
   for i in range(0, len(data_core.data), chunk_size):
       chunk = data_core.data.iloc[i:i+chunk_size]
       chunk_analysis = ej.AnalyticsEngine(chunk)
       # Process chunk...

Custom Stochastic Processes
---------------------------

Define custom stochastic processes:

.. code-block:: python

   import numpy as np
   from evojump.jumprope import StochasticProcess, ModelParameters

   class CustomProcess(StochasticProcess):
       """Custom stochastic process implementation."""

       def __init__(self, parameters: ModelParameters):
           super().__init__(parameters)
           self.process_name = "Custom Process"

       def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
           """Simulate custom process trajectories."""
           dt = np.diff(t)
           n_steps = len(t) - 1
           paths = np.zeros((n_paths, len(t)))
           paths[:, 0] = x0

           for i in range(n_paths):
               x = x0
               for j in range(n_steps):
                   # Custom dynamics
                   drift = self.parameters.drift * x * dt[j]
                   diffusion = self.parameters.diffusion * np.sqrt(dt[j]) * np.random.normal()

                   # Custom jump process
                   jump_prob = self.parameters.jump_intensity * dt[j]
                   if np.random.random() < jump_prob:
                       jump_size = np.random.normal(self.parameters.jump_mean,
                                                  self.parameters.jump_std)
                       x += jump_size

                   x += drift + diffusion
                   paths[i, j + 1] = x

           return paths

       def log_likelihood(self, data: np.ndarray, dt: float) -> float:
           """Custom log-likelihood computation."""
           # Implement custom likelihood
           return 0.0  # Placeholder

       def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
           """Custom parameter estimation."""
           # Implement parameter estimation
           return self.parameters

   # Use custom process
   custom_params = ModelParameters(drift=0.1, diffusion=0.5, jump_intensity=0.05)
   custom_process = CustomProcess(custom_params)
   model = ej.JumpRope(custom_process, time_points)

Advanced Statistical Methods
----------------------------

Bayesian Analysis
~~~~~~~~~~~~~~~~~

Advanced Bayesian inference:

.. code-block:: python

   import evojump as ej

   analytics = ej.AnalyticsEngine(data_core)

   # Custom prior specification
   custom_priors = {
       'location': {'mean': 10.0, 'precision': 0.1},
       'scale': {'shape': 2.0, 'rate': 1.0}
   }

   # Bayesian analysis with custom priors
   bayes_result = analytics.bayesian_analysis(
       'phenotype1', 'phenotype2',
       n_samples=5000,
       priors=custom_priors
   )

   # Model comparison
   comparison = analytics.bayesian_analyzer.bayesian_model_comparison(
       model1_likelihood=-100.0,
       model2_likelihood=-95.0,
       model1_complexity=3,
       model2_complexity=4
   )

Network Analysis
~~~~~~~~~~~~~~~~

Advanced network analysis techniques:

.. code-block:: python

   import evojump as ej
   import networkx as nx

   analytics = ej.AnalyticsEngine(data_core)

   # Custom correlation thresholds
   network_result = analytics.network_analysis(correlation_threshold=0.8)

   # Add custom node attributes
   for node in network_result.graph.nodes():
       network_result.graph.nodes[node]['custom_attribute'] = some_value

   # Advanced community detection
   from networkx.algorithms import community
   communities = community.louvain_communities(network_result.graph)

   # Path analysis
   paths = nx.all_shortest_paths(network_result.graph, 'gene1', 'gene2')

Causal Inference
~~~~~~~~~~~~~~~~

Advanced causal discovery methods:

.. code-block:: python

   import evojump as ej

   analytics = ej.AnalyticsEngine(data_core)

   # Multiple lag analysis
   causal_result = analytics.causal_inference(
       'cause_variable', 'effect_variable',
       max_lag=10,
       method='granger'
   )

   # Mediation analysis
   mediation_result = analytics.causal_inference(
       'treatment', 'outcome',
       mediator='intermediate_variable'
   )

   # Sensitivity analysis
   sensitivity = causal_result.sensitivity_analysis

Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~

Advanced dimensionality reduction techniques:

.. code-block:: python

   import evojump as ej

   analytics = ej.AnalyticsEngine(data_core)

   # Custom t-SNE parameters
   tsne_result = analytics.advanced_dimensionality_reduction(
       method='tsne',
       n_components=3,
       perplexity=50,
       learning_rate=500,
       n_iter=2000
   )

   # UMAP embedding
   umap_result = analytics.advanced_dimensionality_reduction(
       method='umap',
       n_components=2,
       n_neighbors=15,
       min_dist=0.1
   )

   # Diffusion maps
   diffusion_result = analytics.advanced_dimensionality_reduction(
       method='diffusion_maps',
       n_components=2,
       alpha=1.0,
       n_neighbors=10
   )

Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

Advanced time series methods:

.. code-block:: python

   import evojump as ej

   analytics = ej.AnalyticsEngine(data_core)

   # Wavelet analysis
   wavelet_result = analytics.advanced_time_series_analysis(
       method='wavelet',
       wavelet='morl',
       scales=np.arange(1, 128)
   )

   # State space modeling
   ssm_result = analytics.advanced_time_series_analysis(
       method='state_space',
       order=(2, 1, 1)  # ARIMA order
   )

   # Change point detection with multiple methods
   changes = analytics.detect_changes(
       method='multiple',
       methods=['cusum', 'bayesian', 'information']
   )

Robust Statistics
~~~~~~~~~~~~~~~~~

Advanced robust statistical methods:

.. code-block:: python

   import evojump as ej

   analytics = ej.AnalyticsEngine(data_core)

   # Robust regression
   robust_result = analytics.robust_statistical_analysis(
       'response_variable',
       method='robust_regression',
       robust_method='huber'
   )

   # Outlier detection
   outlier_result = analytics.robust_statistical_analysis(
       'data_column',
       method='outlier_detection',
       contamination=0.1
   )

   # Influence analysis
   influence_result = analytics.robust_statistical_analysis(
       'data_column',
       method='influence_analysis'
   )

Machine Learning Integration
----------------------------

Deep Learning
~~~~~~~~~~~~~

Integrate with PyTorch/TensorFlow:

.. code-block:: python

   import evojump as ej
   import torch
   import torch.nn as nn

   # Convert EvoJump data to PyTorch tensors
   trajectories = model.generate_trajectories(n_samples=1000)
   trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)

   # Define neural network
   class TrajectoryPredictor(nn.Module):
       def __init__(self):
           super().__init__()
           self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2)
           self.fc = nn.Linear(50, 1)

       def forward(self, x):
           lstm_out, _ = self.lstm(x)
           return self.fc(lstm_out[-1])

   # Train model
   model = TrajectoryPredictor()
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters())

   # Training loop...

Ensemble Methods
~~~~~~~~~~~~~~~~

Advanced ensemble modeling:

.. code-block:: python

   import evojump as ej
   from sklearn.ensemble import StackingRegressor, VotingRegressor

   analytics = ej.AnalyticsEngine(data_core)

   # Custom ensemble
   base_models = [
       ('rf', ej.PredictiveModeler.models['random_forest']),
       ('gb', ej.PredictiveModeler.models['gradient_boosting']),
       ('nn', ej.PredictiveModeler.models['neural_network'])
   ]

   ensemble = VotingRegressor(base_models)
   ensemble_result = analytics.predictive_modeling(
       'target',
       feature_variables=['feature1', 'feature2'],
       models={'ensemble': ensemble}
   )

Feature Engineering
~~~~~~~~~~~~~~~~~~~

Advanced feature engineering:

.. code-block:: python

   import evojump as ej

   # Extract features from trajectories
   def extract_trajectory_features(trajectory):
       """Extract statistical features from a trajectory."""
       return {
           'mean': np.mean(trajectory),
           'std': np.std(trajectory),
           'trend': np.polyfit(range(len(trajectory)), trajectory, 1)[0],
           'autocorr': np.corrcoef(trajectory[:-1], trajectory[1:])[0, 1],
           'peaks': len(scipy.signal.find_peaks(trajectory)[0])
       }

   # Apply to all trajectories
   trajectories = model.generate_trajectories(n_samples=100)
   features = [extract_trajectory_features(traj) for traj in trajectories]

   # Create feature dataset
   features_df = pd.DataFrame(features)

Custom Visualization
---------------------

Advanced Plotting
~~~~~~~~~~~~~~~~~

Create custom visualizations:

.. code-block:: python

   import evojump as ej
   import matplotlib.pyplot as plt
   import seaborn as sns

   # Custom trajectory plot
   def plot_custom_trajectories(model, highlight_individuals=None):
       """Create custom trajectory visualization."""

       fig, axes = plt.subplots(2, 2, figsize=(15, 12))

       # Individual trajectories
       trajectories = model.trajectories
       time_points = model.time_points

       for i in range(min(50, len(trajectories))):
           alpha = 1.0 if highlight_individuals and i in highlight_individuals else 0.3
           axes[0, 0].plot(time_points, trajectories[i], alpha=alpha)

       # Highlighted trajectories
       if highlight_individuals:
           for i in highlight_individuals:
               axes[0, 0].plot(time_points, trajectories[i], linewidth=3, alpha=0.8)

       axes[0, 0].set_title('Individual Trajectories')
       axes[0, 0].set_xlabel('Time')
       axes[0, 0].set_ylabel('Phenotype')

       # Distribution evolution
       for i, time_idx in enumerate([0, len(time_points)//2, -1]):
           cross_section = model.compute_cross_sections(time_idx)
           axes[0, 1].hist(cross_section, bins=30, alpha=0.7,
                          label=f'Time {time_points[time_idx]:.1f}')

       axes[0, 1].set_title('Distribution Evolution')
       axes[0, 1].set_xlabel('Phenotype Value')
       axes[0, 1].set_ylabel('Frequency')
       axes[0, 1].legend()

       # 3D landscape
       ax3d = fig.add_subplot(2, 2, 3, projection='3d')
       for i in range(min(20, len(trajectories))):
           ax3d.plot(time_points, [i]*len(time_points), trajectories[i], alpha=0.6)

       ax3d.set_xlabel('Time')
       ax3d.set_ylabel('Individual')
       ax3d.set_zlabel('Phenotype')
       ax3d.set_title('3D Phenotypic Landscape')

       # Statistics over time
       stats_data = []
       for time_idx in range(len(time_points)):
           cross_section = model.compute_cross_sections(time_idx)
           stats_data.append({
               'time': time_points[time_idx],
               'mean': np.mean(cross_section),
               'std': np.std(cross_section),
               'cv': np.std(cross_section) / np.mean(cross_section)
           })

       stats_df = pd.DataFrame(stats_data)
       axes[1, 0].plot(stats_df['time'], stats_df['mean'], 'b-', label='Mean')
       axes[1, 0].fill_between(stats_df['time'],
                              stats_df['mean'] - stats_df['std'],
                              stats_df['mean'] + stats_df['std'],
                              alpha=0.3)
       axes[1, 0].set_title('Statistics Over Time')
       axes[1, 0].set_xlabel('Time')
       axes[1, 0].set_ylabel('Value')
       axes[1, 0].legend()

       plt.tight_layout()
       return fig

   # Use custom visualization
   model = ej.JumpRope.fit(data_core)
   custom_plot = plot_custom_trajectories(model, highlight_individuals=[5, 10, 15])
   custom_plot.savefig('custom_trajectories.png', dpi=150, bbox_inches='tight')

Interactive Dashboards
~~~~~~~~~~~~~~~~~~~~~~

Create interactive web dashboards:

.. code-block:: python

   import evojump as ej
   import dash
   from dash import dcc, html
   import plotly.graph_objs as go

   # Create Dash app
   app = dash.Dash(__name__)

   # Load data
   data_core = ej.DataCore.load_from_csv('data.csv')
   model = ej.JumpRope.fit(data_core)

   app.layout = html.Div([
       html.H1('EvoJump Interactive Dashboard'),

       # Trajectory plot
       dcc.Graph(id='trajectory-plot'),

       # Controls
       html.Div([
           html.Label('Model Type'),
           dcc.Dropdown(
               id='model-type',
               options=[
                   {'label': 'Jump-Diffusion', 'value': 'jump-diffusion'},
                   {'label': 'Ornstein-Uhlenbeck', 'value': 'ornstein-uhlenbeck'},
                   {'label': 'Geometric Jump-Diffusion', 'value': 'geometric-jump-diffusion'}
               ],
               value='jump-diffusion'
           ),

           html.Label('Number of Trajectories'),
           dcc.Slider(
               id='n-trajectories',
               min=10, max=100, step=10, value=50
           )
       ]),

       # Cross-section plot
       dcc.Graph(id='cross-section-plot'),

       # Analysis results
       html.Div(id='analysis-results')
   ])

   @app.callback(
       [dash.dependencies.Output('trajectory-plot', 'figure'),
        dash.dependencies.Output('cross-section-plot', 'figure'),
        dash.dependencies.Output('analysis-results', 'children')],
       [dash.dependencies.Input('model-type', 'value'),
        dash.dependencies.Input('n-trajectories', 'value')]
   )
   def update_plots(model_type, n_trajectories):
       # Refit model with new parameters
       model = ej.JumpRope.fit(data_core, model_type=model_type)
       trajectories = model.generate_trajectories(n_samples=n_trajectories)

       # Create trajectory plot
       trajectory_fig = go.Figure()
       for i in range(min(20, n_trajectories)):
           trajectory_fig.add_trace(go.Scatter(
               x=model.time_points,
               y=trajectories[i],
               mode='lines',
               opacity=0.6,
               name=f'Trajectory {i}'
           ))

       # Create cross-section plot
       cross_section_fig = go.Figure()
       for time_idx in [0, len(model.time_points)//2, -1]:
           cross_section = model.compute_cross_sections(time_idx)
           cross_section_fig.add_trace(go.Histogram(
               x=cross_section,
               nbinsx=30,
               opacity=0.7,
               name=f'Time {model.time_points[time_idx]:.1f}'
           ))

       # Analysis results
       analyzer = ej.LaserPlaneAnalyzer(model)
       result = analyzer.analyze_cross_section(time_point=model.time_points[-1])

       analysis_text = f"""
       Analysis Results:
       - Distribution: {result.distribution_fit.get('distribution', 'unknown')}
       - Mean: {result.moments['mean']:.3f}
       - Std: {result.moments['std']:.3f}
       - AIC: {result.goodness_of_fit['aic']:.2f}
       """

       return trajectory_fig, cross_section_fig, analysis_text

   if __name__ == '__main__':
       app.run_server(debug=True)

Integration with External Tools
-------------------------------

R Integration
~~~~~~~~~~~~~

Integrate with R for specialized analyses:

.. code-block:: python

   import evojump as ej

   # Export data to R
   data_core.save_processed_data('data_for_r.csv')

   # Use rpy2 for R integration
   import rpy2.robjects as ro
   from rpy2.robjects import pandas2ri

   pandas2ri.activate()

   # Load data in R
   r_data = ro.r('read.csv("data_for_r.csv")')

   # Run R analysis (example: lme4 mixed models)
   ro.r('''
   library(lme4)
   model <- lmer(phenotype ~ time + (1|individual), data = data)
   summary(model)
   ''')

   # Import results back to Python
   results = ro.r('summary(model)')

Jupyter Notebook Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Best practices for Jupyter notebooks:

.. code-block:: python

   # Cell 1: Setup
   import evojump as ej
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # Enable inline plotting
   %matplotlib inline

   # Set plotting style
   plt.style.use('ggplot')

   # Cell 2: Data loading and preprocessing
   data_core = ej.DataCore.load_from_csv('data.csv')
   data_core.preprocess_data()

   # Cell 3: Model fitting
   model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')

   # Cell 4: Analysis
   analyzer = ej.LaserPlaneAnalyzer(model)
   analytics = ej.AnalyticsEngine(data_core)

   # Cell 5: Visualization
   visualizer = ej.TrajectoryVisualizer()
   fig = visualizer.plot_trajectories(model)
   plt.show()

   # Cell 6: Advanced analysis
   network_result = analytics.network_analysis()
   print(f"Network analysis: {network_result.network_metrics['num_nodes']} nodes")

Database Integration
~~~~~~~~~~~~~~~~~~~~

Store and retrieve EvoJump results in databases:

.. code-block:: python

   import evojump as ej
   from sqlalchemy import create_engine, Column, Integer, String, Float, Text
   from sqlalchemy.ext.declarative import declarative_base
   from sqlalchemy.orm import sessionmaker
   import json

   Base = declarative_base()

   class AnalysisResult(Base):
       __tablename__ = 'analysis_results'

       id = Column(Integer, primary_key=True)
       experiment_name = Column(String)
       model_type = Column(String)
       parameters = Column(Text)  # JSON string
       results = Column(Text)     # JSON string
       created_at = Column(String)

   # Create database
   engine = create_engine('sqlite:///evojump_results.db')
   Base.metadata.create_all(engine)
   Session = sessionmaker(bind=engine)
   session = Session()

   # Store results
   model = ej.JumpRope.fit(data_core)
   params_json = json.dumps(model.fitted_parameters.__dict__)

   result = AnalysisResult(
       experiment_name='experiment_1',
       model_type='jump-diffusion',
       parameters=params_json,
       results='analysis_results_json',
       created_at=str(pd.Timestamp.now())
   )

   session.add(result)
   session.commit()

   # Retrieve results
   stored_result = session.query(AnalysisResult).filter_by(
       experiment_name='experiment_1'
   ).first()

   # Reconstruct model from stored parameters
   stored_params = json.loads(stored_result.parameters)
   model_params = ej.ModelParameters(**stored_params)

Performance Monitoring
----------------------

Profiling and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profile EvoJump performance:

.. code-block:: python

   import evojump as ej
   import cProfile
   import pstats
   import time

   # Profile analysis
   profiler = cProfile.Profile()
   profiler.enable()

   # Run analysis
   data_core = ej.DataCore.load_from_csv('large_dataset.csv')
   model = ej.JumpRope.fit(data_core)
   trajectories = model.generate_trajectories(n_samples=1000)

   profiler.disable()

   # Analyze results
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions

   # Memory profiling
   import psutil
   import os

   process = psutil.Process(os.getpid())
   memory_before = process.memory_info().rss / 1024 / 1024  # MB

   # Run memory-intensive operation
   large_analysis = ej.AnalyticsEngine(data_core).comprehensive_analysis_report()

   memory_after = process.memory_info().rss / 1024 / 1024  # MB
   print(f"Memory usage: {memory_after - memory_before:.1f} MB")

Benchmarking
~~~~~~~~~~~~

Benchmark EvoJump against other tools:

.. code-block:: python

   import evojump as ej
   import time
   import numpy as np

   # Generate benchmark data
   np.random.seed(42)
   large_data = pd.DataFrame({
       'time': np.repeat(np.arange(1, 101), 100),
       'phenotype1': np.random.normal(0, 1, 10000),
       'phenotype2': np.random.normal(0, 1, 10000)
   })

   # Benchmark different operations
   benchmarks = {}

   # Data loading benchmark
   start_time = time.time()
   data_core = ej.DataCore.load_from_csv(
       pd.io.common.StringIO(large_data.to_csv()),
       time_column='time'
   )
   benchmarks['data_loading'] = time.time() - start_time

   # Model fitting benchmark
   start_time = time.time()
   model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')
   benchmarks['model_fitting'] = time.time() - start_time

   # Trajectory generation benchmark
   start_time = time.time()
   trajectories = model.generate_trajectories(n_samples=1000)
   benchmarks['trajectory_generation'] = time.time() - start_time

   # Analysis benchmark
   start_time = time.time()
   analytics = ej.AnalyticsEngine(data_core)
   report = analytics.comprehensive_analysis_report()
   benchmarks['comprehensive_analysis'] = time.time() - start_time

   # Print results
   for operation, duration in benchmarks.items():
       print(f"{operation}: {duration:.3f} seconds")

Logging and Debugging
~~~~~~~~~~~~~~~~~~~~~

Configure detailed logging:

.. code-block:: python

   import evojump as ej
   import logging

   # Configure logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )

   # Enable EvoJump debug logging
   ej.logging.setup_logging(level='DEBUG')

   # Log to file
   file_handler = logging.FileHandler('evojump_analysis.log')
   file_handler.setLevel(logging.INFO)
   ej.logging.get_logger().addHandler(file_handler)

   # Run analysis with detailed logging
   data_core = ej.DataCore.load_from_csv('data.csv')
   model = ej.JumpRope.fit(data_core)

   # Analysis will be logged in detail

Error Handling and Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement robust error handling:

.. code-block:: python

   import evojump as ej
   from evojump.exceptions import EvoJumpError

   def robust_analysis(data_file):
       """Robust analysis with error handling."""
       try:
           # Load data
           data_core = ej.DataCore.load_from_csv(data_file)

           # Validate data quality
           quality = data_core.validate_data_quality()
           if quality['missing_data_percentage']['dataset_0'] > 50:
               raise EvoJumpError("Too much missing data")

           # Preprocess
           data_core.preprocess_data()

           # Fit model with error handling
           try:
               model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')
           except Exception as e:
               print(f"Primary model failed, trying alternative...")
               model = ej.JumpRope.fit(data_core, model_type='ornstein-uhlenbeck')

           # Generate results
           trajectories = model.generate_trajectories(n_samples=100)
           analyzer = ej.LaserPlaneAnalyzer(model)
           results = analyzer.analyze_cross_section(time_point=5.0)

           return {
               'model': model,
               'results': results,
               'status': 'success'
           }

       except EvoJumpError as e:
           print(f"EvoJump error: {e}")
           return {'status': 'failed', 'error': str(e)}

       except Exception as e:
           print(f"Unexpected error: {e}")
           return {'status': 'failed', 'error': str(e)}

   # Use robust analysis
   result = robust_analysis('data.csv')
   if result['status'] == 'success':
       print("Analysis successful!")
   else:
       print(f"Analysis failed: {result['error']}")

Custom Extensions
-----------------

Creating Extensions
~~~~~~~~~~~~~~~~~~~

Create custom EvoJump extensions:

.. code-block:: python

   # custom_extension.py
   import evojump as ej
   from evojump.analytics_engine import AnalyticsEngine

   class CustomAnalyzer:
       """Custom analysis extension."""

       def __init__(self, data_core):
           self.data_core = data_core
           self.analytics = AnalyticsEngine(data_core)

       def custom_analysis_method(self, parameter):
           """Implement custom analysis."""
           # Custom logic here
           return {"custom_result": parameter * 2}

   # Register extension
   def register_custom_analyzer():
       """Register custom analyzer with EvoJump."""
       # This would typically be done through a plugin system
       pass

Plugin System
~~~~~~~~~~~~~

Implement a plugin system for EvoJump:

.. code-block:: python

   import evojump as ej
   import importlib
   import inspect

   class PluginManager:
       """Manage EvoJump plugins."""

       def __init__(self):
           self.plugins = {}
           self.analyzers = {}

       def load_plugin(self, plugin_path):
           """Load a plugin module."""
           try:
               module = importlib.import_module(plugin_path)
               self._register_plugin_components(module)
               return True
           except Exception as e:
               print(f"Failed to load plugin {plugin_path}: {e}")
               return False

       def _register_plugin_components(self, module):
           """Register plugin components."""
           for name, obj in inspect.getmembers(module):
               if inspect.isclass(obj):
                   # Register analyzers
                   if issubclass(obj, ej.LaserPlaneAnalyzer):
                       self.analyzers[name] = obj

                   # Register other plugin components
                   if hasattr(obj, 'plugin_type'):
                       self.plugins[name] = obj

       def get_analyzer(self, analyzer_name):
           """Get registered analyzer."""
           return self.analyzers.get(analyzer_name)

   # Usage
   plugin_manager = PluginManager()
   plugin_manager.load_plugin('my_custom_analyzer')

   # Use custom analyzer
   custom_analyzer_class = plugin_manager.get_analyzer('CustomAnalyzer')
   if custom_analyzer_class:
       analyzer = custom_analyzer_class(data_core)
       results = analyzer.custom_analysis_method(parameter=5.0)

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

Advanced configuration management:

.. code-block:: python

   import evojump as ej
   import json
   from pathlib import Path

   class ConfigurationManager:
       """Advanced configuration management."""

       def __init__(self, config_file=None):
           self.config = self._load_default_config()
           if config_file:
               self.load_config(config_file)

       def _load_default_config(self):
           """Load default configuration."""
           return {
               'analysis': {
                   'default_model': 'jump-diffusion',
                   'n_bootstrap': 1000,
                   'confidence_level': 0.95
               },
               'visualization': {
                   'default_style': 'ggplot',
                   'dpi': 150,
                   'figsize': [12, 8]
               },
               'computation': {
                   'num_threads': 4,
                   'memory_limit': '4GB',
                   'cache_enabled': True
               }
           }

       def load_config(self, config_file):
           """Load configuration from file."""
           if isinstance(config_file, str):
               config_file = Path(config_file)

           if config_file.exists():
               with open(config_file, 'r') as f:
                   file_config = json.load(f)
                   self._merge_config(file_config)

       def save_config(self, config_file):
           """Save configuration to file."""
           config_file = Path(config_file)
           with open(config_file, 'w') as f:
               json.dump(self.config, f, indent=2)

       def _merge_config(self, file_config):
           """Merge file configuration with current config."""
           def merge_dicts(base, update):
               for key, value in update.items():
                   if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                       merge_dicts(base[key], value)
                   else:
                       base[key] = value

           merge_dicts(self.config, file_config)

       def get_config(self, section, key=None):
           """Get configuration value."""
           if key is None:
               return self.config.get(section, {})
           return self.config.get(section, {}).get(key)

       def set_config(self, section, key, value):
           """Set configuration value."""
           if section not in self.config:
               self.config[section] = {}
           self.config[section][key] = value

   # Usage
   config_manager = ConfigurationManager()
   config_manager.load_config('custom_config.json')

   # Get configuration values
   default_model = config_manager.get_config('analysis', 'default_model')
   num_threads = config_manager.get_config('computation', 'num_threads')

   # Override settings
   config_manager.set_config('analysis', 'n_bootstrap', 2000)
   config_manager.save_config('updated_config.json')

Best Practices for Advanced Usage
----------------------------------

**Memory Management**
  * Monitor memory usage with ``psutil`` or ``memory_profiler``
  * Use chunked processing for large datasets
  * Enable garbage collection: ``import gc; gc.collect()``
  * Consider using ``dask`` for out-of-core processing

**Performance Optimization**
  * Profile code with ``cProfile`` to identify bottlenecks
  * Use ``numba`` JIT compilation for numerical code
  * Enable parallel processing for independent operations
  * Cache intermediate results when possible

**Error Handling**
  * Implement comprehensive try-catch blocks
  * Use EvoJump's built-in exception classes
  * Log errors with appropriate detail levels
  * Provide graceful degradation for failed operations

**Reproducibility**
  * Set random seeds for reproducible results
  * Document all parameters and versions
  * Save complete analysis workflows
  * Use version control for analysis scripts

**Scalability**
  * Design analyses to work with varying data sizes
  * Use streaming algorithms for large datasets
  * Implement checkpointing for long-running analyses
  * Consider distributed computing for very large datasets

**Integration**
  * Use established interfaces for external tool integration
  * Implement plugin systems for extensibility
  * Provide configuration management for complex setups
  * Support multiple data formats and sources

This advanced usage guide provides the tools and techniques needed for sophisticated EvoJump applications in research and production environments.
