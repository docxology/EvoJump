Quick Start Guide
=================

This guide provides a rapid introduction to using EvoJump for evolutionary ontogenetic analysis.

5-Minute Tutorial
-----------------

Here's how to analyze developmental trajectories in just 5 minutes:

.. code-block:: python

   import evojump as ej
   import pandas as pd
   import numpy as np

   # Step 1: Load your developmental data
   data = pd.DataFrame({
       'time': [1, 2, 3, 4, 5] * 20,  # 20 individuals, 5 time points each
       'phenotype1': np.random.normal(10, 2, 100),
       'phenotype2': np.random.normal(20, 3, 100)
   })

   # Step 2: Create DataCore instance
   data_core = ej.DataCore.load_from_csv(pd.io.common.StringIO(data.to_csv()),
                                        time_column='time')

   # Step 3: Fit stochastic model
   model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')

   # Step 4: Generate sample trajectories
   trajectories = model.generate_trajectories(n_samples=50, x0=10.0)

   # Step 5: Analyze cross-sections
   analyzer = ej.LaserPlaneAnalyzer(model)
   results = analyzer.analyze_cross_section(time_point=3.0)
   print(f"Mean at time 3.0: {results.moments['mean']:.2f}")

   # Step 6: Create visualizations
   visualizer = ej.TrajectoryVisualizer()
   fig = visualizer.plot_trajectories(model, interactive=False)
   fig.savefig('developmental_trajectories.png')

That's it! You now have a complete analysis of your developmental data.

Complete Workflow Example
-------------------------

Here's a more comprehensive example showing the full EvoJump workflow:

.. code-block:: python

   import evojump as ej
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # Load and preprocess data
   print("1. Loading developmental data...")
   data_core = ej.DataCore.load_from_csv("my_developmental_data.csv",
                                        time_column='developmental_time')

   # Data quality validation
   quality = data_core.validate_data_quality()
   print(f"   Missing data: {quality['missing_data_percentage']['dataset_0']:.1f}%")
   print(f"   Outliers: {quality['outlier_percentage']['dataset_0']:.1f}%")

   # Preprocessing
   data_core.preprocess_data(normalize=True, remove_outliers=True)

   # Model fitting
   print("2. Fitting stochastic process model...")
   time_points = np.sort(data_core.time_series_data[0].data['developmental_time'].unique())
   model = ej.JumpRope.fit(data_core, model_type='jump-diffusion', time_points=time_points)

   # Display model parameters
   params = model.fitted_parameters
   print(f"   Equilibrium: {params.equilibrium:.3f}")
   print(f"   Reversion speed: {params.reversion_speed:.3f}")
   print(f"   Jump intensity: {params.jump_intensity:.4f}")

   # Generate trajectories
   print("3. Generating sample trajectories...")
   trajectories = model.generate_trajectories(n_samples=100, x0=10.0)
   print(f"   Generated {trajectories.shape[0]} trajectories")

   # Cross-sectional analysis
   print("4. Analyzing cross-sectional distributions...")
   analyzer = ej.LaserPlaneAnalyzer(model)

   stages = [2.5, 5.0, 7.5, 10.0]
   for stage in stages:
       result = analyzer.analyze_cross_section(time_point=stage)
       print(f"   Stage {stage:4.1f}: mean = {result.moments['mean']:6.2f}, "
             f"std = {result.moments['std']:.2f}")

   # Advanced analytics
   print("5. Performing advanced analytics...")
   analytics = ej.AnalyticsEngine(data_core)

   # Time series analysis
   ts_results = analytics.analyze_time_series()
   print(f"   Change points detected: {len(ts_results.change_points)}")

   # Bayesian analysis
   bayes_result = analytics.bayesian_analysis('phenotype1', 'phenotype2')
   print(f"   95% credible interval: {bayes_result.credible_intervals.get('95%', 'N/A')}")

   # Network analysis
   network_result = analytics.network_analysis()
   print(f"   Network nodes: {network_result.network_metrics['num_nodes']}")

   # Evolutionary analysis
   print("6. Analyzing evolutionary patterns...")
   sampler = ej.EvolutionSampler(data_core)
   evolution_results = sampler.analyze_evolutionary_patterns()
   pop_stats = evolution_results['population_statistics']
   print(f"   Effective population size: {pop_stats.effective_population_size:.0f}")

   # Create comprehensive visualizations
   print("7. Creating visualizations...")
   visualizer = ej.TrajectoryVisualizer()

   # Trajectory plot
   fig1 = visualizer.plot_trajectories(model, n_trajectories=20)
   plt.savefig('trajectories.png', dpi=150, bbox_inches='tight')

   # Cross-section comparison
   fig2 = visualizer.plot_cross_sections(model, time_points=stages)
   plt.savefig('cross_sections.png', dpi=150, bbox_inches='tight')

   # 3D landscape
   fig3 = visualizer.plot_landscapes(model)
   plt.savefig('landscape.png', dpi=150, bbox_inches='tight')

   # Animation
   anim = visualizer.create_animation(model, n_frames=30)
   anim.save('developmental_animation.gif', writer='pillow', fps=5)

   # Generate comprehensive report
   print("8. Generating analysis report...")
   report = analytics.comprehensive_analysis_report()

   # Save results
   import json
   with open('comprehensive_analysis.json', 'w') as f:
       json.dump(report, f, indent=2, default=str)

   print("✓ Analysis complete! Check output files for results.")

Command Line Interface
----------------------

EvoJump also provides a powerful command-line interface:

.. code-block:: bash

   # Analyze data
   evojump-cli analyze data.csv --output results/

   # Fit model
   evojump-cli fit data.csv --model-type jump-diffusion --output model.pkl

   # Visualize results
   evojump-cli visualize model.pkl --plot-type trajectories --output plots/

   # Sample from populations
   evojump-cli sample population.csv --samples 1000 --output samples.csv

   # Get help
   evojump-cli --help

Real-World Example: Plant Development
-------------------------------------

Here's how EvoJump was used to analyze plant developmental data:

.. code-block:: python

   import evojump as ej
   import pandas as pd

   # Load Arabidopsis thaliana developmental data
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

   # Fit model for each phenotype
   models = {}
   for phenotype in ['leaf_area', 'stem_height', 'root_length', 'chlorophyll_content']:
       # Extract single phenotype data
       phenotype_data = plant_data[['days_after_germination', phenotype]].copy()
       phenotype_data.columns = ['time', 'phenotype']

       # Create temporary DataCore
       temp_data = ej.DataCore.load_from_csv(
           pd.io.common.StringIO(phenotype_data.to_csv()),
           time_column='time'
       )

       # Fit model
       model = ej.JumpRope.fit(temp_data, model_type='jump-diffusion')
       models[phenotype] = model

   # Analyze developmental stages
   analyzer = ej.LaserPlaneAnalyzer(models['leaf_area'])

   stages = [7, 14, 21, 28]  # Days after germination
   for stage in stages:
       result = analyzer.analyze_cross_section(time_point=stage)
       print(f"Leaf area at day {stage}: {result.moments['mean']:.1f} ± {result.moments['std']:.1f}")

   # Compare phenotypes across development
   analytics = ej.AnalyticsEngine(data_core)

   # Multivariate analysis
   mv_results = analytics.analyze_multivariate()
   pca = mv_results['principal_components']
   print(f"PCA explained variance: {pca['explained_variance_ratio'][:3]}")

   # Evolutionary analysis
   sampler = ej.EvolutionSampler(data_core)
   evolution_results = sampler.analyze_evolutionary_patterns()
   print(f"Heritability estimates: {evolution_results['genetic_parameters']}")

Common Patterns
---------------

**Batch Processing Multiple Datasets**

.. code-block:: python

   import glob
   from pathlib import Path

   # Process multiple files
   data_files = glob.glob("data/experiment_*.csv")

   for file_path in data_files:
       experiment_name = Path(file_path).stem

       # Load and analyze each dataset
       data_core = ej.DataCore.load_from_csv(file_path, time_column='time')
       model = ej.JumpRope.fit(data_core)

       # Save results
       output_dir = Path(f"results/{experiment_name}")
       output_dir.mkdir(parents=True, exist_ok=True)

       model.save(output_dir / "model.pkl")
       print(f"✓ Processed {experiment_name}")

**Interactive Analysis**

.. code-block:: python

   import evojump as ej

   # For interactive exploration
   data_core = ej.DataCore.load_from_csv("data.csv", time_column='time')

   # Create interactive visualizer
   visualizer = ej.TrajectoryVisualizer()

   # Launch interactive analysis
   # (This would typically open a web interface or interactive plot)
   print("Interactive analysis ready - implement web interface here")

**Pipeline Integration**

.. code-block:: python

   def analyze_developmental_dataset(data_file, output_dir):
       """Complete analysis pipeline for a single dataset."""

       # Load data
       data_core = ej.DataCore.load_from_csv(data_file, time_column='time')

       # Preprocessing
       data_core.preprocess_data()

       # Model fitting
       model = ej.JumpRope.fit(data_core)

       # Analysis
       analyzer = ej.LaserPlaneAnalyzer(model)
       analytics = ej.AnalyticsEngine(data_core)

       # Generate all outputs
       visualizer = ej.TrajectoryVisualizer()
       visualizer.plot_trajectories(model).savefig(output_dir / 'trajectories.png')
       visualizer.plot_cross_sections(model).savefig(output_dir / 'cross_sections.png')

       # Save model and results
       model.save(output_dir / 'model.pkl')
       report = analytics.comprehensive_analysis_report()

       return {
           'model': model,
           'analyzer': analyzer,
           'report': report,
           'output_files': list(output_dir.glob('*'))
       }

Next Steps
----------

Now that you're familiar with the basics:

1. **Explore Examples**: Check out the ``examples/`` directory for more advanced usage
2. **API Reference**: See :doc:`api_reference` for detailed API documentation
3. **Advanced Usage**: Learn about advanced features in :doc:`advanced_usage`
4. **Contributing**: Help improve EvoJump by reading :doc:`../contributing`

Happy analyzing! 🎉
