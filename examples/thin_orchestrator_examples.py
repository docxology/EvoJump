#!/usr/bin/env python3
"""
Thin Orchestrator Examples for EvoJump Package

This file demonstrates various thin orchestrator patterns for using the EvoJump package
in different research scenarios. These examples show how to use the package's modular
components in a clean, orchestrated way.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import evojump as ej
import matplotlib.pyplot as plt

# Example 1: Developmental Trajectory Analysis Orchestrator
class DevelopmentalTrajectoryOrchestrator:
    """Thin orchestrator for analyzing developmental trajectories."""

    def __init__(self, data_file: Path):
        """Initialize with data file path."""
        self.data_file = data_file
        self.data_core = None
        self.model = None
        self.analyzer = None
        self.visualizer = None

    def load_and_preprocess_data(self):
        """Load and preprocess developmental data."""
        print("Loading and preprocessing data...")
        self.data_core = ej.DataCore.load_from_csv(
            file_path=self.data_file,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        # Preprocess data
        self.data_core.preprocess_data(
            normalize=True,
            remove_outliers=True,
            interpolate_missing=True
        )

        # Validate data quality
        quality = self.data_core.validate_data_quality()
        print(f"Data quality - Missing: {quality['missing_data_percentage']['dataset_0']".2f"}%, "
              f"Outliers: {quality['outlier_percentage']['dataset_0']".2f"}%")

        return self.data_core

    def fit_stochastic_model(self, model_type: str = 'jump-diffusion'):
        """Fit stochastic model to data."""
        print(f"Fitting {model_type} model...")
        time_points = np.sort(self.data_core.time_series_data[0].data['time'].unique())

        self.model = ej.JumpRope.fit(
            self.data_core,
            model_type=model_type,
            time_points=time_points
        )

        print(f"Model fitted - Equilibrium: {self.model.fitted_parameters.equilibrium".2f"}, "
              f"Jump intensity: {self.model.fitted_parameters.jump_intensity".3f"}")

        return self.model

    def analyze_cross_sections(self, time_points: list = None):
        """Analyze cross-sectional distributions."""
        if self.analyzer is None:
            self.analyzer = ej.LaserPlaneAnalyzer(self.model)

        if time_points is None:
            time_points = [5.0, 10.0, 15.0, 20.0]

        print("Analyzing cross-sectional distributions...")
        results = {}

        for time_point in time_points:
            result = self.analyzer.analyze_cross_section(time_point)
            results[time_point] = result
            print(f"Time {time_point"4.1f"}: mean = {result.moments['mean']"6.2f"}, "
                  f"std = {result.moments['std']".2f"}")

        return results

    def create_visualizations(self, output_dir: Path):
        """Create comprehensive visualizations."""
        if self.visualizer is None:
            self.visualizer = ej.TrajectoryVisualizer()

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Creating visualizations...")

        # Generate trajectories
        self.model.generate_trajectories(n_samples=50, x0=10.0)

        # Plot trajectories
        fig = self.visualizer.plot_trajectories(self.model, n_trajectories=10)
        fig.savefig(output_dir / 'developmental_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Plot landscapes
        fig = self.visualizer.plot_landscapes(self.model)
        fig.savefig(output_dir / 'phenotypic_landscape.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Visualizations saved to {output_dir}")

    def run_complete_analysis(self, output_dir: Path):
        """Run complete developmental trajectory analysis."""
        print("Running complete developmental trajectory analysis...")
        print("=" * 60)

        # Step 1: Load and preprocess
        self.load_and_preprocess_data()

        # Step 2: Fit model
        self.fit_stochastic_model()

        # Step 3: Analyze cross-sections
        cross_section_results = self.analyze_cross_sections()

        # Step 4: Create visualizations
        self.create_visualizations(output_dir)

        print("=" * 60)
        print("Analysis complete!")
        return {
            'data_core': self.data_core,
            'model': self.model,
            'cross_sections': cross_section_results
        }


# Example 2: Evolutionary Analysis Orchestrator
class EvolutionaryAnalysisOrchestrator:
    """Thin orchestrator for evolutionary analysis."""

    def __init__(self, population_data_file: Path):
        """Initialize with population data file."""
        self.population_data_file = population_data_file
        self.sampler = None
        self.analytics = None

    def setup_sampling_and_analysis(self):
        """Set up sampling and analysis components."""
        print("Setting up evolutionary analysis components...")

        # Load population data
        population_data = pd.read_csv(self.population_data_file)

        # Create evolution sampler
        self.sampler = ej.EvolutionSampler(population_data, time_column='time')

        # Create analytics engine
        self.analytics = ej.AnalyticsEngine(population_data, time_column='time')

        print("Components initialized successfully")
        return self.sampler, self.analytics

    def perform_evolutionary_sampling(self, n_samples: int = 1000):
        """Perform evolutionary sampling."""
        print(f"Performing evolutionary sampling ({n_samples} samples)...")

        # Monte Carlo sampling
        mc_samples = self.sampler.sample(n_samples=n_samples // 2, method='monte-carlo')

        # Importance sampling
        importance_samples = self.sampler.sample(n_samples=n_samples // 2, method='importance-sampling')

        print(f"Generated {len(mc_samples.samples)} MC samples and {len(importance_samples.samples)} importance samples")
        return mc_samples, importance_samples

    def analyze_evolutionary_patterns(self):
        """Analyze evolutionary patterns."""
        print("Analyzing evolutionary patterns...")

        # Analyze patterns
        patterns = self.sampler.analyze_evolutionary_patterns()

        # Extract key metrics
        pop_stats = patterns['population_statistics']
        genetic_params = patterns['genetic_parameters']

        print("Population Statistics:")
        print(f"  Effective population size: {pop_stats.effective_population_size".0f"}")
        print(f"  Mean heritability: {np.mean(list(pop_stats.heritability_estimates.values()))".3f"}")

        print("Genetic Parameters:")
        print(f"  Additive variance: {genetic_params['additive_variance']".3f"}")
        print(f"  Environmental variance: {genetic_params['environmental_variance']".3f"}")

        return patterns

    def perform_comparative_analysis(self):
        """Perform comparative evolutionary analysis."""
        print("Performing comparative analysis...")

        # Time series analysis
        ts_results = self.analytics.analyze_time_series()
        print(f"Detected {len(ts_results.change_points)} change points")
        print(f"Trend analysis completed for {len(ts_results.trend_analysis)} variables")

        # Multivariate analysis
        mv_results = self.analytics.analyze_multivariate()
        pca_results = mv_results['principal_components']
        print(f"PCA explained variance: {pca_results['explained_variance_ratio'][:3]}")

        # Predictive modeling
        predictions = self.analytics.predictive_modeling(
            target_variable='phenotype_final',
            feature_variables=['phenotype_initial', 'genotype_score']
        )
        print(f"Predictive modeling R²: {predictions['random_forest'].performance_metrics['test_r2']:".3f"")

        return ts_results, mv_results, predictions

    def cluster_trajectories(self, n_clusters: int = 3):
        """Cluster developmental trajectories."""
        print(f"Clustering trajectories into {n_clusters} groups...")

        clusters = self.sampler.cluster_individuals(n_clusters=n_clusters)

        print("Clustering Results:")
        for cluster_name, stats in clusters['cluster_statistics'].items():
            print(f"  {cluster_name}: {stats['size']} individuals")

        return clusters

    def run_complete_evolutionary_analysis(self):
        """Run complete evolutionary analysis."""
        print("Running complete evolutionary analysis...")
        print("=" * 60)

        # Step 1: Setup
        self.setup_sampling_and_analysis()

        # Step 2: Sampling
        mc_samples, importance_samples = self.perform_evolutionary_sampling()

        # Step 3: Pattern analysis
        patterns = self.analyze_evolutionary_patterns()

        # Step 4: Comparative analysis
        ts_results, mv_results, predictions = self.perform_comparative_analysis()

        # Step 5: Clustering
        clusters = self.cluster_trajectories()

        print("=" * 60)
        print("Evolutionary analysis complete!")
        return {
            'samples': {'monte_carlo': mc_samples, 'importance': importance_samples},
            'patterns': patterns,
            'time_series': ts_results,
            'multivariate': mv_results,
            'predictions': predictions,
            'clusters': clusters
        }


# Example 3: Batch Analysis Orchestrator
class BatchAnalysisOrchestrator:
    """Thin orchestrator for batch analysis of multiple datasets."""

    def __init__(self, data_directory: Path):
        """Initialize with data directory."""
        self.data_directory = data_directory
        self.datasets = []
        self.results = {}

    def discover_datasets(self):
        """Discover all datasets in the directory."""
        print("Discovering datasets...")

        # Find all CSV files
        csv_files = list(self.data_directory.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        self.datasets = csv_files
        return csv_files

    def analyze_single_dataset(self, data_file: Path, dataset_name: str):
        """Analyze a single dataset."""
        print(f"Analyzing dataset: {dataset_name}")

        try:
            # Load and preprocess
            data_core = ej.DataCore.load_from_csv(data_file, time_column='time')
            data_core.preprocess_data()

            # Fit model
            model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')

            # Analyze cross-sections
            analyzer = ej.LaserPlaneAnalyzer(model)
            cross_sections = analyzer.analyze_cross_section(time_point=10.0)

            # Store results
            result = {
                'dataset': dataset_name,
                'n_samples': len(data_core.time_series_data[0].data),
                'model_params': model.fitted_parameters,
                'cross_section_mean': cross_sections.moments['mean'],
                'cross_section_std': cross_sections.moments['std'],
                'status': 'success'
            }

            print(f"  ✓ {dataset_name}: {result['n_samples']} samples, "
                  f"mean = {result['cross_section_mean']".2f"}")

        except Exception as e:
            print(f"  ✗ {dataset_name}: {str(e)}")
            result = {
                'dataset': dataset_name,
                'error': str(e),
                'status': 'failed'
            }

        return result

    def run_batch_analysis(self, max_datasets: int = None):
        """Run batch analysis on all datasets."""
        print("Running batch analysis...")
        print("=" * 60)

        datasets = self.discover_datasets()

        if max_datasets:
            datasets = datasets[:max_datasets]

        results = []

        for i, data_file in enumerate(datasets, 1):
            dataset_name = data_file.stem
            print(f"[{i}/{len(datasets)}] Processing {dataset_name}...")

            result = self.analyze_single_dataset(data_file, dataset_name)
            results.append(result)

        # Summarize results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        print("=" * 60)
        print("Batch Analysis Summary:")
        print(f"  Total datasets: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            means = [r['cross_section_mean'] for r in successful]
            print(f"  Mean cross-section values: {np.mean(means)".2f"} ± {np.std(means):".2f"")
            print(f"  Range: {np.min(means)".2f"} - {np.max(means)".2f"}")

        self.results = {
            'summary': {
                'total': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'mean_cross_section': np.mean([r['cross_section_mean'] for r in successful]) if successful else None
            },
            'results': results
        }

        return self.results


# Example 4: Interactive Exploration Orchestrator
class InteractiveExplorationOrchestrator:
    """Thin orchestrator for interactive data exploration."""

    def __init__(self, data_file: Path):
        """Initialize with data file."""
        self.data_file = data_file
        self.data_core = None
        self.model = None
        self.current_analysis = {}

    def load_data(self):
        """Load data for exploration."""
        print("Loading data for interactive exploration...")
        self.data_core = ej.DataCore.load_from_csv(self.data_file, time_column='time')
        print(f"Loaded {len(self.data_core.time_series_data[0].data)} data points")
        return self.data_core

    def explore_parameter_space(self, model_types: list = None):
        """Explore different model parameter spaces."""
        if model_types is None:
            model_types = ['jump-diffusion', 'ornstein-uhlenbeck', 'geometric-jump-diffusion']

        print("Exploring parameter space across model types...")

        results = {}

        for model_type in model_types:
            print(f"  Fitting {model_type} model...")
            model = ej.JumpRope.fit(self.data_core, model_type=model_type)

            # Analyze model characteristics
            model.generate_trajectories(n_samples=100)
            analyzer = ej.LaserPlaneAnalyzer(model)

            # Get characteristics at different time points
            time_points = [5.0, 10.0, 15.0, 20.0]
            characteristics = {}

            for time_point in time_points:
                cross_section = analyzer.analyze_cross_section(time_point)
                characteristics[time_point] = {
                    'mean': cross_section.moments['mean'],
                    'std': cross_section.moments['std'],
                    'distribution': cross_section.distribution_fit.get('distribution', 'unknown')
                }

            results[model_type] = {
                'parameters': model.fitted_parameters,
                'characteristics': characteristics,
                'jump_times': model.estimate_jump_times()
            }

            print(f"    {model_type}: {len(results[model_type]['jump_times'])} estimated jumps")

        self.current_analysis['parameter_exploration'] = results
        return results

    def sensitivity_analysis(self, parameter_ranges: dict = None):
        """Perform sensitivity analysis."""
        if parameter_ranges is None:
            parameter_ranges = {
                'jump_intensity': [0.01, 0.05, 0.1, 0.2],
                'reversion_speed': [0.1, 0.5, 1.0, 2.0]
            }

        print("Performing sensitivity analysis...")

        base_model = ej.JumpRope.fit(self.data_core, model_type='jump-diffusion')
        base_model.generate_trajectories(n_samples=50)

        sensitivity_results = {}

        for param_name, param_values in parameter_ranges.items():
            print(f"  Analyzing sensitivity to {param_name}...")
            param_results = []

            for param_value in param_values:
                # Create modified parameters
                modified_params = ej.ModelParameters(
                    drift=base_model.fitted_parameters.drift,
                    diffusion=base_model.fitted_parameters.diffusion,
                    jump_intensity=getattr(base_model.fitted_parameters, param_name, param_value),
                    jump_mean=base_model.fitted_parameters.jump_mean,
                    jump_std=base_model.fitted_parameters.jump_std,
                    equilibrium=base_model.fitted_parameters.equilibrium,
                    reversion_speed=base_model.fitted_parameters.reversion_speed
                )

                # Update parameter if it exists
                if hasattr(base_model.fitted_parameters, param_name):
                    setattr(modified_params, param_name, param_value)

                # Create new process with modified parameters
                from evojump.jumprope import OrnsteinUhlenbeckJump
                modified_process = OrnsteinUhlenbeckJump(modified_params)

                # Create temporary model
                temp_model = ej.JumpRope(
                    modified_process,
                    base_model.time_points,
                    base_model.initial_conditions
                )
                temp_model.fitted_parameters = modified_params
                temp_model.generate_trajectories(n_samples=50)

                # Analyze
                analyzer = ej.LaserPlaneAnalyzer(temp_model)
                cross_section = analyzer.analyze_cross_section(time_point=10.0)

                param_results.append({
                    'parameter_value': param_value,
                    'mean': cross_section.moments['mean'],
                    'std': cross_section.moments['std']
                })

            sensitivity_results[param_name] = param_results

        self.current_analysis['sensitivity_analysis'] = sensitivity_results
        return sensitivity_results

    def create_exploration_report(self, output_file: Path):
        """Create exploration report."""
        print("Creating exploration report...")

        report = {
            'data_summary': {
                'n_samples': len(self.data_core.time_series_data[0].data),
                'time_range': {
                    'min': float(self.data_core.time_series_data[0].data['time'].min()),
                    'max': float(self.data_core.time_series_data[0].data['time'].max())
                }
            },
            'parameter_exploration': self.current_analysis.get('parameter_exploration', {}),
            'sensitivity_analysis': self.current_analysis.get('sensitivity_analysis', {})
        }

        # Save report
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Exploration report saved to {output_file}")
        return report


# Example Usage
def main():
    """Demonstrate the thin orchestrator examples."""
    print("EvoJump Thin Orchestrator Examples")
    print("=" * 50)

    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'time': list(range(1, 21)) * 10,
        'phenotype1': [10 + i + np.random.normal(0, 0.5) for i in range(20) for _ in range(10)],
        'phenotype2': [20 + i * 1.5 + np.random.normal(0, 1.0) for i in range(20) for _ in range(10)]
    })

    data_file = Path("sample_data.csv")
    sample_data.to_csv(data_file, index=False)

    try:
        # Example 1: Developmental Trajectory Analysis
        print("\n1. Developmental Trajectory Analysis Orchestrator")
        print("-" * 50)
        trajectory_orchestrator = DevelopmentalTrajectoryOrchestrator(data_file)
        trajectory_results = trajectory_orchestrator.run_complete_analysis(Path("trajectory_results"))

        # Example 2: Evolutionary Analysis
        print("\n2. Evolutionary Analysis Orchestrator")
        print("-" * 50)
        evolutionary_orchestrator = EvolutionaryAnalysisOrchestrator(data_file)
        evolutionary_results = evolutionary_orchestrator.run_complete_evolutionary_analysis()

        # Example 3: Batch Analysis
        print("\n3. Batch Analysis Orchestrator")
        print("-" * 50)
        batch_orchestrator = BatchAnalysisOrchestrator(Path("."))
        batch_results = batch_orchestrator.run_batch_analysis(max_datasets=1)

        # Example 4: Interactive Exploration
        print("\n4. Interactive Exploration Orchestrator")
        print("-" * 50)
        exploration_orchestrator = InteractiveExplorationOrchestrator(data_file)
        exploration_orchestrator.load_data()
        param_results = exploration_orchestrator.explore_parameter_space()
        sensitivity_results = exploration_orchestrator.sensitivity_analysis()
        report = exploration_orchestrator.create_exploration_report(Path("exploration_report.json"))

        print("\n" + "=" * 50)
        print("All orchestrator examples completed successfully!")
        print("\nKey Benefits of Thin Orchestrators:")
        print("• Clean separation of concerns")
        print("• Reusable analysis components")
        print("• Easy to test and maintain")
        print("• Flexible parameter configuration")
        print("• Comprehensive result reporting")

    finally:
        # Clean up
        if data_file.exists():
            data_file.unlink()
        print(f"\nCleaned up temporary files")

if __name__ == '__main__':
    main()
