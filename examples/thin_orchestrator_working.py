#!/usr/bin/env python3
"""
Thin Orchestrator Examples for EvoJump Package

This file demonstrates various thin orchestrator patterns for complex
analysis workflows in the EvoJump package.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '/Users/4d/Documents/GitHub/EvoJump/src')
from evojump.datacore import DataCore
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer
from evojump.trajectory_visualizer import TrajectoryVisualizer
from evojump.evolution_sampler import EvolutionSampler
from evojump.analytics_engine import AnalyticsEngine
import matplotlib.pyplot as plt
import json

class DevelopmentalTrajectoryOrchestrator:
    """Orchestrates complete developmental trajectory analysis."""

    def __init__(self, data_path: str):
        """Initialize with data path."""
        self.data_path = data_path
        self.data_core = None
        self.model = None
        self.analyzer = None
        self.results = {}

    def run_complete_analysis(self) -> dict:
        """Run complete developmental trajectory analysis."""
        print("🚀 Running Complete Developmental Trajectory Analysis")
        print("=" * 60)

        # Step 1: Load and preprocess data
        print("📊 Loading and preprocessing data...")
        self.data_core = DataCore.load_from_csv(
            file_path=self.data_path,
            time_column='time',
            phenotype_columns=['phenotype']
        )

        self.data_core.preprocess_data(
            normalize=True,
            remove_outliers=True,
            interpolate_missing=True
        )

        quality = self.data_core.validate_data_quality()
        print(f"   ✅ Data quality - Missing: {quality['missing_data_percentage']['dataset_0']:.2f}%, "
              f"Outliers: {quality['outlier_percentage']['dataset_0']:.2f}%")

        # Step 2: Fit stochastic model
        print("🧮 Fitting stochastic model...")
        time_points = np.sort(pd.read_csv(self.data_path)['time'].unique())
        self.model = JumpRope.fit(self.data_core, model_type='jump-diffusion', time_points=time_points)

        print(f"   ✅ Model fitted - Equilibrium: {self.model.fitted_parameters.equilibrium:.3f}, "
              f"Reversion: {self.model.fitted_parameters.reversion_speed:.3f}")

        # Step 3: Cross-sectional analysis
        print("🔬 Performing cross-sectional analysis...")
        self.analyzer = LaserPlaneAnalyzer(self.model)

        stages = [5.0, 10.0, 15.0, 20.0]
        cross_section_results = {}

        for stage in stages:
            result = self.analyzer.analyze_cross_section(time_point=stage)
            cross_section_results[stage] = {
                'mean': float(result.moments['mean']),
                'std': float(result.moments['std']),
                'distribution': result.distribution_fit.get('distribution', 'unknown')
            }
            print(f"   📊 Stage {stage:4.1f}: {result.distribution_fit.get('distribution', 'normal')} "
                  f"(mean={result.moments['mean']:.2f}, std={result.moments['std']:.2f})")

        # Step 4: Generate outputs
        print("🎨 Generating visualizations...")
        visualizer = TrajectoryVisualizer()
        output_dir = Path("trajectory_analysis_outputs")
        output_dir.mkdir(exist_ok=True)

        # Generate plots
        fig = visualizer.plot_trajectories(self.model, n_trajectories=15, interactive=False)
        fig.savefig(output_dir / 'developmental_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig = visualizer.plot_cross_sections(self.model, time_points=stages, interactive=False)
        fig.savefig(output_dir / 'cross_sections.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig = visualizer.plot_landscapes(self.model, interactive=False)
        fig.savefig(output_dir / 'phenotypic_landscape.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("   ✅ Generated trajectory plots, cross-sections, and 3D landscape")

        # Step 5: Compile results
        self.results = {
            'analysis_type': 'developmental_trajectory',
            'data_quality': quality,
            'model_parameters': {
                'equilibrium': float(self.model.fitted_parameters.equilibrium),
                'reversion_speed': float(self.model.fitted_parameters.reversion_speed),
                'jump_intensity': float(self.model.fitted_parameters.jump_intensity),
                'diffusion': float(self.model.fitted_parameters.diffusion)
            },
            'cross_section_analysis': cross_section_results,
            'outputs_generated': {
                'trajectory_plot': 'developmental_trajectories.png',
                'cross_section_plot': 'cross_sections.png',
                'landscape_plot': 'phenotypic_landscape.png'
            },
            'summary': f"Successfully analyzed developmental trajectories with {len(stages)} cross-sections"
        }

        # Save results
        results_file = output_dir / 'trajectory_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print("   ✅ Saved comprehensive analysis results")
        print(f"\n🎉 Complete analysis finished! Check '{output_dir}' for all outputs")

        return self.results

class EvolutionaryAnalysisOrchestrator:
    """Orchestrates comprehensive evolutionary analysis."""

    def __init__(self, data_path: str):
        """Initialize with data path."""
        self.data_path = data_path
        self.data_core = None
        self.sampler = None
        self.analytics = None
        self.results = {}

    def run_evolutionary_analysis(self) -> dict:
        """Run comprehensive evolutionary analysis."""
        print("🧬 Running Comprehensive Evolutionary Analysis")
        print("=" * 60)

        # Step 1: Load data
        print("📊 Loading evolutionary data...")
        self.data_core = DataCore.load_from_csv(
            file_path=self.data_path,
            time_column='time',
            phenotype_columns=['phenotype']
        )

        # Step 2: Initialize components
        print("🔧 Initializing analysis components...")
        self.sampler = EvolutionSampler(self.data_core)
        self.analytics = AnalyticsEngine(self.data_core)

        # Step 3: Evolutionary sampling
        print("🎲 Performing evolutionary sampling...")
        mc_samples = self.sampler.sample(n_samples=200, method='monte-carlo')
        print(f"   ✅ Generated {mc_samples.samples.shape[0]} Monte Carlo samples")

        # Step 4: Population analysis
        print("👥 Analyzing population patterns...")
        evolution_analysis = self.sampler.analyze_evolutionary_patterns()

        pop_stats = evolution_analysis['population_statistics']
        genetic_params = evolution_analysis['genetic_parameters']

        print(f"   ✅ Estimated effective population size: {pop_stats.effective_population_size:.0f}")
        print(f"   ✅ Mean heritability: {np.mean(list(pop_stats.heritability_estimates.values())):.3f}")

        # Step 5: Time series analysis
        print("⏰ Performing time series analysis...")
        ts_results = self.analytics.analyze_time_series()
        print(f"   ✅ Detected {len(ts_results.change_points)} change points")

        # Step 6: Multivariate analysis
        print("📈 Performing multivariate analysis...")
        mv_results = self.analytics.analyze_multivariate()
        pca_results = mv_results['principal_components']
        print(f"   ✅ PCA completed with {len(pca_results['explained_variance_ratio'])} components")
        print(f"   ✅ Explained variance: {pca_results['explained_variance_ratio'][:3]}")

        # Step 7: Compile comprehensive results
        self.results = {
            'analysis_type': 'evolutionary_comprehensive',
            'evolutionary_sampling': {
                'n_samples': mc_samples.samples.shape[0],
                'method': 'monte-carlo',
                'dimensions': mc_samples.samples.shape[1:]
            },
            'population_statistics': {
                'effective_population_size': float(pop_stats.effective_population_size),
                'mean_heritability': float(np.mean(list(pop_stats.heritability_estimates.values()))),
                'heritability_estimates': {k: float(v) for k, v in pop_stats.heritability_estimates.items()},
                'selection_gradients': {k: float(v) for k, v in pop_stats.selection_gradients.items()}
            },
            'genetic_parameters': {k: float(v) for k, v in genetic_params.items()},
            'time_series_analysis': {
                'change_points_count': len(ts_results.change_points),
                'forecasts_available': list(ts_results.forecasts.keys()),
                'stationarity_results': {k: bool(v) for k, v in ts_results.model_fit['stationarity'].items()}
            },
            'multivariate_analysis': {
                'pca_components': len(pca_results['explained_variance_ratio']),
                'explained_variance': pca_results['explained_variance_ratio'][:5].tolist()
            },
            'summary': "Comprehensive evolutionary analysis completed with population genetics, time series, and multivariate methods"
        }

        # Save results
        output_dir = Path("evolutionary_analysis_outputs")
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / 'evolutionary_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print("   ✅ Saved comprehensive evolutionary analysis results")
        print(f"\n🎉 Evolutionary analysis finished! Check '{output_dir}' for all outputs")

        return self.results

def main():
    """Main function demonstrating thin orchestrator patterns."""
    print("🚀 EvoJump Thin Orchestrator Examples")
    print("=" * 70)
    print("This demo showcases advanced analysis orchestration patterns")
    print("=" * 70)

    # Create sample data for demonstrations
    print("\n📝 Creating sample data for demonstrations...")

    np.random.seed(42)
    n_individuals = 75
    time_points = np.linspace(0, 25, 26)

    data_rows = []
    for individual in range(n_individuals):
        base_pattern = 10 + 4 * np.sin(time_points * 0.25) + time_points * 0.15
        individual_variation = np.random.normal(0, 1.2, len(time_points))
        phenotype = base_pattern + individual_variation

        for t_idx, time_point in enumerate(time_points):
            data_rows.append({
                'individual': f'ind_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx],
                'genotype': np.random.choice(['AA', 'Aa', 'aa'], p=[0.3, 0.5, 0.2])
            })

    sample_data = pd.DataFrame(data_rows)
    data_file = Path("orchestrator_demo_data.csv")
    sample_data.to_csv(data_file, index=False)
    print(f"   ✅ Created sample data: {data_file}")

    # Example 1: Developmental Trajectory Orchestrator
    print("\n" + "="*70)
    trajectory_orchestrator = DevelopmentalTrajectoryOrchestrator(data_file)
    trajectory_results = trajectory_orchestrator.run_complete_analysis()

    # Example 2: Evolutionary Analysis Orchestrator
    print("\n" + "="*70)
    evolutionary_orchestrator = EvolutionaryAnalysisOrchestrator(data_file)
    evolutionary_results = evolutionary_orchestrator.run_evolutionary_analysis()

    # Summary
    print("\n" + "="*70)
    print("🎉 Thin Orchestrator Examples - COMPLETED!")
    print("="*70)

    print("📊 Summary of Generated Outputs:")
    print("   • Developmental trajectory analysis with cross-sections")
    print("   • Comprehensive evolutionary analysis with population genetics")
    print("   • Advanced statistical methods (PCA, time series, change detection)")
    print("   • Multiple visualization types (trajectories, landscapes, cross-sections)")
    print("")
    print("   • evolutionary_analysis_outputs/")

    print("
🔬 Key Scientific Results:"    print("   • Stochastic model fitting with proper parameter estimation"    print(f"   • Cross-sectional analysis at {len([5.0, 10.0, 15.0, 20.0])} developmental stages"    print("   • Population genetic analysis with heritability estimation"    print("   • Multivariate analysis with dimensionality reduction"    print("   • Time series analysis with change point detection"

    # Clean up
    data_file.unlink()
    print("
🧹 Cleaned up temporary files"    print("\n" + "="*70)

    return {
        'trajectory_analysis': trajectory_results,
        'evolutionary_analysis': evolutionary_results,
        'status': 'completed'
    }

if __name__ == '__main__':
    results = main()
    print(f"\n✅ All thin orchestrator examples completed successfully!")
    print(f"📊 Status: {results['status']}")
    print(f"📈 Generated {len(results)} comprehensive analysis results")
