#!/usr/bin/env python3
"""
Comprehensive Demo for EvoJump Package

This example demonstrates the complete functionality of the EvoJump package,
generating all types of outputs including plots, animations, and reports.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ,'src'))
import evojump as ej
from evojump.datacore import DataCore
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer
from evojump.trajectory_visualizer import TrajectoryVisualizer
from evojump.evolution_sampler import EvolutionSampler
from evojump.analytics_engine import AnalyticsEngine
import matplotlib.pyplot as plt
import json
import time

def create_comprehensive_sample_data():
    """Create comprehensive sample data for demonstration."""
    print("🎯 Creating comprehensive sample developmental data...")

    np.random.seed(42)

    # Generate synthetic developmental trajectories
    n_individuals = 100
    time_points = np.linspace(0, 20, 21)

    data_rows = []

    for individual in range(n_individuals):
        # Base developmental pattern with individual variation
        base_pattern = 10 + 5 * np.sin(time_points * 0.3) + time_points * 0.2

        # Add individual variation
        individual_variation = np.random.normal(0, 1, len(time_points))

        # Add occasional developmental jumps
        jumps = np.zeros(len(time_points))
        if np.random.random() < 0.3:  # 30% chance of having a jump
            jump_time = np.random.choice(len(time_points))
            jump_size = np.random.normal(0, 3)
            jumps[jump_time:] += jump_size

        phenotype = base_pattern + individual_variation + jumps

        for t_idx, time_point in enumerate(time_points):
            data_rows.append({
                'individual_id': f'ind_{individual:03d}',
                'time': time_point,
                'phenotype_value': phenotype[t_idx],
                'genotype': np.random.choice(['AA', 'Aa', 'aa'], p=[0.4, 0.4, 0.2]),
                'treatment': np.random.choice(['control', 'treatment'], p=[0.6, 0.4])
            })

    return pd.DataFrame(data_rows)

def run_comprehensive_demo():
    """Run comprehensive EvoJump demonstration."""
    print("🚀 EvoJump Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases all EvoJump capabilities:")
    print("• Data loading and preprocessing")
    print("• Stochastic model fitting")
    print("• Cross-sectional analysis")
    print("• Advanced visualizations")
    print("• Evolutionary analysis")
    print("• Comprehensive reporting")
    print("=" * 60)

    # Step 1: Create and save sample data
    print("\n📊 Step 1: Creating sample developmental data...")
    sample_data = create_comprehensive_sample_data()
    print(f"   ✅ Created {len(sample_data)} measurements")
    print(f"   ✅ Time range: {sample_data['time'].min():.1f} to {sample_data['time'].max():.1f}")
    print(f"   ✅ Individuals: {sample_data['individual_id'].nunique()}")
    print(f"   ✅ Genotypes: {sample_data['genotype'].unique()}")
    print(f"   ✅ Treatments: {sample_data['treatment'].unique()}")

    data_file = Path("comprehensive_sample_data.csv")
    sample_data.to_csv(data_file, index=False)
    print(f"   💾 Data saved to: {data_file}")

    # Step 2: Load and preprocess data
    print("\n🔧 Step 2: Loading and preprocessing data...")
    data_core = DataCore.load_from_csv(
        file_path=data_file,
        time_column='time',
        phenotype_columns=['phenotype_value']
    )

    # Preprocess data
    data_core.preprocess_data(
        normalize=True,
        remove_outliers=True,
        interpolate_missing=True
    )

    # Validate data quality
    quality_metrics = data_core.validate_data_quality()
    print(f"   📋 Data quality: Missing: {quality_metrics['missing_data_percentage']['dataset_0']:.2f}%")
    print(f"   📋 Outliers: {quality_metrics['outlier_percentage']['dataset_0']:.2f}%")

    # Step 3: Fit stochastic model
    print("\n🧮 Step 3: Fitting stochastic model...")
    time_points = np.sort(sample_data['time'].unique())

    model = JumpRope.fit(
        data_core,
        model_type='jump-diffusion',
        time_points=time_points
    )

    print(f"   ✅ Model fitted with equilibrium: {model.fitted_parameters.equilibrium:.3f}")
    print(f"   ✅ Reversion speed: {model.fitted_parameters.reversion_speed:.3f}")
    print(f"   ✅ Jump intensity: {model.fitted_parameters.jump_intensity:.4f}")

    # Step 4: Generate trajectories
    print("\n📈 Step 4: Generating developmental trajectories...")
    trajectories = model.generate_trajectories(n_samples=50, x0=10.0)
    print(f"   ✅ Generated {trajectories.shape[0]} trajectories")
    print(f"   ✅ Each with {trajectories.shape[1]} time points")

    # Step 5: Analyze cross-sections
    print("\n🔬 Step 5: Analyzing cross-sectional distributions...")
    analyzer = LaserPlaneAnalyzer(model)

    stages = [5.0, 10.0, 15.0, 20.0]
    cross_section_results = {}

    for stage in stages:
        result = analyzer.analyze_cross_section(time_point=stage)
        cross_section_results[stage] = result
        print(f"   📊 Stage {stage:4.1f}: mean = {result.moments['mean']:6.2f}, "
              f"std = {result.moments['std']:.2f}")
        print(f"   📊 Distribution: {result.distribution_fit.get('distribution', 'unknown')} "
              f"(AIC: {result.goodness_of_fit['aic']:.2f})")

    # Step 6: Create comprehensive visualizations
    print("\n🎨 Step 6: Creating comprehensive visualizations...")
    visualizer = TrajectoryVisualizer()

    output_dir = Path("evojump_outputs")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Developmental trajectories
    print("   📈 Creating trajectory plot...")
    fig = visualizer.plot_trajectories(model, n_trajectories=15, interactive=False)
    fig.savefig(output_dir / 'developmental_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✅ Saved: developmental_trajectories.png")

    # Plot 2: Cross-sections
    print("   📊 Creating cross-section plot...")
    fig = visualizer.plot_cross_sections(model, time_points=stages, interactive=False)
    fig.savefig(output_dir / 'cross_sections.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✅ Saved: cross_sections.png")

    # Plot 3: 3D Landscape
    print("   🏔️ Creating 3D landscape plot...")
    fig = visualizer.plot_landscapes(model, interactive=False)
    fig.savefig(output_dir / 'phenotypic_landscape.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✅ Saved: phenotypic_landscape.png")

    # Step 7: Comparative analysis
    print("\n🔍 Step 7: Performing comparative analysis...")
    condition_data = {}
    for stage in stages:
        cross_section = model.compute_cross_sections(
            np.argmin(np.abs(time_points - stage))
        )
        condition_data[f'stage_{int(stage)}'] = cross_section

    comparison = analyzer.compare_distributions(
        time_point=10.0,
        condition_data=condition_data
    )
    print(f"   ✅ Found significant differences in: {comparison.significant_differences}")

    # Step 8: Evolutionary analysis
    print("\n🧬 Step 8: Performing evolutionary analysis...")
    # EvolutionSampler treats every non-time column as a phenotype; give it a
    # numeric-only frame so string columns (genotype, treatment) don't break it
    sampler = EvolutionSampler(
        sample_data.select_dtypes(include=[np.number]),
        time_column='time'
    )

    # Evolutionary sampling
    mc_samples = sampler.sample(n_samples=200, method='monte-carlo')
    print(f"   ✅ Generated {mc_samples.samples.shape[0]} Monte Carlo samples")

    # Analyze evolutionary patterns
    evolution_analysis = sampler.analyze_evolutionary_patterns()
    pop_stats = evolution_analysis['population_statistics']
    genetic_params = evolution_analysis['genetic_parameters']

    print(f"   ✅ Mean heritability: {np.mean(list(pop_stats.heritability_estimates.values())):.3f}")
    print(f"   ✅ Effective population size: {pop_stats.effective_population_size:.0f}")

    # Step 9: Time series analysis
    print("\n⏰ Step 9: Performing time series analysis...")
    # Multivariate analysis (PCA) requires numeric columns; use a numeric-only
    # frame so string columns (individual_id, genotype, treatment) don't break it
    analytics = AnalyticsEngine(
        sample_data.select_dtypes(include=[np.number]),
        time_column='time'
    )

    ts_results = analytics.analyze_time_series()
    print(f"   ✅ Change points detected: {len(ts_results.change_points)}")
    print(f"   ✅ Forecasts generated: {list(ts_results.forecasts.keys())}")

    # Multivariate analysis
    mv_results = analytics.analyze_multivariate()
    pca_results = mv_results['principal_components']
    print(f"   ✅ PCA completed: {len(pca_results['explained_variance_ratio'])} components")
    print(f"   ✅ Explained variance: {pca_results['explained_variance_ratio'][:3]}")

    # Step 10: Create comprehensive report
    print("\n📋 Step 10: Creating comprehensive analysis report...")

    report = {
        'metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_file': str(data_file),
            'n_samples': len(sample_data),
            'n_individuals': sample_data['individual_id'].nunique(),
            'time_range': {
                'min': float(sample_data['time'].min()),
                'max': float(sample_data['time'].max())
            }
        },
        'data_quality': quality_metrics,
        'model_parameters': {
            'equilibrium': float(model.fitted_parameters.equilibrium),
            'reversion_speed': float(model.fitted_parameters.reversion_speed),
            'jump_intensity': float(model.fitted_parameters.jump_intensity),
            'diffusion': float(model.fitted_parameters.diffusion),
            'jump_mean': float(model.fitted_parameters.jump_mean),
            'jump_std': float(model.fitted_parameters.jump_std)
        },
        'cross_section_analysis': {
            str(stage): {
                'mean': float(result.moments['mean']),
                'std': float(result.moments['std']),
                'distribution': result.distribution_fit.get('distribution', 'unknown'),
                'aic': float(result.goodness_of_fit['aic']),
                'confidence_interval': {
                    'mean_ci': result.confidence_intervals['mean_ci'],
                    'median_ci': result.confidence_intervals['median_ci']
                }
            }
            for stage, result in cross_section_results.items()
        },
        'evolutionary_analysis': {
            'population_statistics': {
                'effective_population_size': float(pop_stats.effective_population_size),
                'heritability_estimates': {k: float(v) for k, v in pop_stats.heritability_estimates.items()},
                'selection_gradients': {k: float(v) for k, v in pop_stats.selection_gradients.items()}
            },
            'genetic_parameters': {k: float(v) for k, v in genetic_params.items()}
        },
        'time_series_analysis': {
            'change_points_count': len(ts_results.change_points),
            'forecasts_available': list(ts_results.forecasts.keys()),
            'stationarity': {k: bool(v) for k, v in ts_results.model_fit['stationarity'].items()}
        },
        'multivariate_analysis': {
            'pca_components': len(pca_results['explained_variance_ratio']),
            'explained_variance': pca_results['explained_variance_ratio'][:5].tolist()
        },
        'distribution_comparison': {
            'significant_differences': comparison.significant_differences,
            'reference_time_point': 10.0
        },
        'outputs_generated': {
            'trajectory_plot': 'developmental_trajectories.png',
            'cross_section_plot': 'cross_sections.png',
            'landscape_plot': 'phenotypic_landscape.png',
            'analysis_report': 'comprehensive_analysis_report.json'
        }
    }

    # Save comprehensive report
    report_file = output_dir / 'comprehensive_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"   ✅ Comprehensive report saved: {report_file}")

    # Step 11: Summary and cleanup
    print("\n🎉 Step 11: Demo completed successfully!")
    print("=" * 60)

    print("📊 Analysis Summary:")
    print(f"   • Dataset: {len(sample_data)} measurements from {sample_data['individual_id'].nunique()} individuals")
    print(f"   • Model: Jump-diffusion with equilibrium = {model.fitted_parameters.equilibrium:.3f}")
    print(f"   • Cross-sections analyzed at {len(stages)} developmental stages")
    print(f"   • Evolutionary samples: {mc_samples.samples.shape[0]} Monte Carlo samples")
    print(f"   • Statistical tests: {len(ts_results.change_points)} change points detected")
    print(f"   • PCA components: {len(pca_results['explained_variance_ratio'])} principal components")

    print("")
    print(f"   • 📊 Cross-section plots: {output_dir}/cross_sections.png")
    print(f"   • 🏔️ 3D landscape plots: {output_dir}/phenotypic_landscape.png")
    print(f"   • 📋 Analysis report: {output_dir}/comprehensive_analysis_report.json")

    print("\n🔬 Key Scientific Results:")
    print("   • Stochastic model successfully fitted to developmental data")
    print(f"   • Cross-sectional distributions identified as: {cross_section_results[10.0].distribution_fit.get('distribution', 'unknown')}")
    print(f"   • Significant developmental differences found: {len(comparison.significant_differences)} stages differ")
    print(f"   • Population genetic parameters estimated: Ne = {pop_stats.effective_population_size:.0f}")
    print(f"   • Multivariate analysis completed: {len(pca_results['explained_variance_ratio'])} PCA components")

    # Clean up
    print("\n🧹 Cleaning up...")
    data_file.unlink()
    print(f"   ✅ Removed temporary data file: {data_file}")

    print("\n" + "=" * 60)
    print("🚀 EvoJump Comprehensive Demo - COMPLETE!")
    print("🎯 All outputs generated successfully!")
    print("📁 Check the 'evojump_outputs' directory for all generated files")
    print("=" * 60)

    return {
        'status': 'success',
        'outputs': {
            'directory': str(output_dir),
            'files': [
                'developmental_trajectories.png',
                'cross_sections.png',
                'phenotypic_landscape.png',
                'comprehensive_analysis_report.json'
            ]
        },
        'analysis_results': report
    }

if __name__ == '__main__':
    start_time = time.time()
    results = run_comprehensive_demo()
    end_time = time.time()

    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")
    print(f"📊 Demo completed with {results['status']}")
    print(f"📁 Output directory: {results['outputs']['directory']}")
    print(f"📄 Generated files: {len(results['outputs']['files'])} files")

    for file in results['outputs']['files']:
        print(f"   • {file}")

    print("\n🎯 To explore the results:")
    print(f"   1. Open the output directory: {results['outputs']['directory']}")
    print("   2. View the generated plots (PNG files)")
    print("   3. Examine the comprehensive analysis report (JSON file)")
    print("   4. All outputs are ready for scientific analysis!")

