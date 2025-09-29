#!/usr/bin/env python3
"""
Basic Usage Example for EvoJump Package

This example demonstrates the basic usage of the EvoJump package for evolutionary
ontogenetic analysis, including data loading, model fitting, cross-sectional analysis,
and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import evojump as ej

def create_sample_data():
    """Create sample developmental data for demonstration."""
    np.random.seed(42)

    # Generate synthetic developmental trajectories
    n_individuals = 50
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
                'individual': f'ind_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx],
                'genotype': np.random.choice(['AA', 'Aa', 'aa'], p=[0.4, 0.4, 0.2])
            })

    return pd.DataFrame(data_rows)

def main():
    """Main example function."""
    print("EvoJump Basic Usage Example")
    print("=" * 50)

    # Create sample data
    print("1. Creating sample developmental data...")
    sample_data = create_sample_data()
    print(f"   Created data with {len(sample_data)} measurements")
    print(f"   Time range: {sample_data['time'].min():.1f} to {sample_data['time'].max():.1f}")
    print(f"   Number of individuals: {sample_data['individual'].nunique()}")

    # Save sample data
    data_file = Path("sample_developmental_data.csv")
    sample_data.to_csv(data_file, index=False)
    print(f"   Data saved to: {data_file}")

    # Load data using DataCore
    print("\n2. Loading data with DataCore...")
    data_core = ej.DataCore.load_from_csv(
        file_path=data_file,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    print(f"   Loaded {len(data_core.time_series_data[0].data)} data points")

    # Validate data quality
    print("\n3. Validating data quality...")
    quality_metrics = data_core.validate_data_quality()
    print(f"   Missing data: {quality_metrics["missing_data_percentage"]["dataset_0"]:.2f}%")
    print(f"   Outliers: {quality_metrics["outlier_percentage"]["dataset_0"]:.2f}%")

    # Preprocess data
    print("\n4. Preprocessing data...")
    data_core.preprocess_data(
        normalize=False,  # Keep original scale for interpretation
        remove_outliers=True,
        interpolate_missing=False
    )
    print("   Data preprocessing completed")

    # Fit JumpRope model
    print("\n5. Fitting JumpRope model...")
    time_points = np.sort(sample_data['time'].unique())

    model = ej.JumpRope.fit(
        data_core,
        model_type='jump-diffusion',
        time_points=time_points
    )
    print(f"   Fitted model with equilibrium: {model.fitted_parameters.equilibrium".2f"}")
    print(f"   Reversion speed: {model.fitted_parameters.reversion_speed".2f"}")
    print(f"   Jump intensity: {model.fitted_parameters.jump_intensity".3f"}")

    # Generate trajectories
    print("\n6. Generating developmental trajectories...")
    trajectories = model.generate_trajectories(n_samples=20, x0=10.0)
    print(f"   Generated {trajectories.shape[0]} trajectories")
    print(f"   Each with {trajectories.shape[1]} time points")

    # Analyze cross-sections
    print("\n7. Analyzing cross-sectional distributions...")
    analyzer = ej.LaserPlaneAnalyzer(model)

    # Analyze at different developmental stages
    stages = [5.0, 10.0, 15.0, 20.0]
    for stage in stages:
        result = analyzer.analyze_cross_section(time_point=stage)
        print(f"   Stage {stage"4.1f"}: mean = {result.moments['mean']"6.2f"}, "
              f"std = {result.moments['std']:".2f"        print(f"   Distribution: {result.distribution_fit.get('distribution', 'unknown')}")

    # Create visualizations
    print("\n8. Creating visualizations...")

    # Trajectory plot
    visualizer = ej.TrajectoryVisualizer()
    fig = visualizer.plot_trajectories(model, n_trajectories=10)
    plt.savefig('developmental_trajectories.png', dpi=150, bbox_inches='tight')
    print("   Saved: developmental_trajectories.png")

    # Cross-section plot
    fig = visualizer.plot_cross_sections(model, time_points=stages)
    plt.savefig('cross_sections.png', dpi=150, bbox_inches='tight')
    print("   Saved: cross_sections.png")

    # 3D landscape plot
    fig = visualizer.plot_landscapes(model)
    plt.savefig('phenotypic_landscape.png', dpi=150, bbox_inches='tight')
    print("   Saved: phenotypic_landscape.png")

    # Compare distributions across stages
    print("\n9. Comparing distributions across developmental stages...")
    condition_data = {}
    for stage in stages:
        cross_section = model.compute_cross_sections(
            np.argmin(np.abs(time_points - stage))
        )
        condition_data[f'stage_{int(stage)}'] = cross_section

    comparison = analyzer.compare_distributions(
        time_point=10.0,  # Reference time point
        condition_data=condition_data
    )

    print(f"   Found significant differences in: {comparison.significant_differences}")

    # Evolutionary sampling
    print("\n10. Performing evolutionary sampling...")
    sampler = ej.EvolutionSampler(data_core)

    samples = sampler.sample(n_samples=100, method='monte-carlo')
    print(f"   Generated {samples.samples.shape[0]} evolutionary samples")

    # Analyze evolutionary patterns
    print("\n11. Analyzing evolutionary patterns...")
    evolution_analysis = sampler.analyze_evolutionary_patterns()

    print(f"   Mean heritability: {np.mean(list(evolution_analysis['genetic_parameters'].values()))".3f"}")
    print(f"   Effective population size: {evolution_analysis['population_statistics'].effective_population_size".0f"}")

    # Time series analysis
    print("\n12. Performing time series analysis...")
    analytics = ej.AnalyticsEngine(data_core)

    ts_results = analytics.analyze_time_series()
    print(f"   Change points detected: {len(ts_results.change_points)}")
    print(f"   Forecasted values: {list(ts_results.forecasts.keys())}")

    # Clean up
    print("\n13. Cleaning up...")
    data_file.unlink()
    print("   Removed temporary files")

    print("\n" + "=" * 50)
    print("EvoJump example completed successfully!")
    print("\nGenerated files:")
    print("  - developmental_trajectories.png")
    print("  - cross_sections.png")
    print("  - phenotypic_landscape.png")
    print("\nTo explore the results interactively, you can:")
    print("  - Load the plots in your preferred image viewer")
    print("  - Modify the example to create additional analyses")
    print("  - Use the command-line interface: evojump-cli --help")

if __name__ == '__main__':
    main()

