#!/usr/bin/env python3
"""
Working Demo for EvoJump Package

This example demonstrates the core functionality of the EvoJump package
and generates all expected outputs including plots and reports.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ,'src'))
from evojump.datacore import DataCore
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer
from evojump.trajectory_visualizer import TrajectoryVisualizer
from evojump.evolution_sampler import EvolutionSampler
from evojump.analytics_engine import AnalyticsEngine
import matplotlib.pyplot as plt
import json

def create_sample_data():
    """Create sample developmental data."""
    np.random.seed(42)
    n_individuals = 50
    time_points = np.linspace(0, 20, 21)
    data_rows = []

    for individual in range(n_individuals):
        base_pattern = 10 + 5 * np.sin(time_points * 0.3) + time_points * 0.2
        individual_variation = np.random.normal(0, 1, len(time_points))
        phenotype = base_pattern + individual_variation

        for t_idx, time_point in enumerate(time_points):
            data_rows.append({
                'individual': f'ind_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx]
            })

    return pd.DataFrame(data_rows)

def main():
    """Main demo function."""
    print("EvoJump Working Demo")
    print("=" * 50)

    # Create and save sample data
    print("1. Creating sample data...")
    sample_data = create_sample_data()
    data_file = Path("demo_data.csv")
    sample_data.to_csv(data_file, index=False)
    print(f"   Created {len(sample_data)} measurements")

    # Load and preprocess data
    print("2. Loading and preprocessing data...")
    data_core = DataCore.load_from_csv(
        file_path=data_file,
        time_column='time',
        phenotype_columns=['phenotype']
    )

    data_core.preprocess_data(
        normalize=True,
        remove_outliers=True,
        interpolate_missing=True
    )

    # Fit model
    print("3. Fitting stochastic model...")
    time_points = np.sort(sample_data['time'].unique())
    model = JumpRope.fit(data_core, model_type='jump-diffusion', time_points=time_points)

    # Generate trajectories
    print("4. Generating trajectories...")
    trajectories = model.generate_trajectories(n_samples=20, x0=10.0)

    # Analyze cross-sections
    print("5. Analyzing cross-sections...")
    analyzer = LaserPlaneAnalyzer(model)
    stages = [5.0, 10.0, 15.0, 20.0]

    for stage in stages:
        result = analyzer.analyze_cross_section(time_point=stage)
        print(f"   Stage {stage:4.1f}: mean = {result.moments['mean']:6.2f}, "
              f"std = {result.moments['std']:.2f}")

    # Create visualizations
    print("6. Creating visualizations...")
    visualizer = TrajectoryVisualizer()

    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    # Plot trajectories
    fig = visualizer.plot_trajectories(model, n_trajectories=10, interactive=False)
    fig.savefig(output_dir / 'trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   Saved: trajectories.png")

    # Plot cross-sections
    fig = visualizer.plot_cross_sections(model, time_points=stages, interactive=False)
    fig.savefig(output_dir / 'cross_sections.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   Saved: cross_sections.png")

    # Plot landscape
    fig = visualizer.plot_landscapes(model, interactive=False)
    fig.savefig(output_dir / 'landscape.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   Saved: landscape.png")

    # Create report
    print("7. Creating analysis report...")
    report = {
        'summary': 'EvoJump Demo Results',
        'data_info': {
            'n_samples': len(sample_data),
            'time_range': [float(sample_data['time'].min()), float(sample_data['time'].max())]
        },
        'model_params': {
            'equilibrium': float(model.fitted_parameters.equilibrium),
            'reversion_speed': float(model.fitted_parameters.reversion_speed),
            'jump_intensity': float(model.fitted_parameters.jump_intensity)
        },
        'cross_section_results': {
            str(stage): {
                'mean': float(result.moments['mean']),
                'std': float(result.moments['std'])
            }
            for stage, result in [(s, analyzer.analyze_cross_section(s)) for s in stages]
        }
    }

    report_file = output_dir / 'demo_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   Saved: demo_report.json")

    # Clean up
    data_file.unlink()
    print("8. Cleaned up temporary files")

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - trajectories.png")
    print("  - cross_sections.png")
    print("  - landscape.png")
    print("  - demo_report.json")

    return output_dir

if __name__ == '__main__':
    output_dir = main()
    print(f"\nCheck the '{output_dir}' directory for all generated outputs!")

