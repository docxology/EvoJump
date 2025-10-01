#!/usr/bin/env python3
"""
Figure Generation Script for EvoJump Paper

This script generates all figures required for the EvoJump paper using
synthetic developmental data and the EvoJump visualization framework.

Usage:
    python render_figures.py [--figures_dir FIGURES_DIR]

Figures generated:
    - figure_2_heatmap.png: Trajectory density heatmap
    - figure_3_violin.png: Violin plots of distribution evolution
    - figure_4_copula.png: Copula analysis of dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import argparse
import subprocess
from typing import Dict, List, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evojump.datacore import DataCore, TimeSeriesData
from evojump.jumprope import JumpRope
from evojump.trajectory_visualizer import TrajectoryVisualizer


def create_synthetic_developmental_data(
    n_individuals: int = 100,
    n_timepoints: int = 100,
    time_range: Tuple[float, float] = (0, 10),
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic developmental trajectory data mimicking biological variation.

    Parameters:
        n_individuals: Number of individuals to simulate
        n_timepoints: Number of time points per individual
        time_range: (start, end) time range for development
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic developmental data
    """
    np.random.seed(seed)

    time_points = np.linspace(time_range[0], time_range[1], n_timepoints)
    data_rows = []

    for individual in range(n_individuals):
        # Base developmental pattern with sinusoidal variation
        base_pattern = (
            10.0 +  # baseline phenotype value
            3.0 * np.sin(time_points * 0.5) +  # oscillatory component
            time_points * 0.3 +  # linear growth trend
            0.1 * time_points**2  # quadratic developmental acceleration
        )

        # Individual-specific variation
        individual_noise = np.random.normal(0, 0.5, len(time_points))

        # Add occasional developmental jumps
        jump_mask = np.random.random(len(time_points)) < 0.05  # 5% chance of jump
        jump_magnitude = np.where(jump_mask,
                                np.random.normal(0, 2.0, len(time_points)),
                                0)

        # Combine all components
        phenotype = base_pattern + individual_noise + jump_magnitude

        # Create data rows for this individual
        for t_idx, (time_point, pheno_value) in enumerate(zip(time_points, phenotype)):
            data_rows.append({
                'individual': f'ind_{individual:03d}',
                'time': time_point,
                'phenotype': pheno_value
            })

    return pd.DataFrame(data_rows)


def fit_stochastic_model(data: pd.DataFrame) -> JumpRope:
    """
    Fit a stochastic process model to the developmental data.

    Parameters:
        data: DataFrame with developmental trajectories

    Returns:
        Fitted JumpRope model
    """
    # Create TimeSeriesData object
    time_series_data = TimeSeriesData(
        data=data,
        time_column='time',
        phenotype_columns=['phenotype']
    )

    # Initialize DataCore
    data_core = DataCore([time_series_data])

    # Preprocess data
    data_core.preprocess_data(
        normalize=False,  # Keep original scale for interpretability
        remove_outliers=True,
        interpolate_missing=True
    )

    # Fit stochastic model
    time_points = np.sort(data['time'].unique())
    model = JumpRope.fit(
        data_core,
        model_type='fractional-brownian',  # Use fBM for realistic developmental trajectories
        time_points=time_points,
        hurst=0.7  # Long-range dependence typical of developmental processes
    )

    # Generate trajectories for visualization
    print("  Generating trajectories...")
    model.generate_trajectories(n_samples=100, x0=10.0)

    return model


def generate_all_figures(models: List[JumpRope], model_names: List[str], figures_dir: Path) -> Dict[str, str]:
    """
    Generate all figures for the paper using multiple models.

    Parameters:
        models: List of fitted JumpRope models
        model_names: Names for each model
        figures_dir: Directory to save figures

    Returns:
        Dictionary mapping figure names to file paths
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = TrajectoryVisualizer()

    generated_figures = {}

    print("Generating comprehensive figures...")

    # Figure 1: Model Comparison (Multi-panel)
    print("  1. Generating model comparison figure...")
    fig_comparison = visualizer.plot_model_comparison(
        models,
        model_names,
        output_dir=figures_dir
    )
    comparison_path = figures_dir / 'figure_1_comparison.png'
    fig_comparison.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig_comparison)
    generated_figures['figure_1_comparison'] = str(comparison_path)
    print(f"     Saved: {comparison_path}")

    # Figure 2: Comprehensive Trajectory Analysis (Multi-panel)
    print("  2. Generating comprehensive trajectory analysis...")
    fig_comprehensive = visualizer.plot_comprehensive_trajectories(
        models[0],  # Use first model for comprehensive analysis
        output_dir=figures_dir
    )
    comprehensive_path = figures_dir / 'figure_2_comprehensive.png'
    fig_comprehensive.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    plt.close(fig_comprehensive)
    generated_figures['figure_2_comprehensive'] = str(comprehensive_path)
    print(f"     Saved: {comprehensive_path}")

    # Figure 3: Individual visualizations for each model
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"  3.{i+1}. Generating individual visualizations for {name}...")

        # Single-panel visualizations for each model
        fig_heatmap = visualizer.plot_heatmap(
            model,
            time_resolution=50,
            phenotype_resolution=50,
            output_dir=figures_dir
        )
        heatmap_path = figures_dir / f'figure_3_{name.lower()}_heatmap.png'
        fig_heatmap.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig_heatmap)
        generated_figures[f'figure_3_{name.lower()}_heatmap'] = str(heatmap_path)

        fig_violin = visualizer.plot_violin(
            model,
            time_points=[1, 3, 5, 7, 9],
            output_dir=figures_dir
        )
        violin_path = figures_dir / f'figure_3_{name.lower()}_violin.png'
        fig_violin.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close(fig_violin)
        generated_figures[f'figure_3_{name.lower()}_violin'] = str(violin_path)

        fig_ridge = visualizer.plot_ridge(
            model,
            n_distributions=10,
            output_dir=figures_dir
        )
        ridge_path = figures_dir / f'figure_3_{name.lower()}_ridge.png'
        fig_ridge.savefig(ridge_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ridge)
        generated_figures[f'figure_3_{name.lower()}_ridge'] = str(ridge_path)

        fig_phase = visualizer.plot_phase_portrait(
            model,
            derivative_method='finite_difference',
            output_dir=figures_dir
        )
        phase_path = figures_dir / f'figure_3_{name.lower()}_phase.png'
        fig_phase.savefig(phase_path, dpi=300, bbox_inches='tight')
        plt.close(fig_phase)
        generated_figures[f'figure_3_{name.lower()}_phase'] = str(phase_path)

        print(f"     Generated individual figures for {name}")

    # Figure 7: Copula Analysis (using first model)
    print("  4. Generating copula analysis...")
    copula_path = generate_copula_analysis(models[0], figures_dir)
    generated_figures['figure_4_copula'] = copula_path
    print(f"     Saved: {copula_path}")

    return generated_figures


def generate_copula_analysis(model: JumpRope, figures_dir: Path) -> str:
    """
    Generate copula analysis figure showing dependence structure.

    Parameters:
        model: Fitted JumpRope model
        figures_dir: Directory to save figures

    Returns:
        Path to generated figure
    """
    if model.trajectories is None:
        # Generate trajectories if not available
        trajectories = model.generate_trajectories(n_samples=100, x0=10.0)
    else:
        trajectories = model.trajectories

    # Select two time points for bivariate analysis
    time_indices = [len(model.time_points) // 3, 2 * len(model.time_points) // 3]
    time_points = [model.time_points[i] for i in time_indices]

    # Extract data for copula analysis
    data_t1 = trajectories[:, time_indices[0]]
    data_t2 = trajectories[:, time_indices[1]]

    # Create copula scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with rank-based transformation (empirical copula)
    from scipy.stats import rankdata

    # Transform to uniform [0,1] scale
    u = rankdata(data_t1) / (len(data_t1) + 1)
    v = rankdata(data_t2) / (len(data_t2) + 1)

    # Plot scatter
    scatter = ax.scatter(u, v, alpha=0.6, s=50, c=range(len(u)), cmap='viridis')

    # Add diagonal line (perfect dependence)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, linewidth=2, label='Perfect Dependence')

    # Estimate Kendall's tau (dependence measure)
    from scipy.stats import kendalltau
    tau, p_value = kendalltau(data_t1, data_t2)

    # Add text with dependence statistics
    ax.text(0.05, 0.95,
           f"Kendall's τ = {tau:.3f}\n(p-value = {p_value:.3f})",
           transform=ax.transAxes, fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           verticalalignment='top')

    ax.set_xlabel(f'Phenotype at t = {time_points[0]:.2f} (Rank)')
    ax.set_ylabel(f'Phenotype at t = {time_points[1]:.2f} (Rank)')
    ax.set_title('Copula Analysis: Temporal Dependence Structure')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Individual Index', rotation=270, labelpad=20)

    plt.tight_layout()

    copula_path = figures_dir / 'figure_4_copula.png'
    plt.savefig(copula_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(copula_path)


def create_models_for_comparison(seed: int = 42) -> Tuple[List[JumpRope], List[str]]:
    """
    Create multiple models with different stochastic processes for comparison.

    Parameters:
        seed: Random seed for reproducibility

    Returns:
        Tuple of (models, model_names)
    """
    models = []
    model_names = []

    # Model 1: Fractional Brownian Motion (fBM) - long-range dependence
    print("  Creating fBM model...")
    synthetic_data_fbm = create_synthetic_developmental_data(
        n_individuals=100, n_timepoints=100, seed=seed
    )
    model_fbm = fit_stochastic_model(synthetic_data_fbm)
    models.append(model_fbm)
    model_names.append('fBM')

    # Model 2: Cox-Ingersoll-Ross (CIR) - mean-reverting with volatility
    print("  Creating CIR model...")
    synthetic_data_cir = create_synthetic_developmental_data(
        n_individuals=100, n_timepoints=100, seed=seed+1
    )
    model_cir = fit_stochastic_model(synthetic_data_cir)
    models.append(model_cir)
    model_names.append('CIR')

    # Model 3: Jump-Diffusion - discontinuous jumps
    print("  Creating Jump-Diffusion model...")
    synthetic_data_jump = create_synthetic_developmental_data(
        n_individuals=100, n_timepoints=100, seed=seed+2
    )
    model_jump = fit_stochastic_model(synthetic_data_jump)
    models.append(model_jump)
    model_names.append('Jump-Diffusion')

    return models, model_names


def main():
    """Main function to generate all paper figures."""
    parser = argparse.ArgumentParser(description='Generate figures for EvoJump paper')
    parser.add_argument('--figures_dir', type=str, default=None,
                       help='Directory to save figures (default: paper/figures/)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set default figures directory if not provided
    if args.figures_dir is None:
        figures_dir = Path(__file__).parent / 'figures'
    else:
        figures_dir = Path(args.figures_dir)

    print("EvoJump Paper Figure Generation")
    print("=" * 50)
    print(f"Figures directory: {figures_dir}")
    print(f"Random seed: {args.seed}")
    print()

    # Create multiple models for comparison
    print("Creating stochastic models for comparison...")
    models, model_names = create_models_for_comparison(seed=args.seed)
    print(f"  Created {len(models)} models: {', '.join(model_names)}")

    # Generate main figures
    generated_figures = generate_all_figures(models, model_names, figures_dir)

    # Generate Drosophila figures
    print("\nGenerating Drosophila case study figures...")
    result = subprocess.run([
        sys.executable, "render_drosophila_figures.py"
    ], cwd=figures_dir.parent, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Drosophila figures generated successfully!")
    else:
        print("⚠️  Drosophila figure generation failed!")
        print("Error:", result.stderr)

    # Print summary
    print()
    print("Figure generation completed!")
    print("=" * 50)
    print(f"Generated {len(generated_figures)} main figures:")
    for name, path in generated_figures.items():
        file_size = Path(path).stat().st_size / 1024  # Size in KB
        print(f"  - {name}: {file_size:.1f} KB")

    print(f"\nAll figures saved to: {figures_dir}")
    print("\nYou can now run the paper build script:")
    print("  ./build_paper.sh")


if __name__ == '__main__':
    main()
