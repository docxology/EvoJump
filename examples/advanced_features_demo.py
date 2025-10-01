"""
Advanced Features Demonstration for EvoJump

This script demonstrates the newly added advanced features including:
- Advanced stochastic process models (Fractional Brownian Motion, CIR, Levy)
- New visualization types (heatmaps, violin plots, ridge plots, phase portraits)
- Advanced statistical methods (wavelet analysis, copula methods, extreme value theory)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

import evojump as ej

# Set random seed for reproducibility
np.random.seed(42)

def create_synthetic_data(n_samples=100, n_timepoints=50):
    """Create synthetic developmental trajectory data."""
    print("Creating synthetic developmental trajectory data...")
    
    time_points = np.linspace(0, 10, n_timepoints)
    data = []
    
    for i in range(n_samples):
        # Generate non-stationary developmental trajectories with jumps
        trajectory = 10 + 0.5 * time_points + np.cumsum(np.random.normal(0, 0.3, n_timepoints))
        
        # Add developmental jumps
        jump_times = np.random.choice(n_timepoints, size=3, replace=False)
        for jt in jump_times:
            trajectory[jt:] += np.random.normal(2, 0.5)
        
        for t_idx, t in enumerate(time_points):
            data.append({
                'time': t,
                'individual_id': i,
                'phenotype1': trajectory[t_idx] + np.random.normal(0, 0.1),
                'phenotype2': trajectory[t_idx] * 1.5 + np.random.normal(0, 0.5)
            })
    
    return pd.DataFrame(data)


def demonstrate_advanced_models():
    """Demonstrate advanced stochastic process models."""
    print("\n=== ADVANCED STOCHASTIC PROCESS MODELS ===\n")
    
    # Create data
    data = create_synthetic_data(n_samples=30, n_timepoints=30)
    data_core = ej.DataCore.load_from_csv(
        pd.io.common.StringIO(data.to_csv()),
        time_column='time'
    )
    
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test Fractional Brownian Motion
    print("1. Fitting Fractional Brownian Motion (long-range dependence)...")
    fbm_model = ej.JumpRope.fit(
        data_core,
        model_type='fractional-brownian',
        hurst=0.7
    )
    fbm_trajectories = fbm_model.generate_trajectories(n_samples=50, x0=10.0)
    print(f"   ✓ Generated {fbm_trajectories.shape[0]} trajectories with Hurst={0.7}")
    
    # Test Cox-Ingersoll-Ross Process
    print("2. Fitting Cox-Ingersoll-Ross process (mean-reverting, non-negative)...")
    cir_model = ej.JumpRope.fit(
        data_core,
        model_type='cir',
        equilibrium=15.0,
        reversion_speed=0.5
    )
    cir_trajectories = cir_model.generate_trajectories(n_samples=50, x0=10.0)
    print(f"   ✓ Generated {cir_trajectories.shape[0]} trajectories with mean reversion")
    
    # Test Levy Process
    print("3. Fitting Levy process (heavy-tailed distributions)...")
    levy_model = ej.JumpRope.fit(
        data_core,
        model_type='levy',
        levy_alpha=1.5,
        levy_beta=0.0
    )
    levy_trajectories = levy_model.generate_trajectories(n_samples=50, x0=10.0)
    print(f"   ✓ Generated {levy_trajectories.shape[0]} trajectories with stable increments")
    
    return fbm_model, cir_model, levy_model, output_dir


def demonstrate_advanced_visualizations(models, output_dir):
    """Demonstrate new visualization types."""
    print("\n=== ADVANCED VISUALIZATION METHODS ===\n")
    
    fbm_model, cir_model, levy_model = models
    visualizer = ej.TrajectoryVisualizer()
    
    # Heatmap visualization
    print("1. Creating trajectory density heatmap...")
    fig = visualizer.plot_heatmap(
        fbm_model,
        time_resolution=40,
        phenotype_resolution=40,
        output_dir=output_dir,
        interactive=False
    )
    plt.close(fig)
    print("   ✓ Saved density heatmap showing trajectory evolution")
    
    # Violin plots
    print("2. Creating violin plots...")
    fig = visualizer.plot_violin(
        cir_model,
        time_points=None,  # Auto-select time points
        output_dir=output_dir
    )
    plt.close(fig)
    print("   ✓ Saved violin plots showing distribution evolution")
    
    # Ridge plot (joyplot)
    print("3. Creating ridge plot...")
    fig = visualizer.plot_ridge(
        levy_model,
        n_distributions=8,
        output_dir=output_dir
    )
    plt.close(fig)
    print("   ✓ Saved ridge plot showing temporal distribution changes")
    
    # Phase portrait
    print("4. Creating phase portrait...")
    fig = visualizer.plot_phase_portrait(
        fbm_model,
        derivative_method='finite_difference',
        output_dir=output_dir,
        interactive=False
    )
    plt.close(fig)
    print("   ✓ Saved phase portrait showing phenotype vs. rate of change")


def demonstrate_advanced_analytics():
    """Demonstrate advanced statistical methods."""
    print("\n=== ADVANCED STATISTICAL METHODS ===\n")
    
    # Create more complex data for analytics
    data = create_synthetic_data(n_samples=200, n_timepoints=100)
    analytics = ej.AnalyticsEngine(data, time_column='time')
    
    # Wavelet analysis
    print("1. Performing wavelet analysis...")
    try:
        # Install PyWavelets if needed
        import pywt
        wavelet_result = analytics.wavelet_analysis('phenotype1', wavelet='morl')
        print(f"   ✓ Dominant scale: {wavelet_result['dominant_scale']:.2f}")
        print(f"   ✓ Detected {wavelet_result['n_events']} time-localized events")
    except ImportError:
        print("   ⚠ PyWavelets not installed. Install with: pip install PyWavelets")
    except Exception as e:
        print(f"   ⚠ Wavelet analysis error: {e}")
    
    # Copula analysis
    print("2. Performing copula analysis...")
    try:
        copula_result = analytics.copula_analysis('phenotype1', 'phenotype2', copula_type='gaussian')
        print(f"   ✓ Copula parameter: {copula_result['copula_parameter']:.3f}")
        print(f"   ✓ Kendall's tau: {copula_result['kendall_tau']:.3f} (p={copula_result['kendall_tau_pvalue']:.4f})")
        print(f"   ✓ Dependence class: {copula_result['dependence_class']}")
    except Exception as e:
        print(f"   ⚠ Copula analysis error: {e}")
    
    # Extreme value analysis
    print("3. Performing extreme value analysis...")
    try:
        extreme_result = analytics.extreme_value_analysis('phenotype1')
        print(f"   ✓ POT threshold: {extreme_result['pot_method']['threshold']:.2f}")
        print(f"   ✓ Shape parameter: {extreme_result['pot_method']['shape_parameter']:.3f}")
        print(f"   ✓ 100-year return level: {extreme_result['pot_method']['return_levels']['100_year']:.2f}")
        print(f"   ✓ Hill estimator (tail index): {extreme_result['hill_estimator']:.3f}")
    except Exception as e:
        print(f"   ⚠ Extreme value analysis error: {e}")
    
    # Regime switching analysis
    print("4. Performing regime switching analysis...")
    try:
        regime_result = analytics.regime_switching_analysis('phenotype1', n_regimes=3)
        print(f"   ✓ Identified {regime_result['n_regimes']} regimes")
        print(f"   ✓ Number of regime switches: {regime_result['n_switches']}")
        for stat in regime_result['regime_statistics']:
            print(f"   ✓ Regime {stat['regime_id']}: mean={stat['mean']:.2f}, "
                  f"duration={stat['duration_pct']:.1f}%")
    except Exception as e:
        print(f"   ⚠ Regime switching analysis error: {e}")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("EvoJump Advanced Features Demonstration")
    print("=" * 70)
    
    # Demonstrate advanced models
    models = demonstrate_advanced_models()
    fbm_model, cir_model, levy_model = models[:3]
    output_dir = models[3]
    
    # Demonstrate advanced visualizations
    demonstrate_advanced_visualizations((fbm_model, cir_model, levy_model), output_dir)
    
    # Demonstrate advanced analytics
    demonstrate_advanced_analytics()
    
    print("\n" + "=" * 70)
    print("✅ ADVANCED FEATURES DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output files saved to: {output_dir.absolute()}")
    print("\n📊 Summary of New Features:")
    print("   ✅ Fractional Brownian Motion (long-range dependence)")
    print("   ✅ Cox-Ingersoll-Ross process (mean-reverting)")
    print("   ✅ Levy process (heavy-tailed distributions)")
    print("   ✅ Trajectory density heatmaps")
    print("   ✅ Violin plot distributions")
    print("   ✅ Ridge plots (joypllots)")
    print("   ✅ Phase portraits")
    print("   ✅ Wavelet analysis (time-frequency)")
    print("   ✅ Copula analysis (dependence structure)")
    print("   ✅ Extreme value theory")
    print("   ✅ Regime switching detection")
    print("\n🚀 EvoJump is now equipped with state-of-the-art analytical methods!")


if __name__ == '__main__':
    main()
