#!/usr/bin/env python3
"""
Animation Demo for EvoJump Package

This example generates comprehensive animations of developmental processes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump.datacore import DataCore
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer
from evojump.trajectory_visualizer import TrajectoryVisualizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

def create_animation_data():
    """Create sample data for animation demonstration."""
    print("Creating animation data...")

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
    """Main animation demo function."""
    print("EvoJump Animation Demo")
    print("=" * 50)

    # Create and save sample data
    sample_data = create_animation_data()
    data_file = Path("animation_data.csv")
    sample_data.to_csv(data_file, index=False)

    # Load and preprocess data
    datacore = DataCore.load_from_csv(
        file_path=data_file,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    datacore.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

    # Fit model
    time_points = np.sort(sample_data['time'].unique())
    model = JumpRope.fit(datacore, model_type='jump-diffusion', time_points=time_points)
    model.generate_trajectories(n_samples=30, x0=10.0)

    # Create visualizer
    visualizer = TrajectoryVisualizer()

    # Animation 1: Basic trajectory animation
    print("Creating basic trajectory animation...")
    anim1_dir = Path("animation_outputs")
    anim1_dir.mkdir(exist_ok=True)

    animation1 = visualizer.create_animation(
        jump_rope_model=model,
        n_frames=40,
        time_range=(0, 20),
        output_dir=anim1_dir
    )

    print(f"   Saved: {anim1_dir}/animation.gif")

    # Animation 2: Custom animation with multiple subplots
    print("Creating custom multi-panel animation...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def animate_comprehensive(frame_idx):
        current_time = time_points[frame_idx % len(time_points)]
        current_time_idx = frame_idx % len(time_points)

        for ax in axes.flat:
            ax.clear()

        if model.trajectories is not None:
            # Plot 1: Individual trajectories
            for i in range(min(15, model.trajectories.shape[0])):
                axes[0, 0].plot(time_points[:current_time_idx+1],
                              model.trajectories[i, :current_time_idx+1],
                              alpha=0.6, linewidth=1)

            # Plot mean trajectory
            mean_traj = np.mean(model.trajectories, axis=0)
            axes[0, 0].plot(time_points[:current_time_idx+1],
                          mean_traj[:current_time_idx+1],
                          color='red', linewidth=3, label='Mean')

            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Phenotype')
            axes[0, 0].set_title(f'Trajectories (Time: {current_time".2f"})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Cross-sectional distribution
            current_data = model.trajectories[:, current_time_idx]
            axes[0, 1].hist(current_data, bins=20, alpha=0.7, density=True)
            axes[0, 1].axvline(np.mean(current_data), color='red', linewidth=2)
            axes[0, 1].set_xlabel('Phenotype Value')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Cross-Sectional Distribution')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: 3D trajectory evolution
            ax3d = fig.add_subplot(2, 2, 3, projection='3d')
            for i in range(min(5, model.trajectories.shape[0])):
                ax3d.plot(time_points[:current_time_idx+1], [i]*len(time_points[:current_time_idx+1]),
                         model.trajectories[i, :current_time_idx+1], alpha=0.7)

            ax3d.set_xlabel('Time')
            ax3d.set_ylabel('Individual')
            ax3d.set_zlabel('Phenotype')
            ax3d.set_title('3D Trajectory Evolution')

            # Plot 4: Statistics over time
            if current_time_idx > 0:
                all_means = [np.mean(model.trajectories[:, t]) for t in range(current_time_idx+1)]
                all_stds = [np.std(model.trajectories[:, t]) for t in range(current_time_idx+1)]

                axes[1, 0].plot(range(current_time_idx+1), all_means, label='Mean', linewidth=2)
                axes[1, 0].fill_between(range(current_time_idx+1),
                                      np.array(all_means) - np.array(all_stds),
                                      np.array(all_means) + np.array(all_stds),
                                      alpha=0.3)
                axes[1, 0].axvline(current_time_idx, color='red', linestyle='--', alpha=0.7)
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_ylabel('Phenotype Value')
                axes[1, 0].set_title('Statistics Evolution')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Model parameters evolution
            axes[1, 1].text(0.1, 0.9, f'Current Time: {current_time".2f"}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.8, f'Current Mean: {np.mean(current_data)".3f"}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.7, f'Current Std: {np.std(current_data)".3f"}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Current Min: {np.min(current_data)".3f"}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Current Max: {np.max(current_data)".3f"}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Real-time Statistics')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    anim2 = animation.FuncAnimation(
        fig, animate_comprehensive,
        frames=len(time_points),
        interval=200,
        blit=False,
        repeat=True
    )

    anim2.save(anim1_dir / 'comprehensive_animation.gif', writer='pillow', fps=5)

    print(f"   Saved: {anim1_dir}/comprehensive_animation.gif")

    # Clean up
    data_file.unlink()

    print("\n" + "=" * 50)
    print("Animation demo completed!")
    print("
Generated animations:"    print(f"   • {anim1_dir}/animation.gif")
    print(f"   • {anim1_dir}/comprehensive_animation.gif")

    return anim1_dir

if __name__ == '__main__':
    output_dir = main()
    print(f"\nAnimations saved to: {output_dir}")
    print("\nOpen the .gif files in any browser to view the animations!")

