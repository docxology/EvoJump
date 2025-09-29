#!/usr/bin/env python3
"""
Simple Animation Demo for EvoJump Package

This example generates basic animations of developmental processes.
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    """Simple animation demo."""
    print("EvoJump Simple Animation Demo")
    print("=" * 40)

    # Create sample data
    np.random.seed(42)
    n_individuals = 30
    time_points = np.linspace(0, 15, 16)

    data_rows = []
    for individual in range(n_individuals):
        base_pattern = 10 + 4 * np.sin(time_points * 0.4) + time_points * 0.3
        individual_variation = np.random.normal(0, 1, len(time_points))
        phenotype = base_pattern + individual_variation

        for t_idx, time_point in enumerate(time_points):
            data_rows.append({
                'individual': f'ind_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx]
            })

    sample_data = pd.DataFrame(data_rows)
    data_file = Path("simple_animation_data.csv")
    sample_data.to_csv(data_file, index=False)

    # Load and preprocess data
    datacore = DataCore.load_from_csv(
        file_path=data_file,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    datacore.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

    # Fit model
    model = JumpRope.fit(datacore, model_type='jump-diffusion', time_points=time_points)
    model.generate_trajectories(n_samples=20, x0=10.0)

    # Create visualizer
    visualizer = TrajectoryVisualizer()

    # Create output directory
    output_dir = Path("simple_animation_outputs")
    output_dir.mkdir(exist_ok=True)

    print("Creating animations...")

    # Animation 1: Basic trajectory animation
    print("1. Creating basic trajectory animation...")
    animation1 = visualizer.create_animation(
        jump_rope_model=model,
        n_frames=30,
        time_range=(0, 15),
        output_dir=output_dir
    )
    print(f"   Saved: {output_dir}/animation.gif")

    # Animation 2: Custom comprehensive animation
    print("2. Creating comprehensive animation...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def animate_comprehensive(frame_idx):
        current_time = time_points[frame_idx % len(time_points)]
        current_time_idx = frame_idx % len(time_points)

        for ax in axes.flat:
            ax.clear()

        if model.trajectories is not None:
            # Plot trajectories
            for i in range(min(10, model.trajectories.shape[0])):
                axes[0, 0].plot(time_points[:current_time_idx+1],
                              model.trajectories[i, :current_time_idx+1],
                              alpha=0.6, linewidth=1)

            mean_traj = np.mean(model.trajectories, axis=0)
            axes[0, 0].plot(time_points[:current_time_idx+1],
                          mean_traj[:current_time_idx+1],
                          color='red', linewidth=3, label='Mean')

            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Phenotype')
            axes[0, 0].set_title(f'Trajectories (Time: {current_time:.2f})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot cross-section
            current_data = model.trajectories[:, current_time_idx]
            axes[0, 1].hist(current_data, bins=15, alpha=0.7, density=True)
            axes[0, 1].axvline(np.mean(current_data), color='red', linewidth=2)
            axes[0, 1].set_xlabel('Phenotype Value')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Cross-Sectional Distribution')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3D evolution
            ax3d = fig.add_subplot(2, 2, 3, projection='3d')
            for i in range(min(3, model.trajectories.shape[0])):
                ax3d.plot(time_points[:current_time_idx+1], [i]*len(time_points[:current_time_idx+1]),
                         model.trajectories[i, :current_time_idx+1], alpha=0.7)

            ax3d.set_xlabel('Time')
            ax3d.set_ylabel('Individual')
            ax3d.set_zlabel('Phenotype')
            ax3d.set_title('3D Evolution')

            # Plot statistics
            axes[1, 0].plot(range(current_time_idx+1), [np.mean(model.trajectories[:, t]) for t in range(current_time_idx+1)],
                          label='Mean', linewidth=2)
            axes[1, 0].axvline(current_time_idx, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Phenotype Value')
            axes[1, 0].set_title('Statistics Evolution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Real-time stats
            current_data = model.trajectories[:, current_time_idx]
            axes[1, 1].text(0.1, 0.9, f'Time: {current_time:.2f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.8, f'Mean: {np.mean(current_data):.3f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.7, f'Std: {np.std(current_data):.3f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Real-time Statistics')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    anim2 = animation.FuncAnimation(
        fig, animate_comprehensive,
        frames=len(time_points),
        interval=300,
        blit=False,
        repeat=True
    )

    anim2.save(output_dir / 'comprehensive_animation.gif', writer='pillow', fps=4)
    print(f"   Saved: {output_dir}/comprehensive_animation.gif")

    # Clean up
    data_file.unlink()

    print("\n" + "=" * 40)
    print("Simple animation demo completed!")
    print("Generated animations:")
    print(f"   • {output_dir}/animation.gif")
    print(f"   • {output_dir}/comprehensive_animation.gif")

    return output_dir

if __name__ == '__main__':
    output_dir = main()
    print(f"\nAnimations saved to: {output_dir}")
    print("Open the .gif files in any browser to view the animations!")
