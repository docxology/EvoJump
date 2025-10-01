#!/usr/bin/env python3
"""
Comprehensive Animation Demo for EvoJump Package

This example generates multiple types of animations to visualize
the complete developmental process and evolutionary dynamics.
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
import os
from datetime import datetime

def create_rich_sample_data():
    """Create rich sample data with multiple conditions for animation."""
    print("🎬 Creating rich sample data for comprehensive animations...")

    np.random.seed(42)

    # Generate data with multiple developmental patterns
    n_individuals = 100
    time_points = np.linspace(0, 30, 31)  # Longer time series for better animation
    n_conditions = 3  # Control, treatment, mutant

    data_rows = []

    for condition_idx, condition in enumerate(['control', 'treatment', 'mutant']):
        # Different base patterns for each condition
        if condition == 'control':
            base_pattern = 10 + 6 * np.sin(time_points * 0.2) + time_points * 0.15
            variation_scale = 1.0
        elif condition == 'treatment':
            base_pattern = 12 + 4 * np.sin(time_points * 0.25) + time_points * 0.2
            variation_scale = 1.5
        else:  # mutant
            base_pattern = 8 + 8 * np.sin(time_points * 0.15) + time_points * 0.1
            variation_scale = 2.0

        for individual in range(n_individuals // n_conditions):
            # Add developmental jumps occasionally
            jumps = np.zeros(len(time_points))
            if np.random.random() < 0.2:  # 20% chance of developmental jumps
                jump_times = np.random.choice(len(time_points), size=np.random.randint(1, 4), replace=False)
                for jump_time in jump_times:
                    jump_size = np.random.normal(0, 2 + condition_idx)
                    jumps[jump_time:] += jump_size

            phenotype = base_pattern + np.random.normal(0, variation_scale, len(time_points)) + jumps

            for t_idx, time_point in enumerate(time_points):
                data_rows.append({
                    'individual': f'{condition}_{individual:03d}',
                    'time': time_point,
                    'phenotype': phenotype[t_idx],
                    'condition': condition,
                    'genotype': np.random.choice(['AA', 'Aa', 'aa'], p=[0.3, 0.5, 0.2])
                })

    return pd.DataFrame(data_rows)

def create_multiple_models_for_animation():
    """Create multiple models for comparative animations."""
    print("🔬 Creating multiple models for comparative animation...")

    # Create different model types for comparison
    models = {}

    # Model 1: Basic jump-diffusion
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
                'individual': f'model1_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx]
            })

    data1 = pd.DataFrame(data_rows)
    data_file1 = Path("animation_model1_data.csv")
    data1.to_csv(data_file1, index=False)

    datacore1 = DataCore.load_from_csv(
        file_path=data_file1,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    datacore1.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

    model1 = JumpRope.fit(datacore1, model_type='jump-diffusion', time_points=time_points)
    model1.generate_trajectories(n_samples=30, x0=10.0)
    models['jump_diffusion'] = model1

    # Model 2: Ornstein-Uhlenbeck
    data_rows2 = []
    for individual in range(n_individuals):
        base_pattern = 8 + 4 * np.exp(-time_points * 0.1) + np.random.normal(0, 0.8, len(time_points))
        phenotype = base_pattern

        for t_idx, time_point in enumerate(time_points):
            data_rows2.append({
                'individual': f'model2_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx]
            })

    data2 = pd.DataFrame(data_rows2)
    data_file2 = Path("animation_model2_data.csv")
    data2.to_csv(data_file2, index=False)

    datacore2 = DataCore.load_from_csv(
        file_path=data_file2,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    datacore2.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

    model2 = JumpRope.fit(datacore2, model_type='ornstein-uhlenbeck', time_points=time_points)
    model2.generate_trajectories(n_samples=30, x0=8.0)
    models['ornstein_uhlenbeck'] = model2

    # Model 3: Geometric jump-diffusion
    data_rows3 = []
    for individual in range(n_individuals):
        base_pattern = 5 * np.exp(time_points * 0.05) + np.random.normal(0, 0.5, len(time_points))
        phenotype = base_pattern

        for t_idx, time_point in enumerate(time_points):
            data_rows3.append({
                'individual': f'model3_{individual:03d}',
                'time': time_point,
                'phenotype': phenotype[t_idx]
            })

    data3 = pd.DataFrame(data_rows3)
    data_file3 = Path("animation_model3_data.csv")
    data3.to_csv(data_file3, index=False)

    datacore3 = DataCore.load_from_csv(
        file_path=data_file3,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    datacore3.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

    model3 = JumpRope.fit(datacore3, model_type='geometric-jump-diffusion', time_points=time_points)
    model3.generate_trajectories(n_samples=30, x0=5.0)
    models['geometric_jump_diffusion'] = model3

    # Clean up temporary files
    for f in [data_file1, data_file2, data_file3]:
        if f.exists():
            f.unlink()

    return models

def run_comprehensive_animation_demo():
    """Run comprehensive animation demo with multiple animation types."""
    print("🎬 EvoJump Comprehensive Animation Demo")
    print("=" * 70)
    print("This demo generates multiple types of animations:")
    print("• Basic developmental trajectory animations")
    print("• Multi-condition comparative animations")
    print("• Model comparison animations")
    print("• Cross-sectional evolution animations")
    print("• 3D landscape animations")
    print("=" * 70)

    # Create output directories
    base_output_dir = Path("comprehensive_animation_outputs")
    base_output_dir.mkdir(exist_ok=True)

    animations_generated = []
    animation_log = []

    # Animation 1: Basic developmental trajectory animation
    print("\n🎬 Animation 1: Basic Developmental Trajectory Animation")
    print("-" * 50)

    # Create sample data
    sample_data = create_rich_sample_data()
    data_file = Path("animation_rich_data.csv")
    sample_data.to_csv(data_file, index=False)

    # Load and preprocess
    datacore = DataCore.load_from_csv(
        file_path=data_file,
        time_column='time',
        phenotype_columns=['phenotype']
    )
    datacore.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

    # Fit model
    time_points = np.sort(sample_data['time'].unique())
    model = JumpRope.fit(datacore, model_type='jump-diffusion', time_points=time_points)
    model.generate_trajectories(n_samples=40, x0=10.0)

    # Create visualizer
    visualizer = TrajectoryVisualizer()

    # Generate basic animation
    anim1_dir = base_output_dir / "animation_1_basic_trajectory"
    animation1 = visualizer.create_animation(
        jump_rope_model=model,
        n_frames=50,
        time_range=(0, 25),
        output_dir=anim1_dir
    )

    animation_info = {
        'name': 'Basic Developmental Trajectory Animation',
        'directory': str(anim1_dir),
        'description': 'Shows developmental trajectories evolving over time with cross-sectional distributions',
        'frames': 50,
        'time_range': (0, 25),
        'model_type': 'jump_diffusion',
        'n_trajectories': 40
    }
    animations_generated.append(animation_info)
    animation_log.append(f"✅ {animation_info['name']}: {anim1_dir}/animation.gif")

    print(f"   📁 Animation saved to: {anim1_dir}/animation.gif")
    print(f"   📊 {animation_info['frames']} frames, {animation_info['time_range']} time range")

    # Animation 2: Multi-condition animation
    print("\n🎬 Animation 2: Multi-Condition Comparative Animation")
    print("-" * 50)

    # Create separate models for each condition
    condition_models = {}
    for condition in ['control', 'treatment', 'mutant']:
        condition_data = sample_data[sample_data['condition'] == condition].copy()
        condition_data['phenotype'] = condition_data['phenotype']  # Ensure correct column name

        temp_file = Path(f"animation_{condition}_data.csv")
        condition_data.to_csv(temp_file, index=False)

        cond_datacore = DataCore.load_from_csv(
            file_path=temp_file,
            time_column='time',
            phenotype_columns=['phenotype']
        )
        cond_datacore.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

        cond_model = JumpRope.fit(cond_datacore, model_type='jump-diffusion', time_points=time_points)
        cond_model.generate_trajectories(n_samples=25, x0=10.0)
        condition_models[condition] = cond_model

        temp_file.unlink()

    # Create comparative animation
    anim2_dir = base_output_dir / "animation_2_multi_condition"
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def animate_multi_condition(frame_idx):
        current_time = time_points[frame_idx % len(time_points)]
        current_time_idx = frame_idx % len(time_points)

        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        # Plot each condition's trajectories
        colors = ['blue', 'red', 'green']
        conditions = list(condition_models.keys())

        for i, (condition, cond_model) in enumerate(condition_models.items()):
            color = colors[i]

            # Plot trajectories up to current time
            if cond_model.trajectories is not None:
                for j in range(min(10, cond_model.trajectories.shape[0])):  # Plot first 10 trajectories
                    axes[0, 0].plot(time_points[:current_time_idx+1],
                                  cond_model.trajectories[j, :current_time_idx+1],
                                  alpha=0.6, color=color, linewidth=1)

                # Plot mean trajectory
                mean_traj = np.mean(cond_model.trajectories, axis=0)
                axes[0, 0].plot(time_points[:current_time_idx+1],
                              mean_traj[:current_time_idx+1],
                              color=color, linewidth=3, label=f'{condition.capitalize()} (n={len(condition_data)//len(time_points)})')

        axes[0, 0].set_xlabel('Developmental Time')
        axes[0, 0].set_ylabel('Phenotype Value')
        axes[0, 0].set_title(f'Multi-Condition Trajectories (Time: {current_time:.2f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot cross-sections for each condition
        for i, (condition, cond_model) in enumerate(condition_models.items()):
            if cond_model.trajectories is not None:
                current_data = cond_model.trajectories[:, current_time_idx]
                axes[0, 1].hist(current_data, bins=15, alpha=0.6, label=condition.capitalize(),
                              color=colors[i], density=True)

        axes[0, 1].set_xlabel('Phenotype Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Cross-Sectional Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3D landscape evolution
        ax3d = fig.add_subplot(2, 2, 3, projection='3d')
        ax3d.clear()

        for i, (condition, cond_model) in enumerate(condition_models.items()):
            if cond_model.trajectories is not None:
                # Create a mesh grid for 3D surface
                T, X = np.meshgrid(time_points[:current_time_idx+1],
                                 np.linspace(np.min(cond_model.trajectories),
                                           np.max(cond_model.trajectories), 20))

                # Simple surface approximation
                Z = np.mean(cond_model.trajectories[:, :current_time_idx+1], axis=0)
                for j in range(len(Z)):
                    ax3d.plot(time_points[:j+1], [Z[j]] * (j+1), np.linspace(0, Z[j], j+1),
                            color=colors[i], alpha=0.7)

        ax3d.set_xlabel('Time')
        ax3d.set_ylabel('Condition')
        ax3d.set_zlabel('Phenotype')
        ax3d.set_title('3D Phenotypic Landscape Evolution')

        # Plot summary statistics
        axes[1, 0].clear()
        stats_data = []
        for condition, cond_model in condition_models.items():
            if cond_model.trajectories is not None:
                current_values = cond_model.trajectories[:, current_time_idx]
                stats_data.append({
                    'condition': condition.capitalize(),
                    'mean': np.mean(current_values),
                    'std': np.std(current_values),
                    'n': len(current_values)
                })

        conditions_plot = [s['condition'] for s in stats_data]
        means_plot = [s['mean'] for s in stats_data]
        stds_plot = [s['std'] for s in stats_data]

        axes[1, 0].bar(conditions_plot, means_plot, yerr=stds_plot, capsize=5, alpha=0.7)
        axes[1, 0].set_ylabel('Phenotype Value')
        axes[1, 0].set_title('Mean ± SD by Condition')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot condition comparison over time
        axes[1, 1].clear()
        for i, (condition, cond_model) in enumerate(condition_models.items()):
            if cond_model.trajectories is not None:
                mean_trajectory = np.mean(cond_model.trajectories, axis=0)
                axes[1, 1].plot(time_points, mean_trajectory,
                              label=condition.capitalize(), color=colors[i], linewidth=2)

        axes[1, 1].axvline(current_time, color='red', linestyle='--', alpha=0.7,
                         label=f'Current: {current_time:.1f}')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean Phenotype')
        axes[1, 1].set_title('Mean Trajectories Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    import matplotlib.animation as animation
    anim2 = animation.FuncAnimation(
        fig, animate_multi_condition,
        frames=len(time_points),
        interval=200,
        blit=False,
        repeat=True
    )

    anim2.save(anim2_dir / 'multi_condition_animation.gif', writer='pillow', fps=5)

    animation_info = {
        'name': 'Multi-Condition Comparative Animation',
        'directory': str(anim2_dir),
        'description': 'Compares developmental trajectories across control, treatment, and mutant conditions',
        'frames': len(time_points),
        'time_range': (0, 25),
        'model_type': 'multi_condition_comparison',
        'n_conditions': 3,
        'conditions': ['control', 'treatment', 'mutant']
    }
    animations_generated.append(animation_info)
    animation_log.append(f"✅ {animation_info['name']}: {anim2_dir}/multi_condition_animation.gif")

    print(f"   📁 Animation saved to: {anim2_dir}/multi_condition_animation.gif")
    print(f"   📊 {animation_info['frames']} frames, {len(animation_info['conditions'])} conditions")

    # Animation 3: Model comparison animation
    print("\n🎬 Animation 3: Model Comparison Animation")
    print("-" * 50)

    models = create_multiple_models_for_animation()
    anim3_dir = base_output_dir / "animation_3_model_comparison"

    # Create comparative animation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def animate_model_comparison(frame_idx):
        current_time = time_points[frame_idx % len(time_points)]
        current_time_idx = frame_idx % len(time_points)

        for ax in axes.flat:
            ax.clear()

        model_types = list(models.keys())
        colors = ['blue', 'red', 'green']

        for i, (model_name, model) in enumerate(models.items()):
            color = colors[i]

            if model.trajectories is not None:
                # Plot individual trajectories (subset)
                for j in range(min(8, model.trajectories.shape[0])):
                    axes[0, 0].plot(time_points[:current_time_idx+1],
                                  model.trajectories[j, :current_time_idx+1],
                                  alpha=0.4, color=color, linewidth=0.8)

                # Plot mean trajectory
                mean_traj = np.mean(model.trajectories, axis=0)
                axes[0, 0].plot(time_points[:current_time_idx+1],
                              mean_traj[:current_time_idx+1],
                              color=color, linewidth=3, label=model_name.replace('_', ' ').title())

                # Plot cross-section
                current_data = model.trajectories[:, current_time_idx]
                axes[0, 1].hist(current_data, bins=15, alpha=0.6, label=model_name.replace('_', ' ').title(),
                              color=color, density=True)

                # Plot parameter evolution
                axes[1, 0].plot(time_points[:current_time_idx+1],
                              mean_traj[:current_time_idx+1],
                              color=color, linewidth=2, label=model_name.replace('_', ' ').title())

        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Phenotype')
        axes[0, 0].set_title(f'Model Trajectories (Time: {current_time:.2f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Phenotype Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Cross-Sectional Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Mean Phenotype')
        axes[1, 0].set_title('Mean Trajectories Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Model statistics
        stats_text = "Model Statistics:\n"
        for model_name, model in models.items():
            if model.trajectories is not None:
                current_vals = model.trajectories[:, current_time_idx]
                stats_text += f"{model_name.replace('_', ' ').title()}: μ={np.mean(current_vals):.2f}, σ={np.std(current_vals):.2f}\n"

        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Real-time Statistics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        # 3D comparison
        ax3d = fig.add_subplot(2, 3, 6, projection='3d')
        ax3d.clear()

        for i, (model_name, model) in enumerate(models.items()):
            if model.trajectories is not None:
                mean_traj = np.mean(model.trajectories, axis=0)
                ax3d.plot(time_points[:current_time_idx+1], [i] * (current_time_idx+1),
                         mean_traj[:current_time_idx+1], color=colors[i], linewidth=3,
                         label=model_name.replace('_', ' ').title())

        ax3d.set_xlabel('Time')
        ax3d.set_ylabel('Model Type')
        ax3d.set_zlabel('Phenotype')
        ax3d.set_title('3D Model Comparison')
        ax3d.legend()

        plt.tight_layout()
        return fig

    anim3 = animation.FuncAnimation(
        fig, animate_model_comparison,
        frames=len(time_points),
        interval=300,
        blit=False,
        repeat=True
    )

    anim3.save(anim3_dir / 'model_comparison_animation.gif', writer='pillow', fps=3)

    animation_info = {
        'name': 'Model Comparison Animation',
        'directory': str(anim3_dir),
        'description': 'Compares different stochastic models (jump-diffusion, OU, geometric)',
        'frames': len(time_points),
        'time_range': (0, 20),
        'model_types': list(models.keys()),
        'n_models': len(models)
    }
    animations_generated.append(animation_info)
    animation_log.append(f"✅ {animation_info['name']}: {anim3_dir}/model_comparison_animation.gif")

    print(f"   📁 Animation saved to: {anim3_dir}/model_comparison_animation.gif")
    print(f"   📊 {animation_info['frames']} frames, {animation_info['n_models']} model types")

    # Animation 4: Evolutionary dynamics animation
    print("\n🎬 Animation 4: Evolutionary Dynamics Animation")
    print("-" * 50)

    # Create evolutionary data
    np.random.seed(123)
    n_generations = 50
    n_individuals = 200

    evolutionary_data = []
    for gen in range(n_generations):
        # Simulate evolutionary change
        if gen == 0:
            base_phenotype = 10.0
        else:
            # Gradual evolutionary shift
            base_phenotype += np.random.normal(0.1, 0.05)

        # Add variation and selection
        for ind in range(n_individuals):
            phenotype = base_phenotype + np.random.normal(0, 1.5)
            fitness = np.exp(-((phenotype - (base_phenotype + 1))**2) / 4)  # Selection favoring higher values

            evolutionary_data.append({
                'generation': gen,
                'individual': ind,
                'phenotype': phenotype,
                'fitness': fitness
            })

    evo_df = pd.DataFrame(evolutionary_data)
    evo_file = Path("evolutionary_animation_data.csv")
    evo_df.to_csv(evo_file, index=False)

    evo_datacore = DataCore.load_from_csv(
        file_path=evo_file,
        time_column='generation',
        phenotype_columns=['phenotype']
    )

    sampler = EvolutionSampler(evo_datacore)
    analytics = AnalyticsEngine(evo_datacore)

    # Create evolutionary animation
    anim4_dir = base_output_dir / "animation_4_evolutionary_dynamics"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def animate_evolution(frame_idx):
        current_gen = min(frame_idx, n_generations - 1)
        gen_data = evo_df[evo_df['generation'] <= current_gen]

        for ax in axes.flat:
            ax.clear()

        if len(gen_data) > 0:
            # Plot phenotype distribution over generations
            for gen in range(0, current_gen + 1, 5):  # Every 5th generation
                gen_subset = gen_data[gen_data['generation'] == gen]
                if len(gen_subset) > 0:
                    axes[0, 0].hist(gen_subset['phenotype'], bins=20, alpha=0.6,
                                  label=f'Gen {gen}', density=True)

            axes[0, 0].set_xlabel('Phenotype Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title(f'Phenotype Evolution (Generation {current_gen})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot fitness vs phenotype
            current_gen_data = gen_data[gen_data['generation'] == current_gen]
            if len(current_gen_data) > 0:
                axes[0, 1].scatter(current_gen_data['phenotype'], current_gen_data['fitness'],
                                alpha=0.6, c=current_gen_data['phenotype'], cmap='viridis')
                axes[0, 1].set_xlabel('Phenotype')
                axes[0, 1].set_ylabel('Fitness')
                axes[0, 1].set_title('Fitness Landscape')
                axes[0, 1].grid(True, alpha=0.3)

            # Plot mean phenotype over time
            mean_phenotypes = []
            for gen in range(n_generations):
                gen_subset = gen_data[gen_data['generation'] == gen]
                if len(gen_subset) > 0:
                    mean_phenotypes.append(np.mean(gen_subset['phenotype']))

            axes[1, 0].plot(range(len(mean_phenotypes)), mean_phenotypes, linewidth=2)
            axes[1, 0].fill_between(range(len(mean_phenotypes)),
                                  np.array(mean_phenotypes) - np.array(mean_phenotypes).std(),
                                  np.array(mean_phenotypes) + np.array(mean_phenotypes).std(),
                                  alpha=0.3)
            axes[1, 0].axvline(current_gen, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Mean Phenotype')
            axes[1, 0].set_title('Evolutionary Trajectory')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot selection differential
            if current_gen > 0:
                prev_gen_data = gen_data[gen_data['generation'] == current_gen - 1]
                current_gen_data = gen_data[gen_data['generation'] == current_gen]

                if len(prev_gen_data) > 0 and len(current_gen_data) > 0:
                    prev_mean = np.mean(prev_gen_data['phenotype'])
                    current_mean = np.mean(current_gen_data['phenotype'])

                    # Covariance between phenotype and fitness
                    prev_cov = np.cov(prev_gen_data['phenotype'], prev_gen_data['fitness'])[0, 1]
                    current_cov = np.cov(current_gen_data['phenotype'], current_gen_data['fitness'])[0, 1]

                    selection_diff = current_cov * (current_gen - (current_gen - 1))

                    axes[1, 1].bar(['Previous', 'Current'], [prev_cov, current_cov],
                                 color=['blue', 'red'], alpha=0.7)
                    axes[1, 1].set_ylabel('Covariance (Phenotype × Fitness)')
                    axes[1, 1].set_title('Selection Differential')
                    axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    anim4 = animation.FuncAnimation(
        fig, animate_evolution,
        frames=n_generations,
        interval=200,
        blit=False,
        repeat=True
    )

    anim4.save(anim4_dir / 'evolutionary_dynamics_animation.gif', writer='pillow', fps=8)

    animation_info = {
        'name': 'Evolutionary Dynamics Animation',
        'directory': str(anim4_dir),
        'description': 'Shows evolutionary change over generations with fitness landscapes',
        'frames': n_generations,
        'generations': n_generations,
        'n_individuals': n_individuals,
        'evolutionary_process': 'directional_selection'
    }
    animations_generated.append(animation_info)
    animation_log.append(f"✅ {animation_info['name']}: {anim4_dir}/evolutionary_dynamics_animation.gif")

    print(f"   📁 Animation saved to: {anim4_dir}/evolutionary_dynamics_animation.gif")
    print(f"   📊 {animation_info['frames']} frames, {animation_info['generations']} generations")

    # Clean up
    for f in [data_file, evo_file]:
        if f.exists():
            f.unlink()

    # Create comprehensive animation report
    print("\n📋 Creating Comprehensive Animation Report...")
    print("-" * 50)

    animation_report = {
        'timestamp': datetime.now().isoformat(),
        'total_animations': len(animations_generated),
        'base_output_directory': str(base_output_dir),
        'animations': animations_generated,
        'summary': f"Generated {len(animations_generated)} comprehensive animations demonstrating various aspects of developmental and evolutionary processes"
    }

    report_file = base_output_dir / 'comprehensive_animation_report.json'
    with open(report_file, 'w') as f:
        json.dump(animation_report, f, indent=4)

    print(f"   ✅ Animation report saved: {report_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 Comprehensive Animation Demo - COMPLETED!")
    print("=" * 70)

    print(f"📊 Generated {len(animations_generated)} comprehensive animations:")
    for i, anim in enumerate(animations_generated, 1):
        print(f"   {i}. {anim['name']}")
        print(f"      📁 {anim['directory']}")
        print(f"      📝 {anim['description']}")
        print(f"      🖼️ {anim.get('frames', 'N/A')} frames")
        print()

    print("🎬 Animation Types Generated:")
    print("   • Basic developmental trajectory animation")
    print("   • Multi-condition comparative animation")
    print("   • Model comparison animation (jump-diffusion, OU, geometric)")
    print("   • Evolutionary dynamics animation")
    print()

    print("📁 All animations saved as GIF files:")
    for log_entry in animation_log:
        print(f"   • {log_entry}")

    print(f"\n📋 Comprehensive report: {report_file}")
    print("\n🎯 All animations demonstrate:")
    print("   • Real-time trajectory evolution")
    print("   • Cross-sectional distribution changes")
    print("   • Multi-condition comparisons")
    print("   • Model parameter evolution")
    print("   • Evolutionary dynamics over generations")
    print("   • Statistical summaries and fitness landscapes")

    return {
        'status': 'completed',
        'animations_generated': len(animations_generated),
        'animation_directories': [anim['directory'] for anim in animations_generated],
        'report_file': str(report_file),
        'animation_log': animation_log
    }

if __name__ == '__main__':
    results = run_comprehensive_animation_demo()
    print("
✅ Comprehensive animation demo completed successfully!"    print(f"📊 Status: {results['status']}")
    print(f"🎬 Generated: {results['animations_generated']} animations")
    print(f"📁 Output directory: {results['animation_directories'][0] if results['animation_directories'] else 'N/A'}")
    print(f"📋 Report: {results['report_file']}")

    print("\n🎯 To view animations:")
    print("   1. Open any web browser")
    print("   2. Navigate to the output directories listed above")
    print("   3. Open the .gif files to view the animations")
    print("   4. Each animation shows real-time evolution of developmental processes!")

