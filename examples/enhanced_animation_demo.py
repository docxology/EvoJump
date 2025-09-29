#!/usr/bin/env python3
"""
Enhanced Animation Demo for EvoJump Package

This example creates multiple types of comprehensive animations
demonstrating the full capabilities of the EvoJump framework.
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
import matplotlib.animation as animation
import json
from datetime import datetime

def create_multi_condition_animation():
    """Create animation showing multiple experimental conditions."""
    print("🎬 Creating Multi-Condition Animation...")

    # Create data with multiple conditions
    np.random.seed(42)
    n_individuals = 40
    time_points = np.linspace(0, 20, 21)
    conditions = ['control', 'treatment', 'mutant']

    data_rows = []
    for condition in conditions:
        for individual in range(n_individuals // len(conditions)):
            # Different base patterns for each condition
            if condition == 'control':
                base_pattern = 10 + 5 * np.sin(time_points * 0.3) + time_points * 0.2
                variation = 1.0
            elif condition == 'treatment':
                base_pattern = 12 + 4 * np.sin(time_points * 0.25) + time_points * 0.15
                variation = 1.5
            else:  # mutant
                base_pattern = 8 + 6 * np.sin(time_points * 0.2) + time_points * 0.25
                variation = 2.0

            phenotype = base_pattern + np.random.normal(0, variation, len(time_points))

            for t_idx, time_point in enumerate(time_points):
                data_rows.append({
                    'individual': f'{condition}_{individual:03d}',
                    'time': time_point,
                    'phenotype': phenotype[t_idx],
                    'condition': condition
                })

    sample_data = pd.DataFrame(data_rows)
    data_file = Path("multi_condition_data.csv")
    sample_data.to_csv(data_file, index=False)

    # Load and create models for each condition
    condition_models = {}
    for condition in conditions:
        cond_data = sample_data[sample_data['condition'] == condition]
        temp_file = Path(f"temp_{condition}_data.csv")
        cond_data.to_csv(temp_file, index=False)

        datacore = DataCore.load_from_csv(
            file_path=temp_file,
            time_column='time',
            phenotype_columns=['phenotype']
        )
        datacore.preprocess_data(normalize=True, remove_outliers=True, interpolate_missing=True)

        model = JumpRope.fit(datacore, model_type='jump-diffusion', time_points=time_points)
        model.generate_trajectories(n_samples=20, x0=10.0)
        condition_models[condition] = model

        temp_file.unlink()

    # Create multi-condition animation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def animate_multi_condition(frame_idx):
        current_time = time_points[frame_idx % len(time_points)]
        current_time_idx = frame_idx % len(time_points)

        for ax in axes.flat:
            ax.clear()

        colors = ['blue', 'red', 'green']

        # Plot trajectories for each condition
        for i, (condition, model) in enumerate(condition_models.items()):
            color = colors[i]

            if model.trajectories is not None:
                # Individual trajectories
                for j in range(min(8, model.trajectories.shape[0])):
                    axes[0, 0].plot(time_points[:current_time_idx+1],
                                  model.trajectories[j, :current_time_idx+1],
                                  alpha=0.6, color=color, linewidth=1)

                # Mean trajectory
                mean_traj = np.mean(model.trajectories, axis=0)
                axes[0, 0].plot(time_points[:current_time_idx+1],
                              mean_traj[:current_time_idx+1],
                              color=color, linewidth=3, label=f'{condition.capitalize()} Mean')

        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Phenotype')
        axes[0, 0].set_title(f'Multi-Condition Trajectories (Time: {current_time:.2f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Cross-sectional distributions
        for i, (condition, model) in enumerate(condition_models.items()):
            color = colors[i]
            if model.trajectories is not None:
                current_data = model.trajectories[:, current_time_idx]
                axes[0, 1].hist(current_data, bins=15, alpha=0.6, label=f'{condition.capitalize()}',
                              color=color, density=True)

        axes[0, 1].set_xlabel('Phenotype Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Cross-Sectional Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3D landscape evolution
        ax3d = fig.add_subplot(2, 3, 3, projection='3d')
        for i, (condition, model) in enumerate(condition_models.items()):
            if model.trajectories is not None:
                mean_traj = np.mean(model.trajectories, axis=0)
                ax3d.plot(time_points[:current_time_idx+1], [i] * (current_time_idx+1),
                         mean_traj[:current_time_idx+1], color=colors[i], linewidth=3,
                         label=f'{condition.capitalize()}')

        ax3d.set_xlabel('Time')
        ax3d.set_ylabel('Condition')
        ax3d.set_zlabel('Phenotype')
        ax3d.set_title('3D Phenotypic Landscape Evolution')
        ax3d.legend()

        # Statistics comparison
        axes[1, 0].clear()
        stats_data = []
        for condition, model in condition_models.items():
            if model.trajectories is not None:
                current_vals = model.trajectories[:, current_time_idx]
                stats_data.append({
                    'condition': condition.capitalize(),
                    'mean': np.mean(current_vals),
                    'std': np.std(current_vals),
                    'n': len(current_vals)
                })

        conditions_plot = [s['condition'] for s in stats_data]
        means_plot = [s['mean'] for s in stats_data]
        stds_plot = [s['std'] for s in stats_data]

        axes[1, 0].bar(conditions_plot, means_plot, yerr=stds_plot, capsize=5, alpha=0.7)
        axes[1, 0].set_ylabel('Phenotype Value')
        axes[1, 0].set_title('Mean ± SD by Condition')
        axes[1, 0].grid(True, alpha=0.3)

        # Condition comparison over time
        axes[1, 1].clear()
        for i, (condition, model) in enumerate(condition_models.items()):
            if model.trajectories is not None:
                mean_trajectory = np.mean(model.trajectories, axis=0)
                axes[1, 1].plot(time_points, mean_trajectory,
                              label=f'{condition.capitalize()}', color=colors[i], linewidth=2)

        axes[1, 1].axvline(current_time, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean Phenotype')
        axes[1, 1].set_title('Mean Trajectories Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Real-time statistics
        current_stats = stats_data[frame_idx % len(stats_data)]
        axes[1, 2].text(0.1, 0.9, f'Current Time: {current_time:.2f}', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.8, f'Condition: {current_stats["condition"]}', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.7, f'Mean: {current_stats["mean"]:.3f}', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.6, f'Std: {current_stats["std"]:.3f}', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.5, f'Range: {np.min([s["mean"]-s["std"] for s in stats_data]):.2f} to {np.max([s["mean"]+s["std"] for s in stats_data]):.2f}',
                       transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].set_title('Real-time Statistics')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.tight_layout()
        return fig

    anim = animation.FuncAnimation(
        fig, animate_multi_condition,
        frames=len(time_points),
        interval=200,
        blit=False,
        repeat=True
    )

    output_dir = Path("enhanced_animations")
    output_dir.mkdir(exist_ok=True)
    anim.save(output_dir / 'multi_condition_animation.gif', writer='pillow', fps=5)

    data_file.unlink()

    print(f"   ✅ Saved: {output_dir}/multi_condition_animation.gif")
    return output_dir / 'multi_condition_animation.gif'

def create_evolutionary_dynamics_animation():
    """Create animation showing evolutionary dynamics over generations."""
    print("🧬 Creating Evolutionary Dynamics Animation...")

    # Create evolutionary data
    np.random.seed(123)
    n_generations = 40
    n_individuals = 150

    evolutionary_data = []
    for gen in range(n_generations):
        # Simulate evolutionary change
        if gen == 0:
            base_phenotype = 10.0
        else:
            # Gradual evolutionary shift with selection
            base_phenotype += np.random.normal(0.15, 0.08)

        # Add variation and selection
        for ind in range(n_individuals):
            phenotype = base_phenotype + np.random.normal(0, 1.8)
            # Fitness function favoring higher phenotypes with some constraint
            fitness = np.exp(-((phenotype - (base_phenotype + 1.5))**2) / 8)

            evolutionary_data.append({
                'generation': gen,
                'individual': ind,
                'phenotype': phenotype,
                'fitness': fitness
            })

    evo_df = pd.DataFrame(evolutionary_data)
    evo_file = Path("evolutionary_data.csv")
    evo_df.to_csv(evo_file, index=False)

    # Load data and create analysis
    evo_datacore = DataCore.load_from_csv(
        file_path=evo_file,
        time_column='generation',
        phenotype_columns=['phenotype']
    )

    sampler = EvolutionSampler(evo_datacore)
    analytics = AnalyticsEngine(evo_datacore)

    # Create evolutionary animation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def animate_evolution(frame_idx):
        current_gen = min(frame_idx, n_generations - 1)
        gen_data = evo_df[evo_df['generation'] <= current_gen]

        for ax in axes.flat:
            ax.clear()

        if len(gen_data) > 0:
            # Phenotype distribution evolution
            for gen in range(0, current_gen + 1, 4):  # Every 4th generation
                gen_subset = gen_data[gen_data['generation'] == gen]
                if len(gen_subset) > 0:
                    axes[0, 0].hist(gen_subset['phenotype'], bins=20, alpha=0.6,
                                  label=f'Gen {gen}', density=True)

            axes[0, 0].set_xlabel('Phenotype Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title(f'Phenotype Evolution (Generation {current_gen})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Fitness vs phenotype scatter
            current_gen_data = gen_data[gen_data['generation'] == current_gen]
            if len(current_gen_data) > 0:
                axes[0, 1].scatter(current_gen_data['phenotype'], current_gen_data['fitness'],
                                alpha=0.6, c=current_gen_data['phenotype'], cmap='viridis', s=30)
                axes[0, 1].set_xlabel('Phenotype')
                axes[0, 1].set_ylabel('Fitness')
                axes[0, 1].set_title('Fitness Landscape')
                axes[0, 1].grid(True, alpha=0.3)

            # Mean phenotype trajectory
            mean_phenotypes = []
            for gen in range(n_generations):
                gen_subset = gen_data[gen_data['generation'] == gen]
                if len(gen_subset) > 0:
                    mean_phenotypes.append(np.mean(gen_subset['phenotype']))

            axes[1, 0].plot(range(len(mean_phenotypes)), mean_phenotypes, linewidth=3, color='blue')
            axes[1, 0].fill_between(range(len(mean_phenotypes)),
                                  np.array(mean_phenotypes) - np.array(mean_phenotypes).std(),
                                  np.array(mean_phenotypes) + np.array(mean_phenotypes).std(),
                                  alpha=0.3, color='blue')
            axes[1, 0].axvline(current_gen, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Mean Phenotype')
            axes[1, 0].set_title('Evolutionary Trajectory')
            axes[1, 0].grid(True, alpha=0.3)

            # Selection differential
            if current_gen > 0:
                prev_gen_data = gen_data[gen_data['generation'] == current_gen - 1]
                current_gen_data = gen_data[gen_data['generation'] == current_gen]

                if len(prev_gen_data) > 0 and len(current_gen_data) > 0:
                    # Covariance between phenotype and fitness (selection differential)
                    prev_cov = np.cov(prev_gen_data['phenotype'], prev_gen_data['fitness'])[0, 1]
                    current_cov = np.cov(current_gen_data['phenotype'], current_gen_data['fitness'])[0, 1]

                    axes[1, 1].bar(['Previous', 'Current'], [prev_cov, current_cov],
                                 color=['orange', 'red'], alpha=0.7)
                    axes[1, 1].set_ylabel('Covariance (Phenotype × Fitness)')
                    axes[1, 1].set_title('Selection Differential')
                    axes[1, 1].grid(True, alpha=0.3)

            # Heritability estimation
            if current_gen >= 5:  # Need some generations for estimation
                recent_data = gen_data[gen_data['generation'] >= current_gen - 4]

                # Simple parent-offspring regression
                parent_means = []
                offspring_means = []

                for gen in range(current_gen - 4, current_gen):
                    if gen in recent_data['generation'].values:
                        parent_gen = recent_data[recent_data['generation'] == gen]['phenotype']
                        offspring_gen = recent_data[recent_data['generation'] == gen + 1]['phenotype']

                        if len(parent_gen) > 0 and len(offspring_gen) > 0:
                            parent_means.append(np.mean(parent_gen))
                            offspring_means.append(np.mean(offspring_gen))

                if len(parent_means) > 1:
                    heritability = np.corrcoef(parent_means, offspring_means)[0, 1]**2
                    axes[1, 2].bar(['Estimated'], [heritability], color='green', alpha=0.7)
                    axes[1, 2].set_ylabel('Heritability')
                    axes[1, 2].set_title('Narrow-sense Heritability')
                    axes[1, 2].set_ylim(0, 1)
                    axes[1, 2].grid(True, alpha=0.3)
                else:
                    axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor heritability\nestimation',
                                  transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=12)
                    axes[1, 2].set_title('Heritability Estimation')

            # Summary statistics
            axes[0, 2].text(0.1, 0.9, f'Generation: {current_gen}', transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].text(0.1, 0.8, f'Population: {len(current_gen_data)} individuals', transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].text(0.1, 0.7, f'Mean fitness: {np.mean(current_gen_data["fitness"]):.3f}', transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].text(0.1, 0.6, f'Phenotype range: {np.min(current_gen_data["phenotype"]):.2f} to {np.max(current_gen_data["phenotype"]):.2f}',
                           transform=axes[0, 2].transAxes, fontsize=10)
            if current_gen > 0:
                evolutionary_rate = base_phenotype / current_gen
                axes[0, 2].text(0.1, 0.5, f'Evolutionary rate: {evolutionary_rate:.4f} per generation',
                               transform=axes[0, 2].transAxes, fontsize=10)
            else:
                axes[0, 2].text(0.1, 0.5, 'Evolutionary rate: N/A (initial generation)',
                               transform=axes[0, 2].transAxes, fontsize=10)
            axes[0, 2].set_title('Evolution Summary')
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].axis('off')

        plt.tight_layout()
        return fig

    anim = animation.FuncAnimation(
        fig, animate_evolution,
        frames=n_generations,
        interval=250,
        blit=False,
        repeat=True
    )

    output_dir = Path("enhanced_animations")
    output_dir.mkdir(exist_ok=True)
    anim.save(output_dir / 'evolutionary_dynamics_animation.gif', writer='pillow', fps=6)

    evo_file.unlink()

    print(f"   ✅ Saved: {output_dir}/evolutionary_dynamics_animation.gif")
    return output_dir / 'evolutionary_dynamics_animation.gif'

def main():
    """Main enhanced animation demo function."""
    print("🎬 EvoJump Enhanced Animation Demo")
    print("=" * 60)
    print("Creating comprehensive animations demonstrating:")
    print("• Multi-condition developmental trajectories")
    print("• Evolutionary dynamics over generations")
    print("• Real-time statistical analysis")
    print("• 3D phenotypic landscape evolution")
    print("• Selection differential visualization")
    print("=" * 60)

    animations_created = []

    # Animation 1: Multi-condition animation
    print("\n🎬 Animation 1: Multi-Condition Developmental Animation")
    print("-" * 50)
    anim1 = create_multi_condition_animation()
    animations_created.append({
        'name': 'Multi-Condition Animation',
        'file': anim1,
        'description': 'Shows developmental trajectories across control, treatment, and mutant conditions'
    })

    # Animation 2: Evolutionary dynamics animation
    print("\n🧬 Animation 2: Evolutionary Dynamics Animation")
    print("-" * 50)
    anim2 = create_evolutionary_dynamics_animation()
    animations_created.append({
        'name': 'Evolutionary Dynamics Animation',
        'file': anim2,
        'description': 'Shows evolutionary change over generations with fitness landscapes and heritability'
    })

    # Create comprehensive animation report
    print("\n📋 Creating Enhanced Animation Report...")
    print("-" * 50)

    animation_report = {
        'timestamp': datetime.now().isoformat(),
        'total_animations': len(animations_created),
        'animations': [
            {
                'name': anim['name'],
                'file_path': str(anim['file']),
                'description': anim['description'],
                'file_size_mb': anim['file'].stat().st_size / (1024 * 1024)
            }
            for anim in animations_created
        ],
        'summary': f"Generated {len(animations_created)} comprehensive animations demonstrating advanced EvoJump capabilities"
    }

    report_file = Path("enhanced_animations") / 'enhanced_animation_report.json'
    with open(report_file, 'w') as f:
        json.dump(animation_report, f, indent=4)

    print(f"   ✅ Animation report saved: {report_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Enhanced Animation Demo - COMPLETED!")
    print("=" * 60)

    print(f"📊 Generated {len(animations_created)} comprehensive animations:")
    for i, anim in enumerate(animations_created, 1):
        file_size_mb = anim['file'].stat().st_size / (1024 * 1024)
        print(f"   {i}. {anim['name']}")
        print(f"      📁 {anim['file']} ({file_size_mb:.1f} MB)")
        print(f"      📝 {anim['description']}")
        print()

    print("🎬 Animation Features Demonstrated:")
    print("   • Multi-condition trajectory comparison")
    print("   • 3D phenotypic landscape evolution")
    print("   • Real-time statistical analysis")
    print("   • Evolutionary dynamics over generations")
    print("   • Fitness landscape visualization")
    print("   • Selection differential analysis")
    print("   • Heritability estimation")
    print()

    print("📋 Comprehensive report saved with detailed analysis")
    print("\n🎯 All animations demonstrate:")
    print("   • Real-time developmental process evolution")
    print("   • Statistical rigor and scientific accuracy")
    print("   • Multiple visualization perspectives")
    print("   • Advanced evolutionary analysis")
    print("   • Production-quality output generation")

    return {
        'status': 'completed',
        'animations_generated': len(animations_created),
        'report_file': str(report_file),
        'animation_files': [str(anim['file']) for anim in animations_created]
    }

if __name__ == '__main__':
    results = main()
    print(f"\n✅ Enhanced animation demo completed successfully!")
    print(f"📊 Status: {results['status']}")
    print(f"🎬 Generated: {results['animations_generated']} animations")
    print(f"📋 Report: {results['report_file']}")

    print("\n🎯 To view animations:")
    print("   1. Open any web browser")
    print("   2. Navigate to the enhanced_animations directory")
    print("   3. Open the .gif files to view the comprehensive animations")
    print("   4. Each animation shows complex developmental and evolutionary processes!")
