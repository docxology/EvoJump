#!/usr/bin/env python3
"""
Generate figures for the Drosophila case study section.

This script creates the figures referenced in the Drosophila case study section
of the EvoJump paper, demonstrating selective sweeps and genetic hitchhiking
in fruit fly populations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import sys
import os
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import EvoJump components
import evojump as ej
from evojump.datacore import DataCore
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer
from evojump.trajectory_visualizer import TrajectoryVisualizer
from evojump.evolution_sampler import EvolutionSampler
from evojump.analytics_engine import AnalyticsEngine

# Set up matplotlib
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Set random seed for reproducible results
np.random.seed(42)


@dataclass
class DrosophilaPopulation:
    """Represents a Drosophila melanogaster population with genetic and phenotypic data.
    
    This model simulates two distinct traits:
    1. Eye color: red (advantageous) vs white (ancestral) - the genetic trait under selection
    2. Eye size: a correlated morphological phenotype influenced by eye color genetics
    """

    population_size: int = 100
    generations: int = 100  # Extended to 100 generations for long-term dynamics
    mutation_rate: float = 0.01
    selection_coefficient: float = 0.15  # 15% selection advantage for red-eyed allele
    recombination_rate: float = 0.1

    # Genetic parameters (eye color)
    initial_red_eyed_proportion: float = 0.1  # Start with 10% red-eyed (advantageous allele)
    advantageous_trait_fitness: float = 1.2  # 20% fitness advantage of red-eye allele

    # Phenotypic parameters (eye size)
    base_eye_size: float = 10.0  # Baseline eye size (arbitrary units)
    trait_effect_size: float = 2.5  # Eye size increase associated with red-eye allele
    environmental_variance: float = 0.8  # Environmental variation in eye size


class DrosophilaDataGenerator:
    """Generates synthetic Drosophila population data for analysis."""

    def __init__(self, population_config: DrosophilaPopulation):
        """Initialize data generator with population configuration."""
        self.config = population_config
        np.random.seed(42)  # For reproducible results

    def generate_population_data(self) -> pd.DataFrame:
        """
        Generate synthetic population data simulating Drosophila evolution.
        
        Simulates two levels:
        1. Genetic: Eye color (red vs white) evolving under selection
        2. Phenotypic: Eye size (continuous trait) correlated with eye color genotype

        Returns:
            DataFrame with population genetic and phenotypic data
        """
        data_rows = []

        # Initial population setup
        initial_red_eyed = int(self.config.population_size * self.config.initial_red_eyed_proportion)

        for generation in range(self.config.generations):
            # Simulate allele frequencies
            if generation == 0:
                red_eyed_count = initial_red_eyed
            else:
                red_eyed_count = self._simulate_selection(red_eyed_count)

            # Generate individual phenotypes
            for individual_id in range(self.config.population_size):
                # Genotype: Eye color (0 = white-eyed/ancestral, 1 = red-eyed/derived)
                eye_color_genotype = 1 if individual_id < red_eyed_count else 0

                # Phenotype: Eye size (correlated with eye color due to pleiotropy/linkage)
                # Red-eyed flies have larger eyes on average
                genetic_effect = eye_color_genotype * self.config.trait_effect_size
                environmental_effect = np.random.normal(0, self.config.environmental_variance)
                eye_size = self.config.base_eye_size + genetic_effect + environmental_effect

                # Fitness based on eye color genotype (selection acts on color, not size)
                fitness = self.config.advantageous_trait_fitness if eye_color_genotype == 1 else 1.0

                data_rows.append({
                    'generation': generation,
                    'individual_id': f'gen_{generation:03d}_ind_{individual_id:03d}',
                    'eye_color': 'red' if eye_color_genotype == 1 else 'white',
                    'eye_color_genotype': eye_color_genotype,
                    'eye_size': eye_size,
                    'fitness': fitness,
                    'population_size': self.config.population_size,
                    'red_eyed_count': red_eyed_count,
                    'red_allele_frequency': red_eyed_count / self.config.population_size
                })

        return pd.DataFrame(data_rows)

    def _simulate_selection(self, current_red_eyed: int) -> int:
        """Simulate one generation of selection and reproduction."""
        current_freq = current_red_eyed / self.config.population_size

        # Selection differential
        mean_fitness = (current_freq * self.config.advantageous_trait_fitness +
                       (1 - current_freq) * 1.0)

        # New frequency after selection
        new_freq = (current_freq * self.config.advantageous_trait_fitness) / mean_fitness

        # Add some genetic drift
        drift = np.random.normal(0, 0.01)
        new_freq = np.clip(new_freq + drift, 0.01, 0.99)

        # Convert back to count
        new_count = int(new_freq * self.config.population_size)

        return new_count


class DrosophilaFigureGenerator:
    """Generate figures for the Drosophila case study."""

    def __init__(self, output_dir: Path):
        """Initialize figure generator."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_selective_sweep_figure(self):
        """Generate Figure showing selective sweep dynamics over 100 generations."""
        print("Generating selective sweep dynamics figure (100 generations)...")

        # Create population data with extended time series
        population_config = DrosophilaPopulation(
            population_size=100,
            generations=100,  # Extended simulation
            initial_red_eyed_proportion=0.1,
            advantageous_trait_fitness=1.2,
            selection_coefficient=0.15
        )

        # Generate data using the Drosophila data generator
        data_generator = DrosophilaDataGenerator(population_config)
        population_data = data_generator.generate_population_data()

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot allele frequency evolution
        gen_data = population_data.groupby('generation').agg({
            'red_allele_frequency': 'first',
            'eye_size': 'mean'
        }).reset_index()

        ax.plot(gen_data['generation'], gen_data['red_allele_frequency'],
               marker='o', linewidth=2, markersize=3, label='Red-eye Allele Frequency',
               color='#d62728')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Fixation threshold')
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Red-eyed Allele Frequency', fontsize=11)
        ax.set_title('Selective Sweep Dynamics: Red-eye Allele Over 100 Generations', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_ylim(-0.05, 1.05)

        # Save figure
        output_path = self.output_dir / 'figure_drosophila_sweep.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {output_path}")
        return output_path

    def generate_network_analysis_figure(self):
        """Generate Figure showing network analysis of marker correlations."""
        print("Generating network analysis figure (neutral markers)...")

        # Create synthetic sweep data with neutral markers at different linkage distances
        generations = np.arange(0, 101, 5)  # Sample every 5 generations
        n_markers = 20  # Neutral markers at varying distances from selected locus (increased resolution)

        sweep_data = []
        for gen in generations:
            for marker_id in range(n_markers):
                # Simulate linkage disequilibrium with finer resolution
                # Distance ranges from 0 (tightly linked) to 2.0 (distant)
                linkage_distance = marker_id * 0.1
                ld_strength = np.exp(-linkage_distance)

                # Marker frequency follows selected allele with noise
                base_frequency = 0.1 + 0.8 * (1 - np.exp(-0.1 * gen))  # Sweep dynamics
                # Add linkage-dependent noise (more noise for distant markers)
                noise_level = 0.03 * (1 + linkage_distance)
                marker_frequency = base_frequency * ld_strength + np.random.normal(0, noise_level)
                marker_frequency = np.clip(marker_frequency, 0.0, 1.0)

                sweep_data.append({
                    'generation': gen,
                    'marker_id': marker_id,
                    'linkage_distance': linkage_distance,
                    'marker_frequency': marker_frequency,
                    'genetic_diversity': 1.0 - ld_strength * (1 - base_frequency) * base_frequency
                })

        sweep_df = pd.DataFrame(sweep_data)

        # Create correlation matrix
        correlation_matrix = np.zeros((n_markers, n_markers))
        for i in range(n_markers):
            for j in range(n_markers):
                marker1_data = sweep_df[sweep_df['marker_id'] == i]['marker_frequency']
                marker2_data = sweep_df[sweep_df['marker_id'] == j]['marker_frequency']
                if len(marker1_data) > 0 and len(marker2_data) > 0:
                    correlation_matrix[i, j] = np.corrcoef(marker1_data, marker2_data)[0, 1]

        # Create network graph
        G = nx.Graph()
        for i in range(n_markers):
            G.add_node(f'Marker_{i}', distance=sweep_df[sweep_df['marker_id'] == i]['linkage_distance'].iloc[0])

        # Add edges based on correlation (stricter threshold for clarity with 20 markers)
        for i in range(n_markers):
            for j in range(i+1, n_markers):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # Higher threshold for 20 markers
                    G.add_edge(f'Marker_{i}', f'Marker_{j}', weight=abs(corr))

        # Create plot with larger size for 20 markers
        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        node_sizes = [200 + 150 * G.nodes[node]['distance'] for node in G.nodes()]

        # Draw network with labels only for every other marker for clarity
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_size=node_sizes,
                              node_color=[G.nodes[node]['distance'] for node in G.nodes()],
                              cmap='viridis',
                              alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edge_color='gray',
                              alpha=0.4,
                              width=1.5)
        
        # Label only markers 0, 5, 10, 15, 19 for clarity
        labels_to_show = {f'Marker_{i}': f'M{i}' for i in [0, 5, 10, 15, 19]}
        labels_dict = {node: labels_to_show.get(node, '') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels_dict, ax=ax,
                               font_size=9,
                               font_weight='bold')

        ax.set_title('Marker Correlation Network (20 Neutral Markers)\n(Color indicates linkage distance from selected locus)', 
                    fontsize=12, fontweight='bold')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=0, vmax=max([G.nodes[node]['distance'] for node in G.nodes()])))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Linkage Distance (cM)', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        # Save figure
        output_path = self.output_dir / 'figure_drosophila_network.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {output_path}")
        return output_path

    def generate_cross_sections_figure(self):
        """Generate Figure showing cross-sectional distributions of eye size."""
        print("Generating cross-sectional eye size distributions (100 generations)...")

        # Create model data
        population_config = DrosophilaPopulation(
            population_size=100,
            generations=100,
            initial_red_eyed_proportion=0.1,
            advantageous_trait_fitness=1.2
        )

        data_generator = DrosophilaDataGenerator(population_config)
        population_data = data_generator.generate_population_data()

        # Create model data focused on mean eye size per generation
        mean_eye_sizes = population_data.groupby('generation')['eye_size'].mean().reset_index()
        temp_data = pd.DataFrame({
            'generation': mean_eye_sizes['generation'],
            'eye_size': mean_eye_sizes['eye_size']
        })

        temp_datacore = DataCore.load_from_csv(
            pd.io.common.StringIO(temp_data.to_csv()),
            time_column='generation',
            phenotype_columns=['eye_size']
        )

        # Fit jump-diffusion model
        time_points = np.sort(population_data['generation'].unique())
        model = JumpRope.fit(temp_datacore, model_type='jump-diffusion', time_points=time_points)
        model.generate_trajectories(n_samples=50, x0=10.0)

        # Create analyzer
        analyzer = LaserPlaneAnalyzer(model)

        # Analyze cross-sections at key stages across 100 generations
        stages = [10, 50, 90]  # Early, mid, and late sweep
        fig, axes = plt.subplots(1, len(stages), figsize=(15, 5))

        for i, stage in enumerate(stages):
            result = analyzer.analyze_cross_section(time_point=float(stage))
            axes[i].hist(result.data, bins=25, alpha=0.7, density=True, color='steelblue')
            axes[i].axvline(result.moments['mean'], color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {result.moments['mean']:.2f}")
            axes[i].set_xlabel('Eye Size (arbitrary units)', fontsize=10)
            axes[i].set_ylabel('Probability Density', fontsize=10)
            axes[i].set_title(f'Generation {stage}', fontsize=11, fontweight='bold')
            axes[i].legend(loc='upper right', fontsize=9)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_drosophila_cross_sections.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {output_path}")
        return output_path

    def generate_all_figures(self):
        """Generate all figures for the Drosophila case study."""
        print("🎨 Generating Drosophila case study figures...")

        figures = {}

        # Generate each figure
        figures['sweep'] = self.generate_selective_sweep_figure()
        figures['network'] = self.generate_network_analysis_figure()
        figures['cross_sections'] = self.generate_cross_sections_figure()

        print("✅ All figures generated successfully!")

        # Create summary
        summary = {
            'figures_generated': len(figures),
            'output_directory': str(self.output_dir),
            'figure_paths': {k: str(v) for k, v in figures.items()},
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Save summary
        summary_file = self.output_dir / 'drosophila_figures_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Summary saved to: {summary_file}")

        return figures


def main():
    """Main function to generate all Drosophila figures."""
    print("🚀 Drosophila Case Study Figure Generation")
    print("=" * 50)

    # Create output directory (use relative path from current script location)
    script_dir = Path(__file__).parent
    output_dir = script_dir / "figures"

    # Generate figures
    generator = DrosophilaFigureGenerator(output_dir)
    figures = generator.generate_all_figures()

    print("\n" + "=" * 50)
    print("🎉 Figure Generation Complete!")
    print("=" * 50)

    print(f"📊 Generated {len(figures)} figures:")
    for name, path in figures.items():
        print(f"   • {name}: {path}")

    print(f"\n📁 All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
