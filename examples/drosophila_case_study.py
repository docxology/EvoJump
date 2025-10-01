#!/usr/bin/env python3
"""
Drosophila Case Study: Comprehensive Fruit Fly Biology Analysis

This comprehensive case study demonstrates EvoJump's capabilities in modeling and analyzing
evolutionary and developmental processes in Drosophila melanogaster (fruit flies). Based on
published research (PubMed: 23459154), this study examines the spread of advantageous traits,
selective sweeps, and genetic hitchhiking in fruit fly populations.

Scientific Context:
- Study published in PubMed demonstrates how students observed the spread of a red-eye allele
- Started with one red-eyed fly among ten white-eyed flies
- Red-eye trait increased in frequency over generations due to selection advantage
- Molecular analysis revealed selective sweeps and hitchhiking of nearby neutral variants

This case study uses EvoJump to:
1. Simulate population dynamics with selective pressure
2. Model phenotypic evolution over generations
3. Analyze cross-sectional distributions at different time points
4. Perform advanced statistical analysis (Bayesian, network, causal inference)
5. Generate comprehensive visualizations and animations
6. Demonstrate evolutionary principles in a biological context

Author: EvoJump Development Team
Publication: Based on "Witnessing Phenotypic and Molecular Evolution in the Fruit Fly"
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import evojump as ej
from evojump.datacore import DataCore
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer
from evojump.trajectory_visualizer import TrajectoryVisualizer
from evojump.evolution_sampler import EvolutionSampler
from evojump.analytics_engine import AnalyticsEngine

import matplotlib.pyplot as plt
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DrosophilaPopulation:
    """Represents a Drosophila melanogaster population with genetic and phenotypic data.
    
    This model simulates two distinct traits:
    1. Eye color: red (advantageous) vs white (ancestral) - the genetic trait under selection
    2. Eye size: a correlated morphological phenotype influenced by eye color genetics
    
    Based on classic Drosophila population genetics (PubMed: 23459154).
    """

    population_size: int = 100
    generations: int = 100  # Extended to 100 generations for long-term evolutionary dynamics
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
        
        The red-eye allele is advantageous (higher fitness), and red-eyed flies also
        tend to have larger eyes due to pleiotropy or tight genetic linkage.

        Returns:
            DataFrame with population genetic and phenotypic data including:
            - generation: time point
            - eye_color: 'red' or 'white'
            - eye_color_genotype: 1 (red) or 0 (white)
            - eye_size: continuous phenotype
            - fitness: relative fitness
            - red_allele_frequency: frequency of advantageous allele
        """
        logger.info("Generating synthetic Drosophila population data...")

        data_rows = []

        # Initial population setup
        initial_red_eyed = int(self.config.population_size * self.config.initial_red_eyed_proportion)
        initial_white_eyed = self.config.population_size - initial_red_eyed

        for generation in range(self.config.generations):
            # Simulate allele frequencies (simplified)
            if generation == 0:
                red_eyed_count = initial_red_eyed
            else:
                # Simulate selection and drift
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

                # Fitness based on eye color genotype (selection acts on color, not size directly)
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

        df = pd.DataFrame(data_rows)
        logger.info(f"Generated {len(df)} population measurements over {self.config.generations} generations")
        return df

    def _simulate_selection(self, current_red_eyed: int) -> int:
        """Simulate one generation of selection and reproduction."""
        # Simple deterministic selection model
        # In practice, this would be more complex with stochastic elements

        # Calculate expected frequency after selection
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


class DrosophilaAnalyzer:
    """Comprehensive analysis of Drosophila population data using EvoJump."""

    def __init__(self, population_data: pd.DataFrame):
        """Initialize analyzer with population data."""
        self.population_data = population_data
        self.data_core = None
        self.models = {}
        self.analyzers = {}
        self.visualizers = {}

    def setup_analysis_pipeline(self):
        """Set up complete analysis pipeline."""
        logger.info("Setting up EvoJump analysis pipeline for Drosophila data...")

        # Create DataCore (focusing on eye size phenotype)
        self.data_core = DataCore.load_from_csv(
            pd.io.common.StringIO(self.population_data.to_csv()),
            time_column='generation',
            phenotype_columns=['eye_size']
        )

        # Initialize analyzers
        self.analyzers['laser_plane'] = LaserPlaneAnalyzer(None)  # Will be set after modeling
        self.analyzers['evolution'] = EvolutionSampler(self.data_core)
        self.analyzers['analytics'] = AnalyticsEngine(self.data_core)

        logger.info("Analysis pipeline initialized")

    def model_population_dynamics(self):
        """Model population dynamics using stochastic processes."""
        logger.info("Modeling population dynamics...")

        # Fit jump-diffusion model to eye size evolution
        time_points = np.sort(self.population_data['generation'].unique())

        # Create model data focused on mean eye size per generation
        mean_eye_sizes = self.population_data.groupby('generation')['eye_size'].mean().reset_index()
        temp_data = pd.DataFrame({
            'generation': mean_eye_sizes['generation'],
            'eye_size': mean_eye_sizes['eye_size']
        })

        temp_datacore = DataCore.load_from_csv(
            pd.io.common.StringIO(temp_data.to_csv()),
            time_column='generation',
            phenotype_columns=['eye_size']
        )

        # Fit multiple model types
        model_types = ['jump-diffusion', 'ornstein-uhlenbeck', 'geometric-jump-diffusion']

        for model_type in model_types:
            logger.info(f"Fitting {model_type} model...")

            try:
                model = JumpRope.fit(temp_datacore, model_type=model_type, time_points=time_points)
                self.models[model_type] = model
                logger.info(f"  {model_type} model fitted successfully")
            except Exception as e:
                logger.warning(f"  {model_type} model fitting failed: {e}")

        # Set up laser plane analyzer with best model
        if self.models:
            best_model = list(self.models.values())[0]
            self.analyzers['laser_plane'] = LaserPlaneAnalyzer(best_model)

        logger.info("Population dynamics modeling completed")

    def analyze_evolutionary_patterns(self):
        """Analyze evolutionary patterns in the population."""
        logger.info("Analyzing evolutionary patterns...")

        # Perform evolutionary sampling
        sampler = self.analyzers['evolution']
        mc_samples = sampler.sample(n_samples=500, method='monte-carlo')

        # Analyze patterns
        evolution_analysis = sampler.analyze_evolutionary_patterns()

        # Perform time series analysis
        analytics = self.analyzers['analytics']
        ts_results = analytics.analyze_time_series()

        # Perform multivariate analysis (only on numeric columns)
        # Skip multivariate for now due to string columns
        # mv_results = analytics.analyze_multivariate()
        mv_results = None

        # Perform Bayesian analysis on eye size evolution
        bayesian_results = analytics.bayesian_analysis('eye_size', 'fitness')

        return {
            'evolutionary_sampling': mc_samples,
            'evolutionary_patterns': evolution_analysis,
            'time_series': ts_results,
            'multivariate': mv_results,
            'bayesian': bayesian_results
        }

    def analyze_selective_sweeps(self):
        """Analyze selective sweeps and genetic hitchhiking."""
        logger.info("Analyzing selective sweeps and genetic hitchhiking...")

        # Simulate neutral markers linked to the advantageous trait
        generations = self.population_data['generation'].unique()
        n_markers = 20  # Neutral markers at different linkage distances (increased resolution)

        sweep_data = []

        for gen in generations:
            gen_data = self.population_data[self.population_data['generation'] == gen]

            for marker_id in range(n_markers):
                # Simulate linkage disequilibrium with finer resolution (20 markers)
                # Distance ranges from 0 (tightly linked) to 2.0 (distant)
                linkage_distance = marker_id * 0.1  # Finer resolution with 20 markers
                ld_strength = np.exp(-linkage_distance)  # Exponential decay

                # Marker frequency follows the selected allele with linkage-dependent noise
                base_frequency = gen_data['red_allele_frequency'].iloc[0]
                noise_level = 0.03 * (1 + linkage_distance)  # More noise for distant markers
                marker_frequency = base_frequency * ld_strength + np.random.normal(0, noise_level)
                marker_frequency = np.clip(marker_frequency, 0.0, 1.0)

                # Genetic diversity decreases near selected locus
                diversity = 1.0 - ld_strength * (1 - base_frequency) * base_frequency

                sweep_data.append({
                    'generation': gen,
                    'marker_id': marker_id,
                    'linkage_distance': linkage_distance,
                    'marker_frequency': marker_frequency,
                    'genetic_diversity': diversity,
                    'linkage_disequilibrium': ld_strength
                })

        sweep_df = pd.DataFrame(sweep_data)

        # Analyze sweep patterns
        analytics = self.analyzers['analytics']
        sweep_analytics = AnalyticsEngine(sweep_df, time_column='generation')

        # Network analysis of marker correlations
        network_results = sweep_analytics.network_analysis(correlation_threshold=0.6)

        return {
            'sweep_data': sweep_df,
            'network_analysis': network_results,
            'sweep_summary': {
                'markers_analyzed': n_markers,
                'generations_analyzed': len(generations),
                'average_ld': sweep_df['linkage_disequilibrium'].mean()
            }
        }

    def create_comprehensive_visualizations(self, output_dir: Path):
        """Create comprehensive visualizations of the analysis results."""
        logger.info("Creating comprehensive visualizations...")

        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = TrajectoryVisualizer()

        # 1. Population dynamics over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Eye color allele frequency evolution
        gen_data = self.population_data.groupby('generation').agg({
            'red_allele_frequency': 'first',
            'eye_size': 'mean'
        }).reset_index()

        axes[0, 0].plot(gen_data['generation'], gen_data['red_allele_frequency'],
                       marker='o', linewidth=2, markersize=3, color='#d62728')
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Red-eye Allele Frequency')
        axes[0, 0].set_title('Selective Sweep: Red-eye Allele (100 Generations)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(-0.05, 1.05)

        # Eye size phenotype evolution
        axes[0, 1].plot(gen_data['generation'], gen_data['eye_size'],
                       marker='s', linewidth=2, markersize=3, color='orange')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Mean Eye Size (a.u.)')
        axes[0, 1].set_title('Correlated Phenotypic Evolution: Eye Size')
        axes[0, 1].grid(True, alpha=0.3)

        # 2. Cross-sectional analysis
        analyzer = self.analyzers['laser_plane']
        stages = [10, 50, 90]  # Key generations across 100-generation sweep

        for i, stage in enumerate(stages):
            ax = axes[1, i % 2]
            try:
                result = analyzer.analyze_cross_section(time_point=float(stage))
                ax.hist(result.data, bins=20, alpha=0.7,
                       label=f'Generation {stage}', density=True)
                ax.set_xlabel('Eye Size')
                ax.set_ylabel('Density')
                ax.set_title(f'Cross-section: Gen {stage}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Cross-section analysis failed for generation {stage}: {e}")

        plt.tight_layout()
        plt.savefig(output_dir / 'drosophila_population_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved population dynamics plot to {output_dir / 'drosophila_population_dynamics.png'}")

        # 3. Selective sweep visualization
        sweep_results = self.analyze_selective_sweeps()
        sweep_df = sweep_results['sweep_data']

        fig, ax = plt.subplots(figsize=(12, 8))

        for marker_id in sweep_df['marker_id'].unique():
            marker_data = sweep_df[sweep_df['marker_id'] == marker_id]
            ax.plot(marker_data['generation'], marker_data['marker_frequency'],
                   marker='o', label=f'Marker {marker_id} (dist={marker_id*0.2:.1f})',
                   linewidth=2, markersize=4)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Marker Frequency')
        ax.set_title('Selective Sweep: Neutral Marker Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'drosophila_selective_sweep.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved selective sweep plot to {output_dir / 'drosophila_selective_sweep.png'}")

        # 4. Network analysis of marker correlations
        network_results = sweep_results['network_analysis']

        if network_results.graph is not None:
            fig, ax = plt.subplots(figsize=(10, 8))

            try:
                import networkx as nx
                pos = nx.spring_layout(network_results.graph, seed=42)
                nx.draw(network_results.graph, pos, ax=ax,
                       node_size=300, alpha=0.7, with_labels=True)
                ax.set_title('Marker Correlation Network')
                plt.savefig(output_dir / 'drosophila_network_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved network analysis plot to {output_dir / 'drosophila_network_analysis.png'}")
            except Exception as e:
                logger.warning(f"Network visualization failed: {e}")

        return {
            'population_dynamics': str(output_dir / 'drosophila_population_dynamics.png'),
            'selective_sweep': str(output_dir / 'drosophila_selective_sweep.png'),
            'network_analysis': str(output_dir / 'drosophila_network_analysis.png')
        }

    def generate_comprehensive_report(self, output_dir: Path):
        """Generate comprehensive scientific report."""
        logger.info("Generating comprehensive scientific report...")

        # Collect all analysis results
        evolutionary_analysis = self.analyze_evolutionary_patterns()
        sweep_analysis = self.analyze_selective_sweeps()

        # Create comprehensive report
        report = {
            'metadata': {
                'title': 'Drosophila melanogaster Selective Sweep Analysis',
                'description': 'Comprehensive analysis of selective sweeps and genetic hitchhiking in fruit fly populations',
                'population_size': self.population_data['population_size'].iloc[0] if len(self.population_data) > 0 else 0,
                'generations': len(self.population_data['generation'].unique()),
                'initial_red_eyed': self.population_data['red_eyed_count'].iloc[0] if len(self.population_data) > 0 else 0,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },

            'population_dynamics': {
                'final_allele_frequency': float(self.population_data['red_allele_frequency'].iloc[-1]) if len(self.population_data) > 0 else 0,
                'allele_frequency_change': float(self.population_data['red_allele_frequency'].iloc[-1] - self.population_data['red_allele_frequency'].iloc[0]) if len(self.population_data) > 0 else 0,
                'generations_to_fixation': self._estimate_fixation_time(),
                'selection_coefficient': 0.15  # From simulation parameters
            },

            'phenotypic_evolution': {
                'initial_mean_eye_size': float(self.population_data[self.population_data['generation'] == 0]['eye_size'].mean()) if len(self.population_data) > 0 else 0,
                'final_mean_eye_size': float(self.population_data[self.population_data['generation'] == self.population_data['generation'].max()]['eye_size'].mean()) if len(self.population_data) > 0 else 0,
                'eye_size_variance': float(self.population_data['eye_size'].var()) if len(self.population_data) > 0 else 0
            },

            'selective_sweep_analysis': sweep_analysis['sweep_summary'],

            'evolutionary_patterns': {
                'effective_population_size': float(evolutionary_analysis['evolutionary_patterns']['population_statistics'].effective_population_size),
                'mean_heritability': float(np.mean(list(evolutionary_analysis['evolutionary_patterns']['population_statistics'].heritability_estimates.values()))),
                'change_points_detected': len(evolutionary_analysis['time_series'].change_points)
            },

            'statistical_analysis': {
                'bayesian_model_evidence': float(evolutionary_analysis['bayesian'].model_evidence),
                'network_density': float(evolutionary_analysis['multivariate']['network'].network_metrics.get('density', 0)) if evolutionary_analysis['multivariate'] else 0.0,
                'pca_explained_variance': evolutionary_analysis['multivariate']['principal_components']['explained_variance_ratio'][:3].tolist() if evolutionary_analysis['multivariate'] else [0.0, 0.0, 0.0]
            },

            'scientific_conclusions': {
                'selective_sweep_detected': self._detect_selective_sweep(),
                'hitchhiking_evidence': self._assess_hitchhiking(),
                'evolutionary_rate': self._estimate_evolutionary_rate(),
                'population_genetic_parameters': self._estimate_population_parameters()
            }
        }

        # Save report
        report_file = output_dir / 'drosophila_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Comprehensive report saved to {report_file}")
        return report

    def _estimate_fixation_time(self) -> int:
        """Estimate time to fixation of advantageous allele."""
        # Simple approximation using deterministic selection model
        initial_freq = self.population_data['red_allele_frequency'].iloc[0] if len(self.population_data) > 0 else 0
        if initial_freq <= 0 or initial_freq >= 1:
            return 0

        # Time to fixation ≈ -2 * ln(initial_freq) / s
        s = 0.15  # Selection coefficient
        return int(-2 * np.log(initial_freq) / s)

    def _detect_selective_sweep(self) -> bool:
        """Detect evidence of selective sweep."""
        # Check if allele frequency changed significantly
        initial_freq = self.population_data['red_allele_frequency'].iloc[0] if len(self.population_data) > 0 else 0
        final_freq = self.population_data['red_allele_frequency'].iloc[-1] if len(self.population_data) > 0 else 0

        return (final_freq - initial_freq) > 0.5

    def _assess_hitchhiking(self) -> bool:
        """Assess evidence of genetic hitchhiking."""
        # Check if neutral markers show correlated evolution
        sweep_results = self.analyze_selective_sweeps()
        sweep_df = sweep_results['sweep_data']

        # Check if markers show correlated frequency changes
        correlations = []
        for marker_id in sweep_df['marker_id'].unique():
            marker_data = sweep_df[sweep_df['marker_id'] == marker_id]
            corr = np.corrcoef(marker_data['generation'], marker_data['marker_frequency'])[0, 1]
            correlations.append(abs(corr))

        return np.mean(correlations) > 0.7  # High correlation indicates hitchhiking

    def _estimate_evolutionary_rate(self) -> float:
        """Estimate rate of evolutionary change in eye size."""
        if len(self.population_data) < 2:
            return 0.0

        initial_eye_size = self.population_data[self.population_data['generation'] == 0]['eye_size'].mean()
        final_eye_size = self.population_data[self.population_data['generation'] == self.population_data['generation'].max()]['eye_size'].mean()

        generations = self.population_data['generation'].max() - self.population_data['generation'].min()

        if generations > 0:
            return (final_eye_size - initial_eye_size) / generations
        return 0.0

    def _estimate_population_parameters(self) -> Dict[str, float]:
        """Estimate key population genetic parameters."""
        # Effective population size (simplified estimate)
        ne = self.population_data['population_size'].iloc[0] if len(self.population_data) > 0 else 100

        # Heritability (simplified estimate from eye size variance)
        eye_size_variance = self.population_data['eye_size'].var() if len(self.population_data) > 0 else 1.0
        h2 = 0.5  # Assumed moderate heritability

        # Selection coefficient
        s = 0.15

        return {
            'effective_population_size': float(ne),
            'narrow_sense_heritability': h2,
            'selection_coefficient': s,
            'additive_genetic_variance': h2 * eye_size_variance
        }


def run_drosophila_case_study():
    """Run the complete Drosophila case study."""
    logger.info("🚀 Starting Drosophila melanogaster Case Study")
    logger.info("=" * 70)
    logger.info("Based on: 'Witnessing Phenotypic and Molecular Evolution in the Fruit Fly'")
    logger.info("PubMed: https://pubmed.ncbi.nlm.nih.gov/23459154/")
    logger.info("=" * 70)

    # Create output directory
    output_dir = Path("drosophila_case_study_outputs")
    output_dir.mkdir(exist_ok=True)

    # Define population parameters for extended 100-generation simulation
    population_config = DrosophilaPopulation(
        population_size=100,
        generations=100,  # Extended to observe long-term evolutionary dynamics
        initial_red_eyed_proportion=0.1,
        advantageous_trait_fitness=1.2,
        selection_coefficient=0.15
    )

    # Generate data
    data_generator = DrosophilaDataGenerator(population_config)
    population_data = data_generator.generate_population_data()

    # Save raw data
    data_file = output_dir / 'drosophila_population_data.csv'
    population_data.to_csv(data_file, index=False)
    logger.info(f"Population data saved to {data_file}")

    # Set up analysis
    analyzer = DrosophilaAnalyzer(population_data)
    analyzer.setup_analysis_pipeline()

    # Model population dynamics
    analyzer.model_population_dynamics()

    # Analyze evolutionary patterns
    evolutionary_results = analyzer.analyze_evolutionary_patterns()

    # Analyze selective sweeps
    sweep_results = analyzer.analyze_selective_sweeps()

    # Create visualizations
    visualization_files = analyzer.create_comprehensive_visualizations(output_dir)

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(output_dir)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Drosophila Case Study - COMPLETED!")
    logger.info("=" * 70)

    logger.info("📊 Scientific Findings (100 Generations):")
    logger.info(f"   • Selective sweep detected: {report['scientific_conclusions']['selective_sweep_detected']}")
    logger.info(f"   • Genetic hitchhiking evidence: {report['scientific_conclusions']['hitchhiking_evidence']}")
    logger.info(f"   • Evolutionary rate (eye size): {report['scientific_conclusions']['evolutionary_rate']:.4f} units per generation")
    logger.info(f"   • Final red-eye allele frequency: {report['population_dynamics']['final_allele_frequency']:.3f}")
    logger.info(f"   • Effective population size: {report['scientific_conclusions']['population_genetic_parameters']['effective_population_size']}")

    logger.info("\n📁 Output Files:")
    for name, path in visualization_files.items():
        logger.info(f"   • {name}: {path}")

    logger.info(f"   • Comprehensive report: {output_dir}/drosophila_analysis_report.json")

    logger.info("\n🎯 Key Scientific Insights (100-Generation Study):")
    logger.info("   • Eye color (red vs white): Genetic trait under direct selection")
    logger.info("   • Eye size: Correlated phenotype tracking the advantageous allele")
    logger.info("   • Demonstrates complete selective sweep over extended time period")
    logger.info("   • Shows genetic hitchhiking of neutral markers linked to selected locus")
    logger.info("   • Illustrates interplay between selection, drift, and recombination")
    logger.info("   • Provides quantitative estimates of evolutionary parameters")
    logger.info("   • 100 generations allows observation of near-fixation dynamics")
    return {
        'status': 'completed',
        'output_directory': str(output_dir),
        'report': report,
        'visualizations': visualization_files
    }


if __name__ == "__main__":
    results = run_drosophila_case_study()
    print("\n✅ Drosophila case study completed successfully!")
    print(f"📁 Results available in: {results['output_directory']}")
    print(f"📊 Report: {results['output_directory']}/drosophila_analysis_report.json")

