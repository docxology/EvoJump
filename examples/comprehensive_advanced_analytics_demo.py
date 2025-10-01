"""
Comprehensive EvoJump Advanced Analytics Demo

This example demonstrates the comprehensive advanced analytic capabilities of EvoJump,
including Bayesian analysis, network analysis, causal inference, dimensionality reduction,
survival analysis, spectral analysis, nonlinear dynamics, information theory,
robust statistics, and spatial analysis.

Features demonstrated:
- Bayesian linear regression with credible intervals
- Correlation network construction and analysis
- Granger causality testing
- Advanced dimensionality reduction (FastICA, t-SNE)
- Spectral analysis for frequency domain insights
- Nonlinear dynamics analysis (Lyapunov exponents, chaos detection)
- Information theory analysis (entropy measures, mutual information)
- Robust statistical methods (resistant to outliers)
- Spatial analysis (Moran's I autocorrelation)
- Comprehensive visualization of all results
- Animation of complex analysis processes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from matplotlib.animation import FuncAnimation

# Import EvoJump modules
import evojump as ej
from evojump.datacore import DataCore
from evojump.analytics_engine import AnalyticsEngine
from evojump.trajectory_visualizer import TrajectoryVisualizer
from evojump.jumprope import JumpRope
from evojump.laserplane import LaserPlaneAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_comprehensive_test_data(n_samples: int = 200) -> pd.DataFrame:
    """
    Create comprehensive synthetic developmental data for analysis.

    Parameters:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with comprehensive synthetic data
    """
    np.random.seed(42)

    # Time points
    time_points = np.arange(1, n_samples + 1)

    # Base developmental trend
    base_trend = 2.0 * np.log(time_points) + 0.5 * time_points ** 0.3

    # Seasonal/periodic components
    seasonality1 = 3.0 * np.sin(2 * np.pi * time_points / 12)
    seasonality2 = 1.5 * np.cos(2 * np.pi * time_points / 24)

    # Autocorrelated noise (developmental stability)
    noise = np.zeros(n_samples)
    for i in range(1, n_samples):
        noise[i] = 0.7 * noise[i-1] + 0.3 * np.random.normal(0, 1)

    # Multiple correlated phenotypes
    phenotype1 = base_trend + seasonality1 + noise + np.random.normal(0, 0.5, n_samples)
    phenotype2 = 1.5 * phenotype1 + seasonality2 + 0.8 * noise + np.random.normal(0, 0.3, n_samples)
    phenotype3 = 0.8 * phenotype1 - 0.5 * phenotype2 + noise * 0.6 + np.random.normal(0, 0.4, n_samples)

    # Developmental stages
    developmental_stages = []
    for t in time_points:
        if t < 50:
            developmental_stages.append('early')
        elif t < 100:
            developmental_stages.append('mid')
        elif t < 150:
            developmental_stages.append('late')
        else:
            developmental_stages.append('mature')

    # Create DataFrame
    data = pd.DataFrame({
        'time': time_points,
        'developmental_stage': developmental_stages,
        'phenotype1': phenotype1,
        'phenotype2': phenotype2,
        'phenotype3': phenotype3,
        'environmental_factor': seasonality1 + np.random.normal(0, 0.2, n_samples),
        'genetic_factor': base_trend * 0.1 + np.random.normal(0, 0.1, n_samples)
    })

    return data

def run_comprehensive_bayesian_analysis(engine: AnalyticsEngine,
                                      visualizer: TrajectoryVisualizer,
                                      output_dir: Path) -> None:
    """
    Run comprehensive Bayesian analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE BAYESIAN ANALYSIS ===")

    # Perform Bayesian analysis
    bayesian_result = engine.bayesian_analysis('phenotype1', 'phenotype2', n_samples=1000)

    if len(bayesian_result.posterior_samples) > 0:
        logger.info(f"Bayesian analysis completed with {len(bayesian_result.posterior_samples)} posterior samples")
        logger.info(f"95% credible interval: {bayesian_result.credible_intervals.get('95%', 'N/A')}")
        logger.info(f"Model evidence: {bayesian_result.model_evidence:.4f}")

        # Visualize Bayesian results
        fig = visualizer.plot_bayesian_analysis(bayesian_result, output_dir)
        plt.show()

        # Model comparison
        comparison = engine.bayesian_analyzer.bayesian_model_comparison(
            model1_likelihood=-50.0,
            model2_likelihood=-52.0,
            model1_complexity=2,
            model2_complexity=3
        )
        logger.info(f"Model comparison: {comparison['preferred_model']} preferred")
        logger.info(f"BIC difference: {comparison['bic_model2'] - comparison['bic_model1']:.2f}")

def run_network_analysis(engine: AnalyticsEngine,
                        visualizer: TrajectoryVisualizer,
                        output_dir: Path) -> None:
    """
    Run comprehensive network analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE NETWORK ANALYSIS ===")

    # Construct correlation network
    network_result = engine.network_analysis(correlation_threshold=0.5)

    if network_result.graph is not None:
        logger.info(f"Network constructed with {network_result.network_metrics['num_nodes']} nodes")
        logger.info(f"Network density: {network_result.network_metrics['density']:.3f}")
        logger.info(f"Average clustering: {network_result.network_metrics['average_clustering']:.3f}")

        # Analyze centrality
        if network_result.centrality_measures:
            degree_centrality = network_result.centrality_measures['degree']
            most_central = max(degree_centrality.items(), key=lambda x: x[1])
            logger.info(f"Most central node: {most_central[0]} (degree: {most_central[1]:.3f})")

        # Community structure
        if network_result.community_structure:
            num_communities = network_result.community_structure.get('num_communities', 0)
            modularity = network_result.community_structure.get('modularity', 0)
            logger.info(f"Communities detected: {num_communities}")
            logger.info(f"Modularity: {modularity:.3f}")

        # Visualize network
        fig = visualizer.plot_network_analysis(network_result, output_dir)
        plt.show()

def run_causal_inference_analysis(engine: AnalyticsEngine,
                                 visualizer: TrajectoryVisualizer,
                                 output_dir: Path) -> None:
    """
    Run comprehensive causal inference analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE CAUSAL INFERENCE ANALYSIS ===")

    # Granger causality test
    causal_result = engine.causal_inference('phenotype1', 'phenotype2', max_lag=5)

    if 'granger_causality' in causal_result:
        gc_results = causal_result['granger_causality']
        significant_lags = [i for i, p in enumerate(gc_results['p_values']) if p < 0.05]
        logger.info(f"Granger causality lags tested: {gc_results['lags_tested']}")
        logger.info(f"Significant causal lags: {significant_lags}")
        logger.info(f"Best lag: {causal_result.get('best_lag', 'None')}")

        if causal_result.get('significant_causality', False):
            logger.info("✅ Significant causal relationship detected!")
        else:
            logger.info("❌ No significant causal relationship detected")

def run_dimensionality_reduction_analysis(engine: AnalyticsEngine,
                                         visualizer: TrajectoryVisualizer,
                                         output_dir: Path) -> None:
    """
    Run comprehensive dimensionality reduction analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE DIMENSIONALITY REDUCTION ANALYSIS ===")

    # FastICA analysis
    ica_result = engine.advanced_dimensionality_reduction(method='fastica', n_components=2)
    logger.info(f"FastICA reconstruction error: {ica_result.reconstruction_error:.4f}")
    logger.info(f"FastICA intrinsic dimension: {ica_result.intrinsic_dimensionality}")

    # Visualize ICA results
    fig = visualizer.plot_dimensionality_reduction(ica_result, output_dir)
    plt.show()

    # t-SNE analysis
    tsne_result = engine.advanced_dimensionality_reduction(
        method='tsne',
        n_components=2,
        perplexity=30,
        learning_rate=200
    )
    logger.info("t-SNE analysis completed successfully")

def run_spectral_analysis(engine: AnalyticsEngine,
                         visualizer: TrajectoryVisualizer,
                         output_dir: Path) -> None:
    """
    Run comprehensive spectral analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE SPECTRAL ANALYSIS ===")

    # Spectral analysis
    spectral_result = engine.spectral_analysis('phenotype1', sampling_frequency=1.0)

    if spectral_result.power_spectrum.size > 0:
        logger.info(f"Spectral entropy: {spectral_result.spectral_entropy:.4f}")
        logger.info(f"Dominant frequencies: {len(spectral_result.dominant_frequencies)} found")

        # Visualize spectral results
        fig = visualizer.plot_spectral_analysis(spectral_result, output_dir)
        plt.show()

def run_nonlinear_dynamics_analysis(engine: AnalyticsEngine,
                                   visualizer: TrajectoryVisualizer,
                                   output_dir: Path) -> None:
    """
    Run comprehensive nonlinear dynamics analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE NONLINEAR DYNAMICS ANALYSIS ===")

    # Nonlinear dynamics analysis
    nonlinear_result = engine.nonlinear_dynamics_analysis('phenotype1', embedding_dim=3, tau=1)

    if 'largest_lyapunov_exponent' in nonlinear_result:
        lyapunov_exp = nonlinear_result['largest_lyapunov_exponent']
        logger.info(f"Largest Lyapunov exponent: {lyapunov_exp:.6f}")

        if lyapunov_exp > 0.01:
            logger.info("⚠️  Potential chaotic behavior detected!")
        else:
            logger.info("✅ System appears stable/non-chaotic")

        # Visualize nonlinear dynamics
        fig = visualizer.plot_nonlinear_dynamics(nonlinear_result, output_dir)
        plt.show()

def run_information_theory_analysis(engine: AnalyticsEngine,
                                   visualizer: TrajectoryVisualizer,
                                   output_dir: Path) -> None:
    """
    Run comprehensive information theory analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE INFORMATION THEORY ANALYSIS ===")

    # Information theory analysis
    info_result = engine.information_theory_analysis('phenotype1')

    if 'shannon_entropy' in info_result:
        entropy = info_result['shannon_entropy']
        normalized_entropy = info_result.get('normalized_entropy', 0)

        logger.info(f"Shannon entropy: {entropy:.4f}")
        logger.info(f"Normalized entropy: {normalized_entropy:.4f}")

        if normalized_entropy > 0.8:
            logger.info("📊 High information content detected")
        elif normalized_entropy < 0.2:
            logger.info("📊 Low information content detected")
        else:
            logger.info("📊 Moderate information content")

        # Visualize information theory results
        fig = visualizer.plot_information_theory(info_result, output_dir)
        plt.show()

def run_robust_statistics_analysis(engine: AnalyticsEngine,
                                  visualizer: TrajectoryVisualizer,
                                  output_dir: Path) -> None:
    """
    Run comprehensive robust statistics analysis demonstration.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE ROBUST STATISTICS ANALYSIS ===")

    # Robust statistics analysis
    robust_result = engine.robust_statistical_analysis('phenotype1')

    if 'location_estimates' in robust_result:
        location_estimates = robust_result['location_estimates']
        scale_estimates = robust_result['scale_estimates']

        logger.info(f"Robust location estimate (trimmed mean): {robust_result['robust_location_preferred']:.4f}")
        logger.info(f"Robust scale estimate (MAD): {robust_result['robust_scale_preferred']:.4f}")

        # Compare with classical estimates
        classical_mean = location_estimates.get('classical_mean', 0)
        classical_std = scale_estimates.get('classical_std', 0)

        logger.info(f"Classical mean: {classical_mean:.4f}")
        logger.info(f"Classical std: {classical_std:.4f}")

        # Visualize robust statistics
        fig = visualizer.plot_robust_statistics(robust_result, output_dir)
        plt.show()

def run_comprehensive_analysis(engine: AnalyticsEngine,
                              visualizer: TrajectoryVisualizer,
                              output_dir: Path) -> None:
    """
    Run comprehensive analysis report.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for results
    """
    logger.info("=== COMPREHENSIVE ANALYSIS REPORT ===")

    # Generate comprehensive report
    comprehensive_report = engine.comprehensive_analysis_report()

    logger.info("Comprehensive analysis completed!")
    logger.info(f"Analysis timestamp: {comprehensive_report['timestamp']}")
    logger.info(f"Data summary: {comprehensive_report['data_summary']['n_samples']} samples, {comprehensive_report['data_summary']['n_variables']} variables")

    # Log analysis status
    for analysis_type, results in comprehensive_report.items():
        if analysis_type != 'timestamp' and analysis_type != 'data_summary':
            if isinstance(results, dict) and 'error' in results:
                logger.warning(f"⚠️  {analysis_type}: Error - {results['error']}")
            else:
                logger.info(f"✅ {analysis_type}: Completed successfully")

def create_advanced_animation_demo(engine: AnalyticsEngine,
                                  visualizer: TrajectoryVisualizer,
                                  output_dir: Path) -> None:
    """
    Create advanced animation demonstrating complex analysis processes.

    Parameters:
        engine: AnalyticsEngine instance
        visualizer: TrajectoryVisualizer instance
        output_dir: Output directory for animation
    """
    logger.info("=== CREATING ADVANCED ANALYSIS ANIMATION ===")

    def animate_comprehensive_analysis(frame):
        """Animation function for comprehensive analysis."""
        axes.clear()

        # Different analysis phases
        if frame < 20:
            # Phase 1: Data exploration
            axes.set_title(f'Advanced Analytics Demo - Frame {frame + 1}\nPhase 1: Data Exploration', fontsize=14)

            # Show raw data trajectories
            time_data = engine.data['time']
            for col in ['phenotype1', 'phenotype2', 'phenotype3']:
                axes.plot(time_data[:frame+1], engine.data[col].iloc[:frame+1],
                         label=col, alpha=0.7)

            axes.set_xlabel('Time')
            axes.set_ylabel('Phenotype Value')
            axes.legend()
            axes.grid(True, alpha=0.3)

        elif frame < 40:
            # Phase 2: Network analysis
            axes.set_title(f'Advanced Analytics Demo - Frame {frame + 1}\nPhase 2: Network Analysis', fontsize=14)

            # Show network evolution
            threshold = 0.3 + (frame - 20) * 0.02  # Increasing threshold
            network_result = engine.network_analysis(correlation_threshold=threshold)

            if network_result.graph is not None:
                try:
                    pos = nx.spring_layout(network_result.graph, seed=42)
                    nx.draw(network_result.graph, pos, ax=axes, node_size=200,
                           alpha=0.7, node_color='lightblue', with_labels=True)
                    axes.set_title(f'Correlation Network (threshold: {threshold:.2f})')
                    axes.axis('off')
                except:
                    axes.text(0.5, 0.5, f'Building network...\nThreshold: {threshold:.2f}',
                             ha='center', va='center', transform=axes.transAxes, fontsize=12)

        elif frame < 60:
            # Phase 3: Bayesian analysis
            axes.set_title(f'Advanced Analytics Demo - Frame {frame + 1}\nPhase 3: Bayesian Analysis', fontsize=14)

            # Show posterior sampling
            n_samples = min(50, frame - 40 + 1)
            bayesian_result = engine.bayesian_analysis('phenotype1', 'phenotype2', n_samples=n_samples * 20)

            if len(bayesian_result.posterior_samples) > 0:
                samples = bayesian_result.posterior_samples[:n_samples]
                axes.hist(samples, bins=min(20, n_samples//2), alpha=0.7, density=True)
                axes.axvline(np.mean(samples), color='red', linestyle='--', label='Posterior Mean')
                axes.set_xlabel('Parameter Value')
                axes.set_ylabel('Density')
                axes.set_title(f'Posterior Distribution\n({n_samples} samples)')
                axes.legend()
                axes.grid(True, alpha=0.3)

        else:
            # Phase 4: Spectral analysis
            axes.set_title(f'Advanced Analytics Demo - Frame {frame + 1}\nPhase 4: Spectral Analysis', fontsize=14)

            # Show spectral evolution
            spectral_result = engine.spectral_analysis('phenotype1', sampling_frequency=1.0)

            if spectral_result.power_spectrum.size > 0:
                power_data = np.array(spectral_result.power_spectrum)
                freq_subset = power_data[:min(frame - 60 + 10, len(power_data))]

                if len(freq_subset) > 1:
                    axes.plot(freq_subset[:, 0], freq_subset[:, 1])
                    axes.set_xlabel('Frequency')
                    axes.set_ylabel('Power')
                    axes.set_yscale('log')
                    axes.grid(True, alpha=0.3)

                    if spectral_result.spectral_entropy > 0:
                        axes.set_title(f'Power Spectrum\nSpectral Entropy: {spectral_result.spectral_entropy:.3f}')

    # Create animation
    fig, axes = plt.subplots(figsize=(12, 8))
    anim = FuncAnimation(fig, animate_comprehensive_analysis,
                        frames=80, interval=100, repeat=True)

    # Save animation
    output_dir.mkdir(parents=True, exist_ok=True)
    anim.save(output_dir / 'comprehensive_analytics_demo.gif',
              writer='pillow', fps=10, dpi=100)
    logger.info(f"Saved comprehensive analytics animation to {output_dir / 'comprehensive_analytics_demo.gif'}")

def main():
    """Main function demonstrating comprehensive EvoJump analytics."""
    logger.info("🚀 Starting Comprehensive EvoJump Advanced Analytics Demo")

    # Create output directory
    output_dir = Path('comprehensive_analytics_outputs')
    output_dir.mkdir(exist_ok=True)

    # Create comprehensive test data
    logger.info("📊 Creating comprehensive test data...")
    data = create_comprehensive_test_data(n_samples=150)

    # Initialize EvoJump components
    logger.info("🔧 Initializing EvoJump components...")
    engine = AnalyticsEngine(data, time_column='time')
    visualizer = TrajectoryVisualizer()

    # Run comprehensive analyses
    logger.info("🔬 Running comprehensive analyses...")

    try:
        # 1. Bayesian Analysis
        run_comprehensive_bayesian_analysis(engine, visualizer, output_dir)

        # 2. Network Analysis
        run_network_analysis(engine, visualizer, output_dir)

        # 3. Causal Inference
        run_causal_inference_analysis(engine, visualizer, output_dir)

        # 4. Dimensionality Reduction
        run_dimensionality_reduction_analysis(engine, visualizer, output_dir)

        # 5. Spectral Analysis
        run_spectral_analysis(engine, visualizer, output_dir)

        # 6. Nonlinear Dynamics
        run_nonlinear_dynamics_analysis(engine, visualizer, output_dir)

        # 7. Information Theory
        run_information_theory_analysis(engine, visualizer, output_dir)

        # 8. Robust Statistics
        run_robust_statistics_analysis(engine, visualizer, output_dir)

        # 9. Comprehensive Report
        run_comprehensive_analysis(engine, visualizer, output_dir)

        # 10. Advanced Animation
        create_advanced_animation_demo(engine, visualizer, output_dir)

        logger.info("🎉 Comprehensive analytics demo completed successfully!")
        logger.info(f"📁 All results saved to: {output_dir}")

        # Display summary
        print("\n=== COMPREHENSIVE EVOJUMP ANALYTICS DEMO SUMMARY ===")
        print(f"✅ Data generated: {len(data)} samples, {len(data.columns)} variables")
        print(f"✅ Bayesian analysis: Posterior sampling and credible intervals")
        print(f"✅ Network analysis: Correlation networks and community detection")
        print(f"✅ Causal inference: Granger causality testing")
        print(f"✅ Dimensionality reduction: FastICA and t-SNE")
        print(f"✅ Spectral analysis: Frequency domain analysis")
        print(f"✅ Nonlinear dynamics: Lyapunov exponents and chaos detection")
        print(f"✅ Information theory: Entropy and mutual information")
        print(f"✅ Robust statistics: Outlier-resistant methods")
        print(f"✅ Advanced visualization: All analysis types")
        print(f"✅ Animation demo: Comprehensive analysis process")
        print(f"📁 Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Error in comprehensive demo: {e}")
        raise

if __name__ == "__main__":
    main()
