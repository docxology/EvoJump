"""
Command Line Interface for EvoJump Package

This module provides a command-line interface for the EvoJump package,
allowing users to perform evolutionary ontogenetic analysis from the command line.
"""

import argparse
import sys
import logging
from typing import Optional, List
from pathlib import Path

from . import datacore
from . import jumprope
from . import laserplane
from . import trajectory_visualizer
from . import evolution_sampler
from . import analytics_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="EvoJump: Evolutionary Ontogenetic Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and analyze developmental data
  evojump-cli analyze data.csv --output results/

  # Fit jump rope model
  evojump-cli fit data.csv --model-type jump-diffusion --output model.pkl

  # Visualize trajectories
  evojump-cli visualize model.pkl --output plots/

  # Perform evolutionary sampling
  evojump-cli sample population.csv --samples 1000 --output samples.csv
        """
    )

    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (use -v, -vv, or -vvv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file (YAML format)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze developmental trajectories'
    )
    analyze_parser.add_argument(
        'data_file',
        type=Path,
        help='Input data file'
    )
    analyze_parser.add_argument(
        '--model-type',
        choices=['jump-diffusion', 'ornstein-uhlenbeck', 'compound-poisson'],
        default='jump-diffusion',
        help='Type of stochastic process model'
    )
    analyze_parser.add_argument(
        '--time-column',
        default='time',
        help='Name of time column in data'
    )
    analyze_parser.add_argument(
        '--phenotype-columns',
        nargs='+',
        help='Names of phenotype columns to analyze'
    )

    # Fit command
    fit_parser = subparsers.add_parser(
        'fit',
        help='Fit stochastic process model to data'
    )
    fit_parser.add_argument(
        'data_file',
        type=Path,
        help='Input data file'
    )
    fit_parser.add_argument(
        '--model-type',
        choices=['jump-diffusion', 'ornstein-uhlenbeck', 'compound-poisson'],
        default='jump-diffusion',
        help='Type of stochastic process model'
    )
    fit_parser.add_argument(
        '--output-model',
        type=Path,
        help='Output file for fitted model'
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize developmental trajectories'
    )
    visualize_parser.add_argument(
        'model_file',
        type=Path,
        help='Input model file'
    )
    visualize_parser.add_argument(
        '--plot-type',
        choices=['trajectories', 'cross-sections', 'landscapes', 'animation'],
        default='trajectories',
        help='Type of visualization'
    )
    visualize_parser.add_argument(
        '--interactive',
        action='store_true',
        help='Create interactive plots'
    )

    # Sample command
    sample_parser = subparsers.add_parser(
        'sample',
        help='Sample from evolutionary populations'
    )
    sample_parser.add_argument(
        'population_file',
        type=Path,
        help='Input population data file'
    )
    sample_parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    sample_parser.add_argument(
        '--method',
        choices=['monte-carlo', 'importance-sampling', 'mcmc'],
        default='monte-carlo',
        help='Sampling method'
    )

    return parser


def setup_logging(verbosity: int) -> None:
    """Set up logging based on verbosity level."""
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.getLogger().setLevel(level)

    # Configure specific loggers
    logging.getLogger('evojump').setLevel(level)


def analyze_command(args: argparse.Namespace) -> int:
    """Handle the analyze command."""
    try:
        logger.info(f"Loading data from {args.data_file}")

        # Load data
        data = datacore.DataCore.load_from_csv(
            args.data_file,
            time_column=args.time_column,
            phenotype_columns=args.phenotype_columns
        )

        logger.info("Data loaded successfully")
        logger.info(f"Data shape: {data.data.shape}")

        # Fit model
        logger.info(f"Fitting {args.model_type} model")
        model = jumprope.JumpRope.fit(
            data,
            model_type=args.model_type
        )

        # Analyze cross-sections
        logger.info("Analyzing cross-sections")
        analyzer = laserplane.LaserPlaneAnalyzer(model)

        # Generate output
        output_dir = args.output or Path.cwd() / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        data.save_processed_data(output_dir / "processed_data.csv")
        model.save(output_dir / "model.pkl")

        logger.info(f"Analysis complete. Results saved to {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def fit_command(args: argparse.Namespace) -> int:
    """Handle the fit command."""
    try:
        logger.info(f"Fitting model to {args.data_file}")

        # Load data
        data = datacore.DataCore.load_from_csv(args.data_file)

        # Fit model
        model = jumprope.JumpRope.fit(data, model_type=args.model_type)

        # Save model
        output_file = args.output_model or Path(args.data_file.stem + "_model.pkl")
        model.save(output_file)

        logger.info(f"Model fitted and saved to {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return 1


def visualize_command(args: argparse.Namespace) -> int:
    """Handle the visualize command."""
    try:
        logger.info(f"Creating {args.plot_type} visualization")

        # Load model
        model = jumprope.JumpRope.load(args.model_file)

        # Create visualization
        output_dir = args.output or Path.cwd() / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.plot_type == 'trajectories':
            trajectory_visualizer.TrajectoryVisualizer.plot_trajectories(
                model,
                output_dir=output_dir,
                interactive=args.interactive
            )
        elif args.plot_type == 'cross-sections':
            trajectory_visualizer.TrajectoryVisualizer.plot_cross_sections(
                model,
                output_dir=output_dir,
                interactive=args.interactive
            )
        elif args.plot_type == 'landscapes':
            trajectory_visualizer.TrajectoryVisualizer.plot_landscapes(
                model,
                output_dir=output_dir,
                interactive=args.interactive
            )
        elif args.plot_type == 'animation':
            trajectory_visualizer.TrajectoryVisualizer.create_animation(
                model,
                output_dir=output_dir
            )

        logger.info(f"Visualization complete. Plots saved to {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1


def sample_command(args: argparse.Namespace) -> int:
    """Handle the sample command."""
    try:
        logger.info(f"Sampling from {args.population_file}")

        # Load population data
        population = datacore.DataCore.load_from_csv(args.population_file)

        # Sample
        sampler = evolution_sampler.EvolutionSampler(population)
        samples = sampler.sample(
            n_samples=args.n_samples,
            method=args.method
        )

        # Save samples
        output_dir = args.output or Path.cwd() / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        samples.save(output_dir / "samples.csv")
        logger.info(f"Sampling complete. Samples saved to {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging
    setup_logging(parsed_args.verbose)

    # Handle commands
    if parsed_args.command == 'analyze':
        return analyze_command(parsed_args)
    elif parsed_args.command == 'fit':
        return fit_command(parsed_args)
    elif parsed_args.command == 'visualize':
        return visualize_command(parsed_args)
    elif parsed_args.command == 'sample':
        return sample_command(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

