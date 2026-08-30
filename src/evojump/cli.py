"""
Command Line Interface for EvoJump Package

This module provides a command-line interface for the EvoJump package,
allowing users to perform evolutionary ontogenetic analysis from the command line.
"""

import argparse
import sys
import logging

import numpy as np
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


def _validate_input_file(file_path: Path, expected_format: str = "csv") -> bool:
    """Validate that an input file exists and matches the expected format.

    Raises FileNotFoundError for a missing file and ValueError for a file
    whose content does not parse as the expected format.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    if expected_format == "csv":
        suffix = file_path.suffix.lower()
        if suffix != ".csv":
            raise ValueError(f"Invalid CSV format: {suffix}")
        try:
            import pandas as pd
            pd.read_csv(file_path, nrows=5)
        except Exception as exc:
            raise ValueError(f"Invalid CSV format: {exc}") from exc
    return True


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
        choices=['jump-diffusion', 'ornstein-uhlenbeck', 'compound-poisson',
                      'geometric-jump-diffusion', 'fractional-brownian', 'cir', 'levy'],
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
    analyze_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory for results'
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
        choices=['jump-diffusion', 'ornstein-uhlenbeck', 'compound-poisson',
                      'geometric-jump-diffusion', 'fractional-brownian', 'cir', 'levy'],
        default='jump-diffusion',
        help='Type of stochastic process model'
    )
    fit_parser.add_argument(
        '--output-model',
        type=Path,
        help='Output file for fitted model'
    )
    fit_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file (alias for --output-model) or directory for the fitted model'
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
    visualize_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory for plots'
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
        '--n-samples', '--samples',
        dest='n_samples',
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
    sample_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for samples'
    )

    return parser


def setup_logging(verbosity: int) -> None:
    """Set up logging based on verbosity level."""
    if verbosity == 0:
        level = logging.INFO
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    # Configure the package logger, not the root logger, so host
    # applications (and test harnesses) keep control of root logging.
    logging.getLogger('evojump').setLevel(level)

    # Configure specific loggers
    logging.getLogger('evojump').setLevel(level)


def analyze_command(args: argparse.Namespace) -> int:
    """Handle the analyze command."""
    import json as _json
    try:
        logger.info("Starting analysis")
        logger.info(f"Loading data from {args.data_file}")

        _validate_input_file(args.data_file, "csv")

        # Load data
        data = datacore.DataCore.load_from_csv(
            args.data_file,
            time_column=args.time_column,
            phenotype_columns=args.phenotype_columns
        )

        logger.info("Data loaded successfully")
        first_ts = data.time_series_data[0]
        logger.info(f"Data shape: {first_ts.data.shape}")

        # Fit model
        logger.info(f"Fitting {args.model_type} model")
        model = jumprope.JumpRope.fit(
            data,
            model_type=args.model_type
        )

        # Generate output
        output_dir = args.output or Path.cwd() / "results"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(f"Analysis failed: cannot create output directory {output_dir}: {exc}")
            raise SystemExit(1) from exc

        # Save results
        data.save_processed_data(output_dir / "processed_data.csv")
        model.save(output_dir / "model.pkl")

        # Structured summaries
        quality = data.validate_data_quality()
        data_summary = {
            'n_datasets': len(data.time_series_data),
            'n_samples': int(first_ts.data.shape[0]),
            'n_columns': int(first_ts.data.shape[1]),
            'time_column': first_ts.time_column,
            'phenotype_columns': first_ts.phenotype_columns,
            'n_timepoints': int(first_ts.n_timepoints),
            'quality_metrics': quality,
        }
        (output_dir / 'data_summary.json').write_text(
            _json.dumps(data_summary, indent=2, default=str))

        analysis_results = {
            'model_type': args.model_type,
            'fitted_parameters': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in vars(model.fitted_parameters).items()
                if isinstance(v, (int, float, np.ndarray))
            } if model.fitted_parameters is not None else {},
            'n_trajectories_generated': 0,
        }
        (output_dir / 'analysis_results.json').write_text(
            _json.dumps(analysis_results, indent=2, default=str))

        logger.info("Analysis completed")
        logger.info(f"Analysis results saved to {output_dir}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Analysis failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def fit_command(args: argparse.Namespace) -> int:
    """Handle the fit command."""
    try:
        logger.info(f"Fitting model to {args.data_file}")

        _validate_input_file(args.data_file, "csv")

        # Load data
        data = datacore.DataCore.load_from_csv(args.data_file)

        # Fit model
        model = jumprope.JumpRope.fit(data, model_type=args.model_type, seed=0)

        # Save model (--output alias supported alongside --output-model)
        output_file = getattr(args, 'output_model', None) or getattr(args, 'output', None) \
            or Path(args.data_file.stem + "_model.pkl")
        if output_file.is_dir():
            output_file = output_file / (Path(args.data_file).stem + "_model.pkl")
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

        # Visualizing requires trajectories; generate them on demand
        # when the fitted model has none.
        if getattr(model, 'trajectories', None) is None:
            model.generate_trajectories(n_samples=20, seed=0)

        # Create visualization
        output_dir = args.output or Path.cwd() / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use('Agg')

        visualizer = trajectory_visualizer.TrajectoryVisualizer()

        if args.plot_type == 'trajectories':
            visualizer.plot_trajectories(
                model,
                output_dir=output_dir,
                interactive=args.interactive
            )
        elif args.plot_type == 'cross-sections':
            visualizer.plot_cross_sections(
                model,
                output_dir=output_dir,
                interactive=args.interactive
            )
        elif args.plot_type == 'landscapes':
            visualizer.plot_landscapes(
                model,
                output_dir=output_dir,
                interactive=args.interactive
            )
        elif args.plot_type == 'animation':
            visualizer.create_animation(
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

        # Save samples (--output may be a directory or a file path)
        out_path = args.output or Path.cwd() / "samples"
        if out_path.suffix == '.csv':
            out_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            csv_path = out_path / "samples.csv"

        import pandas as _pd
        sample_array = np.asarray(samples.samples)

        # Recover the phenotype column names used by the sampler so output
        # columns match the input data schema.
        numeric_cols = population.time_series_data[0].data.select_dtypes(
            include=[np.number]).columns.tolist()
        time_col = population.time_series_data[0].time_column
        pheno_names = [c for c in numeric_cols if c != time_col]

        if sample_array.ndim == 3:
            # Long format: one row per (sample, time) with real phenotype names
            n_time = sample_array.shape[1]
            n_pheno = sample_array.shape[2]
            names = pheno_names[:n_pheno] + [f'pheno{k}' for k in range(n_pheno - len(pheno_names))]
            rows = []
            for i, sid in enumerate(samples.sample_ids):
                for j in range(n_time):
                    row = {'sample_id': sid, 'time_index': j}
                    for k, nm in enumerate(names):
                        row[nm] = sample_array[i, j, k]
                    rows.append(row)
            _pd.DataFrame(rows).to_csv(csv_path, index=False)
        else:
            n_pheno = sample_array.shape[1]
            names = pheno_names[:n_pheno] + [f'pheno{k}' for k in range(n_pheno - len(pheno_names))]
            columns = {'sample_id': samples.sample_ids}
            for k, nm in enumerate(names):
                columns[nm] = sample_array[:, k]
            _pd.DataFrame(columns).to_csv(csv_path, index=False)
        logger.info(f"Sampling complete. Samples saved to {csv_path}")
        return 0

    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    global_output = parsed_args.output

    # Set up logging
    setup_logging(parsed_args.verbose)

    # Subcommand-level --output overrides the global default when given;
    # fall back to the global value when only the subcommand form was used.
    if getattr(parsed_args, 'output', None) is None:
        parsed_args.output = global_output

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

