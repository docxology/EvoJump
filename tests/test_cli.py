"""
Test suite for CLI module.

This module tests the command-line interface functionality of the EvoJump package
using real data and methods.
"""

import pytest
import subprocess
import sys
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conftest import make_growth_frame

from evojump import cli


def _cli_frame(n_points: int = 10) -> pd.DataFrame:
    """CLI-shaped synthetic frame: a ``time`` column plus two phenotypes."""
    return make_growth_frame(
        n_points=n_points, phenotype_cols=("phenotype1", "phenotype2"))


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_create_parser_basic(self):
        """Test basic parser creation."""
        parser = cli.create_parser()

        assert parser is not None
        assert parser.description is not None
        assert "EvoJump" in parser.description

    def test_parser_has_required_subcommands(self):
        """Test that parser has all required subcommands."""
        parser = cli.create_parser()

        # Check that main subcommands exist
        subparsers_action = None
        for action in parser._actions:
            choices = getattr(action, 'choices', None)
            if isinstance(choices, dict) and 'analyze' in choices:
                subparsers_action = action
                break

        assert subparsers_action is not None
        assert 'analyze' in subparsers_action.choices
        assert 'fit' in subparsers_action.choices
        assert 'visualize' in subparsers_action.choices
        assert 'sample' in subparsers_action.choices

    def test_parser_help_formatting(self):
        """Test that parser help is properly formatted."""
        parser = cli.create_parser()

        help_text = parser.format_help()

        assert "Examples:" in help_text
        assert "evojump-cli analyze" in help_text
        assert "evojump-cli fit" in help_text
        assert "evojump-cli visualize" in help_text


class TestCLIDataValidation:
    """Test CLI data validation."""


    def test_validate_csv_input_file(self):
        """Test CSV input file validation."""
        # Test with valid CSV
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        try:
            # Should not raise exception for valid CSV
            result = cli._validate_input_file(temp_file, "csv")
            assert result is True
        finally:
            temp_file.unlink()

    def test_validate_csv_invalid_file(self):
        """Test CSV validation with invalid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not,csv,data")
            temp_file = Path(f.name)

        try:
            # Should raise exception for invalid CSV
            with pytest.raises(ValueError, match="Invalid CSV format"):
                cli._validate_input_file(temp_file, "csv")
        finally:
            temp_file.unlink()

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_file = Path("nonexistent_file.csv")

        with pytest.raises(FileNotFoundError):
            cli._validate_input_file(nonexistent_file, "csv")


    def test_validate_directory_path_raises(self, tmp_path):
        """Test that an existing directory fails validation as 'not a file'."""
        with pytest.raises(ValueError, match="not a file"):
            cli._validate_input_file(tmp_path, "csv")

    def test_validate_csv_with_unparseable_content(self, tmp_path):
        """Test that a .csv file with non-CSV bytes fails content validation."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_bytes(b"\xff\xfe\x00\x01 not a csv")
        with pytest.raises(ValueError, match="Invalid CSV format"):
            cli._validate_input_file(bad_csv, "csv")

class TestCLISubcommands:
    """Test CLI subcommand functionality."""


    def test_analyze_command_basic(self):
        """Test basic analyze command functionality."""
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                # Should run without errors and exit 0
                result = cli.main(['analyze', str(temp_file), '--output', str(output_dir)])
                assert result == 0
                # Check that output files were created
                assert (output_dir / 'analysis_results.json').exists()
                assert (output_dir / 'data_summary.json').exists()

            finally:
                temp_file.unlink()

    def test_fit_command_basic(self):
        """Test basic fit command functionality."""
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_file = output_dir / 'model.pkl'

            try:
                result = cli.main([
                    'fit', str(temp_file),
                    '--model-type', 'jump-diffusion',
                    '--output', str(output_file)
                ])
                assert result == 0

                # Check that model file was created
                assert output_file.exists()

            finally:
                temp_file.unlink()

    def test_visualize_command_basic(self):
        """Test basic visualize command functionality."""
        # First create a model file
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        try:
            # Fit a model first
            import evojump as ej

            data_core = ej.DataCore.load_from_csv(str(temp_file), time_column='time')
            model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')

            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as model_file:
                model.save(Path(model_file.name))
                model_path = Path(model_file.name)

            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)

                # Should run without errors
                result = cli.main([
                    'visualize', str(model_path),
                    '--output', str(output_dir)
                ])
                assert result == 0

                # Check that visualization files were created
                assert (output_dir / 'trajectories.png').exists()

        finally:
            temp_file.unlink()
            if 'model_path' in locals():
                model_path.unlink()

    def test_sample_command_basic(self):
        """Test basic sample command functionality."""
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_file = output_dir / 'samples.csv'

            try:
                # Should run without errors
                result = cli.main([
                    'sample', str(temp_file),
                    '--samples', '100',
                    '--output', str(output_file)
                ])
                assert result == 0

                # Check that samples file was created
                assert output_file.exists()

                # Check that samples file has correct structure
                samples_df = pd.read_csv(output_file)
                assert 'sample_id' in samples_df.columns
                assert 'phenotype1' in samples_df.columns
                assert 'phenotype2' in samples_df.columns

            finally:
                temp_file.unlink()


    def test_sample_output_directory_writes_samples_csv(self, tmp_path):
        """Test that a directory --output writes samples.csv inside it."""
        csv_path = tmp_path / "population.csv"
        _cli_frame().to_csv(csv_path, index=False)
        out_dir = tmp_path / "results"

        result = cli.main([
            'sample', str(csv_path),
            '--samples', '20',
            '--output', str(out_dir)
        ])

        assert result == 0
        samples_csv = out_dir / "samples.csv"
        assert samples_csv.exists()
        samples_df = pd.read_csv(samples_csv)
        assert 'sample_id' in samples_df.columns
        assert 'phenotype1' in samples_df.columns

    def test_sample_mcmc_writes_wide_format(self, tmp_path):
        """Test that the mcmc method writes one row per sample (wide format)."""
        csv_path = tmp_path / "population.csv"
        _cli_frame().to_csv(csv_path, index=False)
        output_file = tmp_path / "samples_mcmc.csv"

        result = cli.main([
            'sample', str(csv_path),
            '--n-samples', '15',
            '--method', 'mcmc',
            '--output', str(output_file)
        ])

        assert result == 0
        samples_df = pd.read_csv(output_file)
        assert len(samples_df) == 15
        assert list(samples_df['sample_id']) == [
            f"sample_{i:06d}" for i in range(15)]
        assert 'phenotype1' in samples_df.columns
        assert 'phenotype2' in samples_df.columns

class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_subcommand(self):
        """Test error handling for invalid subcommand."""
        with pytest.raises(SystemExit):
            cli.main(['invalid_command'])

    def test_missing_input_file(self):
        """Test error handling for missing input file."""
        # Runtime failure returns 1, per main()'s documented contract.
        assert cli.main(['analyze', 'nonexistent.csv']) == 1

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        try:
            with pytest.raises(SystemExit):
                cli.main([
                    'fit', str(temp_file),
                    '--model-type', 'invalid_model_type'
                ])
        finally:
            temp_file.unlink()

    def test_missing_output_directory(self):
        """Test error handling for an unwritable output directory."""
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        # Make the output parent read-only so mkdir fails portably, unlike
        # '/nonexistent/...' which root containers happily create.
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_parent = Path(temp_dir) / 'readonly'
            readonly_parent.mkdir()
            readonly_parent.chmod(0o500)
            try:
                if os.name != 'posix' or os.geteuid() == 0:
                    pytest.skip(
                        "unwritable-directory test requires POSIX permissions "
                        "and a non-root user")
                result = cli.main([
                    'analyze', str(temp_file),
                    '--output', str(readonly_parent / 'results')
                ])
                assert result == 1
            finally:
                readonly_parent.chmod(0o700)

        temp_file.unlink()


    @pytest.mark.parametrize('command', ['analyze', 'fit', 'sample'])
    def test_missing_time_column_returns_runtime_failure(self, command, tmp_path):
        """Test that a CSV without the expected time column exits 1, not a traceback."""
        csv_path = tmp_path / "no_time.csv"
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

        assert cli.main(
            [command, str(csv_path), '--output', str(tmp_path / "out")]) == 1

    def test_visualize_with_corrupt_model_file_returns_runtime_failure(self, tmp_path):
        """Test that an unpicklable model file exits 1."""
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_bytes(b"definitely not a pickle")

        assert cli.main(
            ['visualize', str(corrupt), '--output', str(tmp_path / "out")]) == 1

class TestCLIExitCodesAndOutput:
    """Test exit-code contract, global options, and plot-type coverage."""


    def _save_fitted_model(self, csv_path):
        """Fit a model on synthetic data and save it, returning the path."""
        import evojump as ej

        data_core = ej.DataCore.load_from_csv(str(csv_path), time_column='time')
        model = ej.JumpRope.fit(data_core, model_type='jump-diffusion', seed=0)
        model_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        model.save(Path(model_file.name))
        model_file.close()
        return Path(model_file.name)

    def test_no_subcommand_returns_usage_error(self):
        """Test that invoking with no subcommand is a usage error (exit 2)."""
        assert cli.main([]) == 2

    def test_version_exits_successfully(self):
        """Test that --version exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(['--version'])
        assert exc_info.value.code == 0

    def test_global_output_precedes_subcommand(self):
        """Test that a global --output given before the subcommand is used."""
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                global_out = Path(temp_dir) / 'global_out'
                result = cli.main([
                    '--output', str(global_out),
                    'analyze', str(temp_file)
                ])
                assert result == 0
                assert (global_out / 'analysis_results.json').exists()
        finally:
            temp_file.unlink()

    def test_fit_with_renamed_time_column(self):
        """Test that fit accepts a non-default time column via --time-column."""
        data = _cli_frame().rename(columns={"time": "age"})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / 'model.pkl'
                result = cli.main([
                    'fit', str(temp_file),
                    '--time-column', 'age',
                    '--output', str(output_file)
                ])
                assert result == 0
                assert output_file.exists()
        finally:
            temp_file.unlink()

    def test_fit_output_directory_gets_default_model_filename(self, tmp_path):
        """Test that an existing directory --output saves <stem>_model.pkl inside."""
        csv_path = tmp_path / "data.csv"
        _cli_frame().to_csv(csv_path, index=False)
        output_dir = tmp_path / "models"
        output_dir.mkdir()

        result = cli.main(['fit', str(csv_path), '--output', str(output_dir)])

        assert result == 0
        assert (output_dir / 'data_model.pkl').exists()

    @pytest.mark.parametrize('plot_type,expected_name', [
        ('trajectories', 'trajectories.png'),
        ('cross-sections', 'cross_sections.png'),
        ('landscapes', 'landscape.png'),
    ])
    def test_visualize_static_plot_types(self, plot_type, expected_name):
        """Test each static visualize plot type produces its artifact."""
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        model_path = None
        try:
            model_path = self._save_fitted_model(temp_file)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                result = cli.main([
                    'visualize', str(model_path),
                    '--plot-type', plot_type,
                    '--output', str(output_dir)
                ])
                assert result == 0
                assert (output_dir / expected_name).exists()
        finally:
            temp_file.unlink()
            if model_path is not None:
                model_path.unlink()

    @pytest.mark.parametrize('plot_type', ['trajectories', 'cross-sections', 'landscapes'])
    def test_visualize_interactive_writes_html(self, plot_type):
        """Test that --interactive persists a Plotly HTML artifact."""
        data = _cli_frame()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        model_path = None
        try:
            model_path = self._save_fitted_model(temp_file)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                result = cli.main([
                    'visualize', str(model_path),
                    '--plot-type', plot_type,
                    '--interactive',
                    '--output', str(output_dir)
                ])
                assert result == 0
                assert (output_dir / f'{plot_type}.html').exists()
        finally:
            temp_file.unlink()
            if model_path is not None:
                model_path.unlink()

    def test_visualize_animation_writes_gif(self, tmp_path):
        """Test that the animation plot type saves an animated GIF."""
        csv_path = tmp_path / "data.csv"
        _cli_frame().to_csv(csv_path, index=False)
        model_path = self._save_fitted_model(csv_path)
        output_dir = tmp_path / "plots"
        try:
            result = cli.main([
                'visualize', str(model_path),
                '--plot-type', 'animation',
                '--output', str(output_dir)
            ])
            assert result == 0
            assert (output_dir / 'animation.gif').exists()
        finally:
            model_path.unlink()



class TestCLILogging:
    """Test CLI logging functionality."""

    def test_setup_logging_verbosity_levels(self):
        """Test that verbosity levels map to distinct log levels."""
        cli.setup_logging(0)
        assert logging.getLogger('evojump').level == logging.INFO
        cli.setup_logging(1)
        assert logging.getLogger('evojump').level == logging.DEBUG

    def test_setup_logging_double_v_enables_library_debug(self):
        """Test that -vv surfaces DEBUG records from driven libraries."""
        try:
            cli.setup_logging(2)
            assert logging.getLogger('evojump').level == logging.DEBUG
            for name in ('matplotlib', 'pandas', 'plotly'):
                assert logging.getLogger(name).level == logging.DEBUG
        finally:
            for name in ('matplotlib', 'pandas', 'plotly'):
                logging.getLogger(name).setLevel(logging.WARNING)


    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        # Test that logger exists and is configured
        assert hasattr(cli, 'logger')
        assert cli.logger is not None

    def test_log_messages(self, caplog):
        """Test that appropriate log messages are generated."""
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                with caplog.at_level('INFO'):
                    cli.main(['analyze', str(temp_file), '--output', str(output_dir)])

                # Check that log messages were generated
                assert any('Starting analysis' in record.message for record in caplog.records)
                assert any('Analysis completed' in record.message for record in caplog.records)

            finally:
                temp_file.unlink()


class TestCLIVersionResolution:
    """Test get_version() fallbacks when package metadata is unavailable."""

    def test_metadata_failure_falls_back_to_package_version(self, monkeypatch):
        """Test that a failing metadata lookup falls back to evojump.__version__."""
        import evojump

        def broken_version(name):
            raise RuntimeError("metadata unavailable")

        monkeypatch.setattr(cli, '_package_version', broken_version)
        assert cli.get_version() == evojump.__version__

    def test_missing_dunder_version_reports_unknown(self, monkeypatch):
        """Test that get_version() reports 'unknown' when no version source exists."""
        import evojump

        monkeypatch.setattr(cli, '_package_version', None)
        monkeypatch.delattr(evojump, '__version__')
        assert cli.get_version() == "unknown"


class TestCLIModuleEntryPoint:
    """Test the ``python -m evojump.cli`` entry-point guard."""

    def test_module_execution_without_subcommand_exits_two(self, monkeypatch):
        """Test that direct module execution routes through main() (exit 2)."""
        import runpy

        monkeypatch.setattr(sys, 'argv', ['evojump-cli'])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module('evojump.cli', run_name='__main__')
        assert exc_info.value.code == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

