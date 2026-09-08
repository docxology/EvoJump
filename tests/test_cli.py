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

from evojump import cli


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

    def create_test_data(self):
        """Create test data for CLI tests."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 9, 11, 13, 15, 17],
            'phenotype2': [20, 22, 24, 26, 28, 21, 23, 25, 27, 29, 19, 21, 23, 25, 27]
        })
        return data

    def test_validate_csv_input_file(self):
        """Test CSV input file validation."""
        # Test with valid CSV
        data = self.create_test_data()

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


class TestCLISubcommands:
    """Test CLI subcommand functionality."""

    def create_test_data(self):
        """Create test data for CLI tests."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19],
            'phenotype2': [20, 22, 24, 26, 28, 21, 23, 25, 27, 29]
        })
        return data

    def test_analyze_command_basic(self):
        """Test basic analyze command functionality."""
        data = self.create_test_data()

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
        data = self.create_test_data()

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
        data = self.create_test_data()

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
        data = self.create_test_data()

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


class TestCLIExitCodesAndOutput:
    """Test exit-code contract, global options, and plot-type coverage."""

    def create_test_data(self, time_col='time'):
        """Create synthetic test data for CLI tests."""
        data = pd.DataFrame({
            time_col: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19],
            'phenotype2': [20, 22, 24, 26, 28, 21, 23, 25, 27, 29]
        })
        return data

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
        data = self.create_test_data()

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
        data = self.create_test_data(time_col='age')

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

    @pytest.mark.parametrize('plot_type,expected_name', [
        ('trajectories', 'trajectories.png'),
        ('cross-sections', 'cross_sections.png'),
        ('landscapes', 'landscape.png'),
    ])
    def test_visualize_static_plot_types(self, plot_type, expected_name):
        """Test each static visualize plot type produces its artifact."""
        data = self.create_test_data()

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
        data = self.create_test_data()

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


class TestCLILogging:
    """Test CLI logging functionality."""

    def test_setup_logging_verbosity_levels(self):
        """Test that verbosity levels map to distinct log levels."""
        cli.setup_logging(0)
        assert logging.getLogger('evojump').level == logging.INFO
        cli.setup_logging(1)
        assert logging.getLogger('evojump').level == logging.DEBUG

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

