"""
Test suite for JumpRope module.

This module tests the jump-diffusion modeling functionality of the JumpRope module
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope


class TestModelParameters:
    """Test ModelParameters class."""

    def test_model_parameters_default_values(self):
        """Test ModelParameters with default values."""
        params = jumprope.ModelParameters()

        assert params.drift == 0.0
        assert params.diffusion == 1.0
        assert params.jump_intensity == 0.0
        assert params.jump_mean == 0.0
        assert params.jump_std == 1.0
        assert params.equilibrium == 0.0
        assert params.reversion_speed == 1.0

    def test_model_parameters_custom_values(self):
        """Test ModelParameters with custom values."""
        params = jumprope.ModelParameters(
            drift=0.5,
            diffusion=2.0,
            jump_intensity=0.1,
            jump_mean=1.0,
            jump_std=0.5,
            equilibrium=10.0,
            reversion_speed=0.8
        )

        assert params.drift == 0.5
        assert params.diffusion == 2.0
        assert params.jump_intensity == 0.1
        assert params.jump_mean == 1.0
        assert params.jump_std == 0.5
        assert params.equilibrium == 10.0
        assert params.reversion_speed == 0.8


class TestOrnsteinUhlenbeckJump:
    """Test OrnsteinUhlenbeckJump class."""

    def test_ornstein_uhlenbeck_jump_initialization(self):
        """Test OrnsteinUhlenbeckJump initialization."""
        params = jumprope.ModelParameters(
            equilibrium=10.0,
            reversion_speed=0.5,
            diffusion=1.0,
            jump_intensity=0.1,
            jump_mean=2.0,
            jump_std=0.5
        )

        process = jumprope.OrnsteinUhlenbeckJump(params)

        assert process.process_name == "Ornstein-Uhlenbeck with Jumps"
        assert process.parameters.equilibrium == 10.0
        assert process.parameters.reversion_speed == 0.5

    def test_simulate_trajectory(self):
        """Test trajectory simulation."""
        params = jumprope.ModelParameters(
            equilibrium=10.0,
            reversion_speed=0.5,
            diffusion=1.0,
            jump_intensity=0.1,
            jump_mean=0.0,
            jump_std=1.0
        )

        process = jumprope.OrnsteinUhlenbeckJump(params)

        time_points = np.linspace(0, 10, 101)
        trajectories = process.simulate(x0=5.0, t=time_points, n_paths=10)

        assert trajectories.shape == (10, 101)
        assert trajectories[0, 0] == 5.0  # Initial condition

        # Check that trajectories are reasonable (not all NaN or infinite)
        assert np.isfinite(trajectories).all()
        assert not np.allclose(trajectories, 5.0)  # Should have some variation

    def test_log_likelihood_computation(self):
        """Test log-likelihood computation."""
        params = jumprope.ModelParameters(
            equilibrium=10.0,
            reversion_speed=0.5,
            diffusion=1.0,
            jump_intensity=0.1,
            jump_mean=0.0,
            jump_std=1.0
        )

        process = jumprope.OrnsteinUhlenbeckJump(params)

        # Generate test data
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 50)
        dt = 0.1

        log_likelihood = process.log_likelihood(data, dt)

        assert isinstance(log_likelihood, float)
        assert np.isfinite(log_likelihood)

    def test_estimate_parameters(self):
        """Test parameter estimation."""
        params = jumprope.ModelParameters(
            equilibrium=10.0,
            reversion_speed=0.5,
            diffusion=1.0,
            jump_intensity=0.1,
            jump_mean=0.0,
            jump_std=1.0
        )

        process = jumprope.OrnsteinUhlenbeckJump(params)

        # Generate synthetic data
        np.random.seed(42)
        time_points = np.linspace(0, 10, 101)
        true_trajectories = process.simulate(x0=5.0, t=time_points, n_paths=1)
        synthetic_data = true_trajectories[0, :]
        dt = 0.1

        estimated_params = process.estimate_parameters(synthetic_data, dt)

        assert estimated_params is not None
        assert isinstance(estimated_params, jumprope.ModelParameters)
        assert np.isfinite(estimated_params.equilibrium)
        assert estimated_params.reversion_speed > 0
        assert estimated_params.diffusion > 0


class TestGeometricJumpDiffusion:
    """Test GeometricJumpDiffusion class."""

    def test_geometric_jump_diffusion_initialization(self):
        """Test GeometricJumpDiffusion initialization."""
        params = jumprope.ModelParameters(
            drift=0.1,
            diffusion=0.2,
            jump_intensity=0.05,
            jump_mean=0.0,
            jump_std=0.5
        )

        process = jumprope.GeometricJumpDiffusion(params)

        assert process.process_name == "Geometric Jump-Diffusion"
        assert process.parameters.drift == 0.1
        assert process.parameters.diffusion == 0.2

    def test_simulate_geometric_trajectory(self):
        """Test geometric trajectory simulation."""
        params = jumprope.ModelParameters(
            drift=0.05,
            diffusion=0.2,
            jump_intensity=0.1,
            jump_mean=0.0,
            jump_std=0.3
        )

        process = jumprope.GeometricJumpDiffusion(params)

        time_points = np.linspace(0, 5, 51)
        trajectories = process.simulate(x0=100.0, t=time_points, n_paths=5)

        assert trajectories.shape == (5, 51)
        assert trajectories[0, 0] == 100.0  # Initial condition
        assert np.all(trajectories > 0)  # Geometric process should stay positive

    def test_geometric_log_likelihood(self):
        """Test geometric log-likelihood computation."""
        params = jumprope.ModelParameters(
            drift=0.05,
            diffusion=0.2,
            jump_intensity=0.1,
            jump_mean=0.0,
            jump_std=0.3
        )

        process = jumprope.GeometricJumpDiffusion(params)

        # Generate positive test data
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 50) * 100  # Positive values
        dt = 0.1

        log_likelihood = process.log_likelihood(data, dt)

        assert isinstance(log_likelihood, float)
        assert np.isfinite(log_likelihood)


class TestJumpRope:
    """Test JumpRope class."""

    def create_test_data_core(self):
        """Create test DataCore for JumpRope tests."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 9, 11, 13, 15, 17]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        return datacore.DataCore([ts_data])

    def test_jump_rope_initialization(self):
        """Test JumpRope initialization."""
        params = jumprope.ModelParameters()
        process = jumprope.OrnsteinUhlenbeckJump(params)
        time_points = np.array([1, 2, 3, 4, 5])

        model = jumprope.JumpRope(process, time_points)

        assert model.stochastic_process == process
        assert len(model.time_points) == 5
        assert model.fitted_parameters is None

    def test_fit_model_jump_diffusion(self):
        """Test fitting jump-diffusion model."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        assert model.fitted_parameters is not None
        assert isinstance(model.fitted_parameters, jumprope.ModelParameters)
        assert model.fitted_parameters.equilibrium > 0
        assert model.fitted_parameters.reversion_speed > 0

    def test_fit_model_ornstein_uhlenbeck(self):
        """Test fitting Ornstein-Uhlenbeck model."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='ornstein-uhlenbeck',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        assert model.fitted_parameters is not None
        assert isinstance(model.fitted_parameters, jumprope.ModelParameters)

    def test_fit_model_geometric_jump_diffusion(self):
        """Test fitting geometric jump-diffusion model."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='geometric-jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        assert model.fitted_parameters is not None
        assert isinstance(model.fitted_parameters, jumprope.ModelParameters)

    def test_generate_trajectories(self):
        """Test trajectory generation."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        trajectories = model.generate_trajectories(n_samples=10, x0=10.0)

        assert trajectories.shape == (10, 5)
        assert trajectories[0, 0] == 10.0  # Initial condition

    def test_compute_cross_sections(self):
        """Test cross-section computation."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        model.generate_trajectories(n_samples=10, x0=10.0)

        cross_section = model.compute_cross_sections(2)  # Time point index 2

        assert len(cross_section) == 10  # Should have 10 samples
        assert cross_section.ndim == 1

    def test_estimate_jump_times(self):
        """Test jump time estimation."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        model.generate_trajectories(n_samples=20, x0=10.0)

        jump_times = model.estimate_jump_times()

        assert isinstance(jump_times, list)
        assert all(isinstance(t, (int, float)) for t in jump_times)

    def test_aggregate_parameters(self):
        """Test parameter aggregation."""
        params1 = jumprope.ModelParameters(
            drift=0.1, diffusion=1.0, jump_intensity=0.05,
            jump_mean=0.0, jump_std=0.5, equilibrium=10.0, reversion_speed=0.5
        )

        params2 = jumprope.ModelParameters(
            drift=0.2, diffusion=1.2, jump_intensity=0.08,
            jump_mean=0.1, jump_std=0.6, equilibrium=12.0, reversion_speed=0.6
        )

        aggregated = jumprope.JumpRope._aggregate_parameters([params1, params2])

        assert abs(aggregated.drift - 0.15) < 1e-10  # Mean of 0.1 and 0.2
        assert abs(aggregated.diffusion - 1.1) < 1e-10  # Mean of 1.0 and 1.2
        assert abs(aggregated.jump_intensity - 0.065) < 1e-10  # Mean of 0.05 and 0.08
        assert abs(aggregated.equilibrium - 11.0) < 1e-10  # Mean of 10.0 and 12.0

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = Path(f.name)

        try:
            model.save(temp_file)
            loaded_model = jumprope.JumpRope.load(temp_file)

            assert loaded_model.fitted_parameters is not None
            assert loaded_model.fitted_parameters.equilibrium == model.fitted_parameters.equilibrium
            assert loaded_model.fitted_parameters.reversion_speed == model.fitted_parameters.reversion_speed

        finally:
            temp_file.unlink()

    def test_fit_with_insufficient_data(self):
        """Test model fitting with insufficient data."""
        # Create minimal data
        data = pd.DataFrame({
            'time': [1, 2],
            'phenotype1': [10, 12]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        # Should handle gracefully with default parameters
        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2])
        )

        assert model.fitted_parameters is not None

    def test_trajectory_generation_with_different_models(self):
        """Test trajectory generation with different model types."""
        data_core = self.create_test_data_core()

        # Test with different model types
        for model_type in ['jump-diffusion', 'ornstein-uhlenbeck', 'geometric-jump-diffusion']:
            model = jumprope.JumpRope.fit(
                data_core,
                model_type=model_type,
                time_points=np.array([1, 2, 3, 4, 5])
            )

            trajectories = model.generate_trajectories(n_samples=5, x0=10.0)

            assert trajectories.shape == (5, 5)
            assert trajectories[0, 0] == 10.0
            assert np.isfinite(trajectories).all()

    def test_parameter_estimation_bounds(self):
        """Test that parameter estimation respects bounds."""
        # Create data with known characteristics
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 100)

        params = jumprope.ModelParameters()
        process = jumprope.OrnsteinUhlenbeckJump(params)

        estimated_params = process.estimate_parameters(data, dt=0.1)

        assert estimated_params.equilibrium >= np.min(data)
        assert estimated_params.equilibrium <= np.max(data)
        assert estimated_params.reversion_speed > 0
        assert estimated_params.diffusion > 0
        assert estimated_params.jump_intensity >= 0

    def test_cross_section_at_different_times(self):
        """Test cross-section computation at different time indices."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        model.generate_trajectories(n_samples=20, x0=10.0)

        # Test cross-sections at different time points
        for time_idx in [0, 2, 4]:
            cross_section = model.compute_cross_sections(time_idx)
            assert len(cross_section) == 20
            assert cross_section.ndim == 1
            assert np.isfinite(cross_section).all()
