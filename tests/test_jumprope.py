"""
Test suite for JumpRope module.

This module tests the jump-diffusion modeling functionality of the JumpRope module
using real data and methods.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope
from conftest import make_growth_frame


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

        # Generate test data (seeded for determinism)
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 50)
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

        process = jumprope.OrnsteinUhlenbeckJump(params, rng=np.random.default_rng(42))

        # Generate synthetic data
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
        rng = np.random.default_rng(42)
        data = rng.lognormal(0, 0.5, 50) * 100  # Positive values
        dt = 0.1

        log_likelihood = process.log_likelihood(data, dt)

        assert isinstance(log_likelihood, float)
        assert np.isfinite(log_likelihood)


class TestJumpRope:
    """Test JumpRope class."""

    def create_test_data_core(self):
        """Create test DataCore for JumpRope tests."""
        frame = make_growth_frame(n_points=5, phenotype_cols=("phenotype1",), seed=42)
        ts_data = datacore.TimeSeriesData(
            data=frame,
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
            time_points=np.array([0, 1, 2, 3, 4]),
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
            time_points=np.array([0, 1, 2, 3, 4]),
        )

        assert model.fitted_parameters is not None
        assert isinstance(model.fitted_parameters, jumprope.ModelParameters)

    def test_fit_model_geometric_jump_diffusion(self):
        """Test fitting geometric jump-diffusion model."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='geometric-jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]),
        )

        assert model.fitted_parameters is not None
        assert isinstance(model.fitted_parameters, jumprope.ModelParameters)

    def test_generate_trajectories(self):
        """Test trajectory generation."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
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
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
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
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
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
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
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
        frame = make_growth_frame(n_points=2, phenotype_cols=("phenotype1",), seed=42)

        ts_data = datacore.TimeSeriesData(
            data=frame,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        # Should handle gracefully with default parameters
        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([0, 1])
        )

        assert model.fitted_parameters is not None

    @pytest.mark.parametrize("model_type", [
        'jump-diffusion', 'ornstein-uhlenbeck',
        'geometric-jump-diffusion', 'compound-poisson',
    ])
    def test_trajectory_generation_with_different_models(self, model_type):
        """Trajectory generation works for every supported model type."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type=model_type,
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
        )

        trajectories = model.generate_trajectories(n_samples=5, x0=10.0)

        assert trajectories.shape == (5, 5)
        assert trajectories[0, 0] == 10.0
        assert np.isfinite(trajectories).all()

    def test_parameter_estimation_bounds(self):
        """Test that parameter estimation respects bounds."""
        # Create data with known characteristics
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

        params = jumprope.ModelParameters()
        process = jumprope.OrnsteinUhlenbeckJump(params)

        estimated_params = process.estimate_parameters(data, dt=0.1)

        assert estimated_params.equilibrium >= np.min(data)
        assert estimated_params.equilibrium <= np.max(data)
        assert estimated_params.reversion_speed > 0
        assert estimated_params.diffusion > 0
        assert estimated_params.jump_intensity >= 0

    @pytest.mark.parametrize("time_idx", [0, 2, 4])
    def test_cross_section_at_different_times(self, time_idx):
        """Cross-sections are one value per sample at any time index."""
        data_core = self.create_test_data_core()

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
        )

        model.generate_trajectories(n_samples=20, x0=10.0)

        cross_section = model.compute_cross_sections(time_idx)
        assert len(cross_section) == 20
        assert cross_section.ndim == 1
        assert np.isfinite(cross_section).all()


class TestLikelihoodDiscrimination:
    """Log-likelihoods must prefer parameters the data were simulated from."""

    def test_gjd_log_likelihood_prefers_true_parameters(self):
        """GJD one-jump component must include drift/diffusion (Gaussian
        mixture in log-return space), so the true parameters outscore
        badly shifted ones on data simulated at the truth."""
        true_params = jumprope.ModelParameters(
            drift=0.05, diffusion=0.2, jump_intensity=0.3,
            jump_mean=0.02, jump_std=0.1
        )
        process = jumprope.GeometricJumpDiffusion(true_params, rng=np.random.default_rng(42))
        t = np.linspace(0, 10, 201)
        data = process.simulate(x0=100.0, t=t, n_paths=1)[0]
        dt = 0.05

        ll_true = process.log_likelihood(data, dt)
        shifted = jumprope.GeometricJumpDiffusion(jumprope.ModelParameters(
            drift=0.5, diffusion=1.0, jump_intensity=0.3,
            jump_mean=0.02, jump_std=0.1
        ))
        ll_shifted = shifted.log_likelihood(data, dt)

        assert np.isfinite(ll_true)
        assert ll_true > ll_shifted

    def test_ou_log_likelihood_prefers_true_parameters(self):
        """Full Poisson-Gaussian mixture: true parameters must outscore
        shifted ones."""
        true_params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=0.8, diffusion=0.5,
            jump_intensity=0.5, jump_mean=1.0, jump_std=0.5
        )
        process = jumprope.OrnsteinUhlenbeckJump(true_params, rng=np.random.default_rng(7))
        t = np.linspace(0, 50, 501)
        data = process.simulate(x0=10.0, t=t, n_paths=1)[0]
        dt = 0.1

        ll_true = process.log_likelihood(data, dt)
        shifted = jumprope.OrnsteinUhlenbeckJump(jumprope.ModelParameters(
            equilibrium=0.0, reversion_speed=5.0, diffusion=5.0,
            jump_intensity=0.0, jump_mean=0.0, jump_std=1.0
        ))
        ll_shifted = shifted.log_likelihood(data, dt)

        assert np.isfinite(ll_true)
        assert ll_true > ll_shifted

    def test_compound_poisson_log_likelihood_prefers_true_parameters(self):
        """Log-space Poisson-Gaussian mixture must peak at the truth."""
        true_params = jumprope.ModelParameters(
            jump_intensity=0.5, jump_mean=1.0, jump_std=0.5
        )
        process = jumprope.CompoundPoisson(true_params, rng=np.random.default_rng(13))
        t = np.arange(0, 500, 1.0)
        data = process.simulate(x0=0.0, t=t, n_paths=1)[0]
        dt = 1.0

        ll_true = process.log_likelihood(data, dt)
        shifted = jumprope.CompoundPoisson(jumprope.ModelParameters(
            jump_intensity=0.1, jump_mean=0.0, jump_std=2.0
        ))
        ll_shifted = shifted.log_likelihood(data, dt)

        assert np.isfinite(ll_true)
        assert ll_true > ll_shifted

    def test_fit_end_to_end_compound_poisson(self):
        """The fit loop runs end-to-end for model_type='compound-poisson'."""
        frame = make_growth_frame(n_points=5, phenotype_cols=("phenotype1",), seed=21)
        ts_data = datacore.TimeSeriesData(
            data=frame, time_column='time', phenotype_columns=['phenotype1']
        )
        data_core = datacore.DataCore([ts_data])

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='compound-poisson',
            time_points=np.array([0, 1, 2, 3, 4]),
            seed=42,
        )

        assert model.fitted_parameters is not None
        assert np.isfinite(model.fitted_parameters.jump_intensity)
        trajectories = model.generate_trajectories(n_samples=5, x0=10.0, seed=42)
        assert trajectories.shape == (5, 5)
        assert np.isfinite(trajectories).all()


class TestParameterRecovery:
    """Estimators must recover known parameters from simulated data (seeded,
    loose tolerances)."""

    def test_ou_recovers_equilibrium_and_reversion_speed(self):
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=0.8, diffusion=0.5,
            jump_intensity=0.0, jump_mean=0.0, jump_std=0.5
        )
        process = jumprope.OrnsteinUhlenbeckJump(params, rng=np.random.default_rng(1))
        t = np.linspace(0, 100, 1001)
        data = process.simulate(x0=10.0, t=t, n_paths=1)[0]

        estimated = process.estimate_parameters(data, dt=0.1)

        assert abs(estimated.equilibrium - 10.0) < 0.5
        assert abs(estimated.reversion_speed - 0.8) < 0.5
        assert abs(estimated.diffusion - 0.5) < 0.25

    def test_compound_poisson_recovers_low_jump_intensity(self):
        params = jumprope.ModelParameters(
            jump_intensity=0.1, jump_mean=1.0, jump_std=0.5
        )
        process = jumprope.CompoundPoisson(params, rng=np.random.default_rng(1))
        t = np.arange(0, 3000, 1.0)
        data = process.simulate(x0=0.0, t=t, n_paths=1)[0]

        estimated = process.estimate_parameters(data, dt=1.0)

        # P(>=1 jump per step) = 1 - e^-0.1 ~ 0.095, so the frequency
        # estimator sits just below the true intensity.
        assert abs(estimated.jump_intensity - 0.1) < 0.03
        assert abs(estimated.jump_mean - 1.0) < 0.25
        assert abs(estimated.jump_std - 0.5) < 0.25


class TestReproducibility:
    """Seeding contract: same seed -> bitwise identical trajectories."""

    def create_test_data_core(self):
        frame = make_growth_frame(n_points=5, phenotype_cols=("phenotype1",), seed=3)
        ts_data = datacore.TimeSeriesData(
            data=frame, time_column='time', phenotype_columns=['phenotype1']
        )
        return datacore.DataCore([ts_data])

    def test_same_seed_trajectories_identical(self):
        data_core = self.create_test_data_core()
        model = jumprope.JumpRope.fit(
            data_core, model_type='jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]), seed=42
        )

        first = model.generate_trajectories(n_samples=5, x0=10.0, seed=42)
        second = model.generate_trajectories(n_samples=5, x0=10.0, seed=42)

        assert np.array_equal(first, second)

    def test_different_seed_trajectories_diverge(self):
        data_core = self.create_test_data_core()
        model = jumprope.JumpRope.fit(
            data_core, model_type='jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]), seed=42
        )

        first = model.generate_trajectories(n_samples=5, x0=10.0, seed=42)
        other = model.generate_trajectories(n_samples=5, x0=10.0, seed=43)

        assert not np.array_equal(first, other)

    def test_fit_rng_flows_to_trajectories(self):
        """An explicit rng= passed to fit() drives trajectory generation."""
        data_core = self.create_test_data_core()
        model_a = jumprope.JumpRope.fit(
            data_core, model_type='jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]),
            rng=np.random.default_rng(42)
        )
        model_b = jumprope.JumpRope.fit(
            data_core, model_type='jump-diffusion',
            time_points=np.array([0, 1, 2, 3, 4]),
            rng=np.random.default_rng(42)
        )

        traj_a = model_a.generate_trajectories(n_samples=5, x0=10.0)
        traj_b = model_b.generate_trajectories(n_samples=5, x0=10.0)

        assert np.array_equal(traj_a, traj_b)


class TestJumpTimeDetection:
    """estimate_jump_times must use an absolute robust threshold, not a
    per-path percentile that flags ~5% of every path."""

    def _model_with(self, jump_intensity, jump_mean, seed):
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=0.8, diffusion=0.5,
            jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=1.0
        )
        process = jumprope.OrnsteinUhlenbeckJump(params, rng=np.random.default_rng(seed))
        t = np.linspace(0, 50, 501)
        model = jumprope.JumpRope(process, t)
        model.generate_trajectories(n_samples=50, x0=10.0)
        return model


    def test_pure_diffusion_yields_fewer_flags_than_jumped_paths(self):
        clean = self._model_with(jump_intensity=0.0, jump_mean=0.0, seed=11)
        jumped = self._model_with(jump_intensity=0.5, jump_mean=6.0, seed=11)

        clean_times = clean.estimate_jump_times()
        jumped_times = jumped.estimate_jump_times()

        # 50 paths x 500 steps: pure diffusion flags a tiny fraction, while
        # large injected jumps are found throughout.
        assert len(clean_times) < 25
        assert len(jumped_times) > 100
        assert len(jumped_times) > len(clean_times)


class TestDegenerateInputContracts:
    """Degenerate-input behavior of each stochastic process."""

    @pytest.mark.parametrize("model_type", [
        'ornstein-uhlenbeck', 'geometric-jump-diffusion', 'compound-poisson',
    ])
    def test_log_likelihood_of_single_observation_is_zero(self, model_type):
        """Fewer than two observations carry no increment information: the
        log-likelihood is exactly 0.0, not nan or an exception."""
        process = {
            'ornstein-uhlenbeck': jumprope.OrnsteinUhlenbeckJump,
            'geometric-jump-diffusion': jumprope.GeometricJumpDiffusion,
            'compound-poisson': jumprope.CompoundPoisson,
        }[model_type](jumprope.ModelParameters())
        assert process.log_likelihood(np.array([5.0]), dt=0.1) == 0.0

    def test_ou_estimate_unconverged_returns_initial_parameters(self):
        """NaN observations make the optimizer abort without success: the
        pre-fit parameters are returned unchanged under a warning."""
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=0.5, diffusion=1.0,
            jump_intensity=0.1, jump_mean=0.0, jump_std=1.0
        )
        process = jumprope.OrnsteinUhlenbeckJump(params)
        data = np.array([10.0, np.nan, 12.0, 13.0] * 5)

        with pytest.warns(UserWarning, match="estimation failed"):
            estimated = process.estimate_parameters(data, dt=0.1)

        assert estimated.equilibrium == params.equilibrium
        assert estimated.reversion_speed == params.reversion_speed

    def test_ou_estimate_on_constant_series_returns_initial_parameters(self):
        """Zero-variance data makes the optimizer bounds invalid: the error
        is caught, warned about, and the pre-fit parameters survive."""
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=0.5, diffusion=1.0,
            jump_intensity=0.1, jump_mean=0.0, jump_std=1.0
        )
        process = jumprope.OrnsteinUhlenbeckJump(params)

        with pytest.warns(UserWarning, match="estimation error"):
            estimated = process.estimate_parameters(np.full(20, 10.0), dt=0.1)

        assert estimated.equilibrium == params.equilibrium
        assert estimated.reversion_speed == params.reversion_speed

    def test_gjd_log_likelihood_skips_nonpositive_observations(self):
        """A non-positive observation is skipped, so the likelihood over the
        remaining positive steps stays finite instead of becoming nan."""
        params = jumprope.ModelParameters(
            drift=0.05, diffusion=0.2, jump_intensity=0.1,
            jump_mean=0.0, jump_std=0.3
        )
        process = jumprope.GeometricJumpDiffusion(params)

        ll = process.log_likelihood(np.array([100.0, 0.0, 110.0, 120.0]), dt=0.1)

        assert np.isfinite(ll)

    def test_gjd_estimate_without_positive_observations_returns_own_parameters(self):
        """With no valid log-return (all observations non-positive) there is
        nothing to optimize: the process keeps its own parameters."""
        params = jumprope.ModelParameters(
            drift=0.1, diffusion=0.2, jump_intensity=0.1,
            jump_mean=0.0, jump_std=0.3
        )
        process = jumprope.GeometricJumpDiffusion(params)

        estimated = process.estimate_parameters(
            np.array([-1.0, -2.0, -3.0, -4.0]), dt=0.1
        )

        assert estimated.drift == params.drift
        assert estimated.diffusion == params.diffusion

    def test_gjd_estimate_unconverged_returns_initial_parameters(self):
        """NaN observations abort the optimizer without success: pre-fit
        parameters come back under a warning."""
        params = jumprope.ModelParameters(
            drift=0.1, diffusion=0.2, jump_intensity=0.1,
            jump_mean=0.0, jump_std=0.3
        )
        process = jumprope.GeometricJumpDiffusion(params)
        data = np.array([100.0, np.nan, 110.0, 120.0] * 5)

        with pytest.warns(UserWarning, match="estimation failed"):
            estimated = process.estimate_parameters(data, dt=0.1)

        assert estimated.drift == params.drift
        assert estimated.diffusion == params.diffusion

    def test_gjd_estimate_on_constant_series_returns_initial_parameters(self):
        """Constant positive data has zero log-return spread, making the
        optimizer bounds invalid: the error is warned about and the pre-fit
        parameters are returned."""
        params = jumprope.ModelParameters(
            drift=0.1, diffusion=0.2, jump_intensity=0.1,
            jump_mean=0.0, jump_std=0.3
        )
        process = jumprope.GeometricJumpDiffusion(params)

        with pytest.warns(UserWarning, match="estimation error"):
            estimated = process.estimate_parameters(np.full(20, 100.0), dt=0.1)

        assert estimated.drift == params.drift
        assert estimated.diffusion == params.diffusion

    def test_compound_poisson_zero_intensity_rejects_nonzero_increments(self):
        """With jump_intensity=0 the pmf mass sits entirely at k=0: a flat
        series scores 0.0 while any nonzero increment is impossible (-inf)."""
        process = jumprope.CompoundPoisson(
            jumprope.ModelParameters(jump_intensity=0.0, jump_mean=1.0, jump_std=0.5)
        )

        flat = np.full(6, 2.0)
        assert process.log_likelihood(flat, dt=1.0) == 0.0

        jumped = flat.copy()
        jumped[3] += 1.5
        assert process.log_likelihood(jumped, dt=1.0) == -np.inf

    def test_compound_poisson_estimate_on_constant_series(self):
        """A constant series shows no jumps: zero intensity and zero mean,
        with the unit fallback for the undefined jump-size spread."""
        process = jumprope.CompoundPoisson(
            jumprope.ModelParameters(jump_intensity=0.5, jump_mean=2.0, jump_std=1.0)
        )

        estimated = process.estimate_parameters(np.full(10, 5.0), dt=1.0)

        assert estimated.jump_intensity == 0.0
        assert estimated.jump_mean == 0.0
        assert estimated.jump_std == 1.0


class TestJumpRopeErrorPaths:
    """Error and fallback contracts of the JumpRope facade."""

    def test_fit_without_overlapping_timepoints_keeps_initial_parameters(self):
        """When no requested time point occurs in the data, no series can be
        fit: the initial parameters are kept as the fitted result."""
        frame = make_growth_frame(n_points=5, phenotype_cols=("phenotype1",), seed=42)
        ts_data = datacore.TimeSeriesData(
            data=frame, time_column='time', phenotype_columns=['phenotype1']
        )
        data_core = datacore.DataCore([ts_data])

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='compound-poisson',
            time_points=np.array([100.0, 200.0]),
            jump_intensity=0.3,
            jump_mean=1.5,
        )

        assert model.fitted_parameters is not None
        assert model.fitted_parameters.jump_intensity == 0.3
        assert model.fitted_parameters.jump_mean == 1.5

    def test_generate_trajectories_without_any_parameters_raises(self):
        """A process carrying neither fitted nor own parameters cannot
        simulate: ValueError points at fit()."""

        class BareProcess(jumprope.StochasticProcess):
            """Minimal process that deliberately sets no ``parameters``."""

            def __init__(self):
                self.process_name = "Bare"

            def simulate(self, x0, t, n_paths=1):
                return np.zeros((n_paths, len(t)))

            def log_likelihood(self, data, dt):
                return 0.0

            def estimate_parameters(self, data, dt):
                return jumprope.ModelParameters()

        model = jumprope.JumpRope(BareProcess(), np.array([0.0, 1.0]))

        with pytest.raises(ValueError, match="not fitted"):
            model.generate_trajectories(n_samples=2)

    @pytest.mark.parametrize("action", ["cross_sections", "jump_times"])
    def test_analysis_without_trajectories_raises(self, action):
        """Both downstream analyses require generated trajectories first."""
        process = jumprope.OrnsteinUhlenbeckJump(jumprope.ModelParameters())
        model = jumprope.JumpRope(process, np.array([0.0, 1.0, 2.0]))

        with pytest.raises(ValueError, match="generate_trajectories"):
            if action == "cross_sections":
                model.compute_cross_sections(0)
            else:
                model.estimate_jump_times()

    def test_jump_time_detection_falls_back_to_std_scale(self):
        """When MAD is zero (more than half the increments identical) the
        detector falls back to the standard deviation, so a large injected
        jump against a uniform ramp is still flagged at its time point."""
        process = jumprope.OrnsteinUhlenbeckJump(jumprope.ModelParameters())
        time_points = np.arange(31.0)
        model = jumprope.JumpRope(process, time_points)
        ramp = np.concatenate([np.zeros(1), np.cumsum([1.0] * 25 + [6.0] + [1.0] * 4)])
        model.trajectories = ramp.reshape(1, -1)

        jump_times = model.estimate_jump_times()

        assert jump_times == [25.0]

    def test_unfitted_model_simulates_from_process_parameters(self):
        """An unfitted model falls back to the stochastic process's own
        parameters instead of requiring fit() first."""
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=0.5, diffusion=1.0,
            jump_intensity=0.0, jump_mean=0.0, jump_std=1.0
        )
        process = jumprope.OrnsteinUhlenbeckJump(params, rng=np.random.default_rng(5))
        model = jumprope.JumpRope(
            process, np.array([0.0, 1.0, 2.0]), initial_conditions={'x0': 10.0}
        )

        trajectories = model.generate_trajectories(n_samples=3, seed=5)

        assert trajectories.shape == (3, 3)
        assert np.all(trajectories[:, 0] == 10.0)
        assert np.isfinite(trajectories).all()
        # The fallback adopted the process's parameters as fitted parameters.
        assert model.fitted_parameters == params

    def test_jump_time_detection_degenerate_constant_increment_returns_empty(self):
        """All-identical increments carry no jump evidence: the detector
        reports an empty list instead of flagging the ramp."""
        process = jumprope.OrnsteinUhlenbeckJump(jumprope.ModelParameters())
        model = jumprope.JumpRope(process, np.arange(31.0))
        model.trajectories = np.arange(31.0).reshape(1, -1)

        assert model.estimate_jump_times() == []
