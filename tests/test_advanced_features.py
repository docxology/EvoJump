"""
Test suite for advanced features: new stochastic models, visualizations, and analytics.

This module provides comprehensive testing for all newly added features including
Fractional Brownian Motion, CIR process, Levy process, advanced visualizations,
and statistical methods.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope, trajectory_visualizer, analytics_engine
from pathlib import Path


class TestFractionalBrownianMotion:
    """Test Fractional Brownian Motion implementation."""
    
    def test_fbm_initialization(self):
        """Test FBM initialization with different Hurst parameters."""
        params = jumprope.ModelParameters(drift=0.1, diffusion=1.0)
        
        # Test persistent motion (H > 0.5)
        fbm_persistent = jumprope.FractionalBrownianMotion(params, hurst=0.7)
        assert fbm_persistent.hurst == 0.7
        assert fbm_persistent.process_name == "Fractional Brownian Motion"
        
        # Test anti-persistent motion (H < 0.5)
        fbm_antipersistent = jumprope.FractionalBrownianMotion(params, hurst=0.3)
        assert fbm_antipersistent.hurst == 0.3
    
    def test_fbm_simulation(self):
        """Test FBM trajectory generation."""
        params = jumprope.ModelParameters(drift=0.0, diffusion=1.0)
        fbm = jumprope.FractionalBrownianMotion(params, hurst=0.7)
        
        t = np.linspace(0, 10, 100)
        paths = fbm.simulate(x0=10.0, t=t, n_paths=50)
        
        assert paths.shape == (50, 100)
        assert np.all(paths[:, 0] == 10.0)  # Initial condition
        assert np.all(np.isfinite(paths))  # No NaN or Inf
    
    def test_fbm_parameter_estimation(self):
        """Test FBM parameter estimation."""
        params = jumprope.ModelParameters(drift=0.5, diffusion=1.0)
        fbm = jumprope.FractionalBrownianMotion(params, hurst=0.7)
        
        # Generate synthetic data
        t = np.linspace(0, 10, 50)
        data = fbm.simulate(x0=10.0, t=t, n_paths=1)[0, :]
        
        # Estimate parameters
        dt = np.mean(np.diff(t))
        estimated_params = fbm.estimate_parameters(data, dt)
        
        assert estimated_params is not None
        assert np.isfinite(estimated_params.drift)
        assert np.isfinite(estimated_params.diffusion)
        assert estimated_params.diffusion > 0

    def _davies_harte_fbm(self, n, hurst, dt, seed, diffusion=1.0):
        """Generate a true fractional Brownian motion path (Davies-Harte).

        The production simulator draws independent increments, which carry no
        long-range dependence, so genuine fBM is synthesized here to test
        Hurst recovery against ground truth.
        """
        def r(k):
            return 0.5 * diffusion ** 2 * dt ** (2 * hurst) * (
                abs(k + 1) ** (2 * hurst) - 2 * abs(k) ** (2 * hurst)
                + abs(k - 1) ** (2 * hurst)
            )

        rng = np.random.default_rng(seed)
        m = 2 * n
        g = np.zeros(m)
        g[0] = r(0)
        for k in range(1, n):
            g[k] = r(k)
            g[m - k] = r(k)
        eigenvalues = np.maximum(np.real(np.fft.fft(g)), 0)
        sqrt_eig = np.sqrt(eigenvalues / (2 * m))
        noise = rng.normal(size=m) + 1j * rng.normal(size=m)
        fgn = np.real(np.fft.fft(sqrt_eig * noise))[:n] * np.sqrt(m)
        # Match the theoretical marginal scale: fGn increments have std
        # diffusion * dt**hurst. Rescaling preserves the correlation
        # structure (hence H) and removes FFT normalization ambiguity.
        fgn *= diffusion * dt ** hurst / np.std(fgn)
        return np.concatenate([[0.0], np.cumsum(fgn)])

    @pytest.mark.parametrize("true_hurst,seed", [(0.3, 1), (0.7, 2)])
    def test_fbm_recovers_hurst_exponent(self, true_hurst, seed):
        """Hurst estimation on genuine fBM data (Davies-Harte)."""
        dt = 0.1
        path = self._davies_harte_fbm(1024, true_hurst, dt, seed)
        fbm = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=1.0), hurst=0.5
        )
        estimated = fbm.estimate_parameters(path, dt)

        assert abs(fbm.hurst - true_hurst) < 0.15
        assert np.isfinite(estimated.diffusion)
        assert estimated.diffusion > 0

    @pytest.mark.parametrize("diffusion_true", [1.0, 2.0])
    def test_fbm_diffusion_std_scale_convention(self, diffusion_true):
        """diffusion is in std-deviation units: Var[increment]
        = diffusion^2 * dt^(2H), not diffusion * dt^(2H)."""
        t = np.linspace(0, 20, 801)
        dt = t[1] - t[0]
        fbm = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=diffusion_true),
            hurst=0.7, rng=np.random.default_rng(17)
        )
        data = fbm.simulate(x0=0.0, t=t, n_paths=1)[0]

        expected_var = diffusion_true ** 2 * dt ** (2 * 0.7)
        observed_var = np.var(np.diff(data))
        assert 0.6 * expected_var < observed_var < 1.5 * expected_var

    def test_fbm_estimate_diffusion_is_std_scale(self):
        """estimate_parameters must report diffusion in std-deviation units:
        on genuine fBM with unit diffusion the estimate is near 1 (a
        variance-scale convention would report dt**H ~ 0.2)."""
        dt = 0.1
        path = self._davies_harte_fbm(1024, 0.7, dt, seed=2)
        fbm = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=1.0), hurst=0.5
        )
        estimated = fbm.estimate_parameters(path, dt)

        assert 0.6 < estimated.diffusion < 1.6

    def test_fbm_log_likelihood_singular_covariance_is_neg_inf(self):
        """Zero diffusion gives a singular covariance matrix -> -inf."""
        fbm = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=0.0), hurst=0.7
        )
        data = np.zeros(21)
        assert fbm.log_likelihood(data, 0.1) == -np.inf

    def test_fbm_log_likelihood_is_finite_and_scale_sensitive(self):
        """On a regular series the multivariate-normal likelihood is finite
        and falls off in both directions away from the data's own scale."""
        fbm = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=1.0),
            hurst=0.7, rng=np.random.default_rng(9)
        )
        t = np.linspace(0, 3.0, 31)
        data = fbm.simulate(x0=0.0, t=t, n_paths=1)[0]
        dt = t[1] - t[0]

        ll_truth = fbm.log_likelihood(data, dt)
        ll_tiny = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=1e-6), hurst=0.7
        ).log_likelihood(data, dt)
        ll_huge = jumprope.FractionalBrownianMotion(
            jumprope.ModelParameters(drift=0.0, diffusion=1e6), hurst=0.7
        ).log_likelihood(data, dt)

        assert np.isfinite(ll_truth)
        assert ll_truth > ll_tiny
        assert ll_truth > ll_huge


class TestCoxIngersollRoss:
    """Test Cox-Ingersoll-Ross process implementation."""
    
    def test_cir_initialization(self):
        """Test CIR process initialization."""
        params = jumprope.ModelParameters(
            equilibrium=15.0,
            reversion_speed=0.5,
            diffusion=1.0
        )
        cir = jumprope.CoxIngersollRoss(params)
        
        assert cir.process_name == "Cox-Ingersoll-Ross"
        assert cir.parameters.equilibrium == 15.0
        assert cir.parameters.reversion_speed == 0.5
    
    def test_cir_non_negativity(self):
        """Test that CIR process ensures non-negative values."""
        params = jumprope.ModelParameters(
            equilibrium=5.0,
            reversion_speed=1.0,
            diffusion=2.0
        )
        cir = jumprope.CoxIngersollRoss(params)
        
        t = np.linspace(0, 10, 100)
        paths = cir.simulate(x0=5.0, t=t, n_paths=50)
        
        # All values should be positive
        assert np.all(paths > 0)
    
    def test_cir_mean_reversion(self):
        """Test mean reversion property."""
        params = jumprope.ModelParameters(
            equilibrium=10.0,
            reversion_speed=1.0,
            diffusion=0.5
        )
        cir = jumprope.CoxIngersollRoss(params)
        
        t = np.linspace(0, 20, 200)
        paths = cir.simulate(x0=20.0, t=t, n_paths=100)  # Start above equilibrium
        
        # Mean should converge toward equilibrium
        final_mean = np.mean(paths[:, -1])
        assert abs(final_mean - 10.0) < 5.0  # Should be close to equilibrium

    def test_cir_recovers_equilibrium(self):
        """Moment matching must recover the equilibrium level."""
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=1.0, diffusion=0.5
        )
        cir = jumprope.CoxIngersollRoss(params, rng=np.random.default_rng(1))
        t = np.linspace(0, 200, 4001)
        data = cir.simulate(x0=10.0, t=t, n_paths=1)[0]

        estimated = cir.estimate_parameters(data, dt=0.05)

        assert abs(estimated.equilibrium - 10.0) < 0.5
        assert abs(estimated.reversion_speed - 1.0) < 0.5
        assert np.isfinite(estimated.diffusion)
        assert estimated.diffusion > 0

    def test_cir_estimate_short_series_defaults_reversion(self):
        """With only two observations the autocorrelation is undefined: the
        estimator falls back to unit reversion speed while still recovering
        the level."""
        cir = jumprope.CoxIngersollRoss(
            jumprope.ModelParameters(equilibrium=1.0, reversion_speed=2.0, diffusion=0.5)
        )

        estimated = cir.estimate_parameters(np.array([1.0, 1.5]), dt=0.5)

        assert estimated.reversion_speed == 1.0
        assert abs(estimated.equilibrium - 1.25) < 1e-12
        assert estimated.diffusion > 0

    def test_cir_log_likelihood_prefers_true_parameters(self):
        params = jumprope.ModelParameters(
            equilibrium=10.0, reversion_speed=1.0, diffusion=0.5
        )
        cir = jumprope.CoxIngersollRoss(params, rng=np.random.default_rng(4))
        t = np.linspace(0, 100, 2001)
        data = cir.simulate(x0=10.0, t=t, n_paths=1)[0]
        dt = 0.05

        ll_true = cir.log_likelihood(data, dt)
        shifted = jumprope.CoxIngersollRoss(jumprope.ModelParameters(
            equilibrium=2.0, reversion_speed=5.0, diffusion=3.0
        ))
        ll_shifted = shifted.log_likelihood(data, dt)

        assert np.isfinite(ll_true)
        assert ll_true > ll_shifted


class TestLevyProcess:
    """Test Levy process implementation."""
    
    def test_levy_initialization(self):
        """Test Levy process initialization."""
        params = jumprope.ModelParameters(drift=0.0, diffusion=1.0)
        levy = jumprope.LevyProcess(params, levy_alpha=1.5, levy_beta=0.0,
                                     rng=np.random.default_rng(20260830))
        
        assert levy.process_name == "Levy Process"
        assert levy.levy_alpha == 1.5
        assert levy.levy_beta == 0.0
    
    def test_levy_simulation(self):
        """Test Levy process simulation."""
        params = jumprope.ModelParameters(drift=0.1, diffusion=1.0)
        levy = jumprope.LevyProcess(params, levy_alpha=1.8, levy_beta=0.0)
        
        t = np.linspace(0, 10, 100)
        paths = levy.simulate(x0=10.0, t=t, n_paths=50)
        
        assert paths.shape == (50, 100)
        assert np.all(paths[:, 0] == 10.0)
    
    def test_levy_heavy_tails(self):
        """Test that Levy process produces heavy-tailed distributions."""
        params = jumprope.ModelParameters(drift=0.0, diffusion=1.0)
        levy = jumprope.LevyProcess(params, levy_alpha=1.5, levy_beta=0.0,
                                     rng=np.random.default_rng(20260830))
        
        t = np.linspace(0, 10, 100)
        paths = levy.simulate(x0=0.0, t=t, n_paths=1000)
        
        # Check for extreme values (heavy tails)
        final_values = paths[:, -1]
        std_dev = np.std(final_values)
        extreme_values = np.sum(np.abs(final_values) > 3 * std_dev)
        
        # With heavy tails, we expect more extreme values than normal distribution
        assert extreme_values > 10  # More than expected for normal distribution

    def test_levy_recovers_stability_index(self):
        """The characteristic-function estimator must recover alpha < 2."""
        params = jumprope.ModelParameters(drift=0.0, diffusion=1.0)
        levy = jumprope.LevyProcess(params, levy_alpha=1.4, levy_beta=0.0,
                                    rng=np.random.default_rng(3))
        t = np.linspace(0, 100, 4001)
        data = levy.simulate(x0=0.0, t=t, n_paths=1)[0]

        estimated = levy.estimate_parameters(data, dt=0.025)

        assert abs(levy.levy_alpha - 1.4) < 0.3
        assert np.isfinite(estimated.diffusion)
        assert estimated.diffusion > 0

    def test_levy_cauchy_case_simulates_finite_paths(self):
        """The alpha == 1 branch of the Chambers-Mallows-Stuck generator
        (pure Cauchy) produces finite trajectories with the given start."""
        levy = jumprope.LevyProcess(
            jumprope.ModelParameters(drift=0.0, diffusion=1.0),
            levy_alpha=1.0, rng=np.random.default_rng(3)
        )

        paths = levy.simulate(x0=0.0, t=np.linspace(0, 10, 101), n_paths=3)

        assert paths.shape == (3, 101)
        assert np.all(paths[:, 0] == 0.0)
        assert np.isfinite(paths).all()

    def test_levy_log_likelihood_prefers_true_parameters(self):
        params = jumprope.ModelParameters(drift=0.1, diffusion=1.0)
        levy = jumprope.LevyProcess(params, levy_alpha=1.8, levy_beta=0.0,
                                    rng=np.random.default_rng(9))
        t = np.linspace(0, 100, 2001)
        data = levy.simulate(x0=0.0, t=t, n_paths=1)[0]
        dt = 0.05

        ll_true = levy.log_likelihood(data, dt)
        shifted = jumprope.LevyProcess(jumprope.ModelParameters(
            drift=1.0, diffusion=5.0
        ))
        ll_shifted = shifted.log_likelihood(data, dt)

        assert np.isfinite(ll_true)
        assert ll_true > ll_shifted


class TestAdvancedModelIntegration:
    """Test integration of advanced models with JumpRope."""
    
    def create_test_data(self):
        """Create test data for model fitting."""
        np.random.seed(42)
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5] * 10,
            'phenotype1': np.random.normal(10, 2, 50) + np.arange(50) * 0.1
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        return datacore.DataCore([ts_data])
    
    def test_fit_fractional_brownian(self):
        """Test fitting FBM model through JumpRope."""
        data_core = self.create_test_data()
        
        model = jumprope.JumpRope.fit(
            data_core,
            model_type='fractional-brownian',
            hurst=0.7,
            seed=42
        )
        
        assert model is not None
        assert model.fitted_parameters is not None
        assert isinstance(model.stochastic_process, jumprope.FractionalBrownianMotion)
    
    def test_fit_cir(self):
        """Test fitting CIR model through JumpRope."""
        data_core = self.create_test_data()
        
        model = jumprope.JumpRope.fit(
            data_core,
            model_type='cir',
            equilibrium=15.0,
            seed=42
        )
        
        assert model is not None
        assert isinstance(model.stochastic_process, jumprope.CoxIngersollRoss)
    
    def test_fit_levy(self):
        """Test fitting Levy process through JumpRope."""
        data_core = self.create_test_data()
        
        model = jumprope.JumpRope.fit(
            data_core,
            model_type='levy',
            levy_alpha=1.5,
            seed=42
        )
        
        assert model is not None
        assert isinstance(model.stochastic_process, jumprope.LevyProcess)


class TestAdvancedVisualizations:
    """Test advanced visualization methods."""
    
    def create_test_model(self):
        """Create test model with trajectories."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5] * 10,
            'phenotype1': np.random.normal(10, 2, 50)
        })
        
        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )
        
        data_core = datacore.DataCore([ts_data])
        model = jumprope.JumpRope.fit(data_core, model_type='jump-diffusion')
        model.generate_trajectories(n_samples=50, x0=10.0)
        
        return model
    
    def test_plot_heatmap(self):
        """Test trajectory density heatmap."""
        model = self.create_test_model()
        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        
        fig = visualizer.plot_heatmap(model, time_resolution=20, phenotype_resolution=20)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_violin(self):
        """Test violin plots."""
        model = self.create_test_model()
        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        
        fig = visualizer.plot_violin(model)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_ridge(self):
        """Test ridge plots."""
        model = self.create_test_model()
        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        
        fig = visualizer.plot_ridge(model, n_distributions=5)
        
        assert fig is not None
        assert len(fig.axes) >= 5
    
    def test_plot_phase_portrait(self):
        """Test phase portrait."""
        model = self.create_test_model()
        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        
        fig = visualizer.plot_phase_portrait(model, derivative_method='finite_difference')
        
        assert fig is not None
        assert len(fig.axes) > 0


class TestAdvancedAnalytics:
    """Test advanced statistical methods."""
    
    def create_test_data(self):
        """Create test data for analytics."""
        np.random.seed(42)
        time_points = np.arange(1, 101)
        
        data = pd.DataFrame({
            'time': time_points,
            'phenotype1': 10 + 0.1 * time_points + np.random.normal(0, 1, 100),
            'phenotype2': 15 + 0.15 * time_points + np.random.normal(0, 1.5, 100)
        })
        
        return data
    
    def test_copula_analysis(self):
        """Test copula analysis."""
        data = self.create_test_data()
        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')
        
        result = analytics.copula_analysis('phenotype1', 'phenotype2', copula_type='gaussian')
        
        assert 'copula_parameter' in result
        assert 'kendall_tau' in result
        assert 'spearman_rho' in result
        assert 'upper_tail_dependence' in result
        assert 'dependence_class' in result
    
    def test_extreme_value_analysis(self):
        """Test extreme value analysis."""
        data = self.create_test_data()
        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')
        
        result = analytics.extreme_value_analysis('phenotype1')
        
        assert 'pot_method' in result
        assert 'block_maxima_method' in result
        assert 'hill_estimator' in result
        assert 'tail_index' in result
        assert 'threshold' in result['pot_method']
    
    def test_regime_switching_analysis(self):
        """Test regime switching detection."""
        data = self.create_test_data()
        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')
        
        result = analytics.regime_switching_analysis('phenotype1', n_regimes=2)
        
        assert 'n_regimes' in result
        assert result['n_regimes'] == 2
        assert 'regime_labels' in result
        assert 'regime_statistics' in result
        assert 'transition_matrix' in result
        assert 'n_switches' in result
        assert len(result['regime_labels']) == 100
    
    def test_wavelet_analysis_no_pywt(self):
        """Test wavelet analysis gracefully handles missing PyWavelets."""
        data = self.create_test_data()
        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')
        
        try:
            result = analytics.wavelet_analysis('phenotype1')
            # If PyWavelets is installed, check result
            assert 'coefficients' in result or 'scales' in result
        except ImportError:
            # If PyWavelets is not installed, this is expected
            pass


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test handling of empty data."""
        data = pd.DataFrame({'time': [], 'phenotype': []})
        
        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype']
        )
        
        data_core = datacore.DataCore([ts_data])
        
        # Should handle gracefully
        with pytest.raises(ValueError, match="all time series data are empty"):
            model = jumprope.JumpRope.fit(data_core, model_type='cir')
    
    def test_single_trajectory(self):
        """Test visualization with single trajectory."""
        params = jumprope.ModelParameters()
        process = jumprope.OrnsteinUhlenbeckJump(params)
        model = jumprope.JumpRope(process, np.linspace(0, 10, 50))
        model.generate_trajectories(n_samples=1, x0=10.0)
        
        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        
        # Should work with single trajectory
        fig = visualizer.plot_violin(model)
        assert fig is not None
    
    def test_nan_handling_in_analytics(self):
        """Test NaN handling in analytics.

        copula_analysis drops rows with NaN in either column pairwise; with
        enough complete pairs remaining it returns a result, otherwise it
        raises ValueError (contract of the current analytics engine).
        """
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7],
            'phenotype1': [10, np.nan, 14, 16, 18, 20, 22],
            'phenotype2': [20, 22, np.nan, 26, 28, 30, 32]
        })

        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')

        # Should handle NaNs gracefully (drop incomplete pairs, fit on the rest)
        result = analytics.copula_analysis('phenotype1', 'phenotype2')
        assert result is not None
        assert 'kendall_tau' in result

    def test_nan_pairs_insufficient_variation_raises(self):
        """Only two complete (column1, column2) pairs survive NaN dropping,
        which is below the minimum for a copula fit -> ValueError."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, np.nan, 14, 16, np.nan],
            'phenotype2': [20, 22, np.nan, 26, 28]
        })

        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')

        with pytest.raises(ValueError, match="Insufficient variation"):
            analytics.copula_analysis('phenotype1', 'phenotype2')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
