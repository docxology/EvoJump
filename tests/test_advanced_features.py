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


class TestLevyProcess:
    """Test Levy process implementation."""
    
    def test_levy_initialization(self):
        """Test Levy process initialization."""
        params = jumprope.ModelParameters(drift=0.0, diffusion=1.0)
        levy = jumprope.LevyProcess(params, levy_alpha=1.5, levy_beta=0.0)
        
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
        levy = jumprope.LevyProcess(params, levy_alpha=1.5, levy_beta=0.0)
        
        t = np.linspace(0, 10, 100)
        paths = levy.simulate(x0=0.0, t=t, n_paths=1000)
        
        # Check for extreme values (heavy tails)
        final_values = paths[:, -1]
        std_dev = np.std(final_values)
        extreme_values = np.sum(np.abs(final_values) > 3 * std_dev)
        
        # With heavy tails, we expect more extreme values than normal distribution
        assert extreme_values > 10  # More than expected for normal distribution


class TestAdvancedModelIntegration:
    """Test integration of advanced models with JumpRope."""
    
    def create_test_data(self):
        """Create test data for model fitting."""
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
            hurst=0.7
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
            equilibrium=15.0
        )
        
        assert model is not None
        assert isinstance(model.stochastic_process, jumprope.CoxIngersollRoss)
    
    def test_fit_levy(self):
        """Test fitting Levy process through JumpRope."""
        data_core = self.create_test_data()
        
        model = jumprope.JumpRope.fit(
            data_core,
            model_type='levy',
            levy_alpha=1.5
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
        with pytest.raises(Exception):
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
        """Test NaN handling in analytics."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, np.nan, 14, 16, np.nan],
            'phenotype2': [20, 22, np.nan, 26, 28]
        })
        
        analytics = analytics_engine.AnalyticsEngine(data, time_column='time')
        
        # Should handle NaNs gracefully
        result = analytics.copula_analysis('phenotype1', 'phenotype2')
        assert result is not None
        assert 'kendall_tau' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
