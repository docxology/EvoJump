"""
Test suite for LaserPlane module.

This module tests the cross-sectional analysis functionality of the LaserPlane module
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
from evojump import datacore, jumprope, laserplane


class TestDistributionFitter:
    """Test DistributionFitter class."""

    def test_fit_distribution_normal(self):
        """Test fitting normal distribution."""
        fitter = laserplane.DistributionFitter()

        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 100)

        result = fitter.fit_distribution(data, distribution='normal')

        assert result['distribution'] == 'normal'
        assert result['parameters'] is not None
        assert len(result['parameters']) == 2  # mu, sigma
        assert result['aic'] is not None
        assert np.isfinite(result['aic'])

    def test_fit_distribution_lognormal(self):
        """Test fitting lognormal distribution."""
        fitter = laserplane.DistributionFitter()

        # Generate lognormal data
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 100)

        result = fitter.fit_distribution(data, distribution='lognormal')

        assert result['distribution'] == 'lognormal'
        assert result['parameters'] is not None
        assert len(result['parameters']) == 3  # s, loc, scale
        assert result['aic'] is not None

    def test_fit_distribution_auto_selection(self):
        """Test automatic distribution selection."""
        fitter = laserplane.DistributionFitter()

        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 100)

        result = fitter.fit_distribution(data, distribution='auto')

        assert result['distribution'] in fitter.supported_distributions
        assert result['parameters'] is not None
        assert result['aic'] is not None

    def test_fit_distribution_insufficient_data(self):
        """Test fitting with insufficient data."""
        fitter = laserplane.DistributionFitter()

        data = np.array([1, 2, 3])  # Too few data points

        result = fitter.fit_distribution(data, distribution='normal')

        assert result['distribution'] is None
        assert result['parameters'] is None
        assert result['aic'] == np.inf

    def test_fit_distribution_invalid_distribution(self):
        """Test fitting with invalid distribution."""
        fitter = laserplane.DistributionFitter()

        data = np.random.normal(10.0, 2.0, 100)

        with pytest.raises(ValueError, match="Unsupported distribution"):
            fitter.fit_distribution(data, distribution='invalid_distribution')


class TestDistributionComparer:
    """Test DistributionComparer class."""

    def test_compare_distributions_kolmogorov_smirnov(self):
        """Test Kolmogorov-Smirnov test."""
        comparer = laserplane.DistributionComparer()

        np.random.seed(42)
        data1 = np.random.normal(10.0, 2.0, 100)
        data2 = np.random.normal(10.5, 2.0, 100)

        result = comparer.compare_distributions(data1, data2, test='ks')

        assert result['test'] == 'kolmogorov_smirnov'
        assert result['statistic'] is not None
        assert result['p_value'] is not None
        assert isinstance(result['significant'], (bool, np.bool_))

    def test_compare_distributions_mann_whitney(self):
        """Test Mann-Whitney U test."""
        comparer = laserplane.DistributionComparer()

        np.random.seed(42)
        data1 = np.random.normal(10.0, 2.0, 50)
        data2 = np.random.normal(12.0, 2.0, 50)

        result = comparer.compare_distributions(data1, data2, test='mann_whitney')

        assert result['test'] == 'mann_whitney'
        assert result['statistic'] is not None
        assert result['p_value'] is not None
        assert isinstance(result['significant'], (bool, np.bool_))

    def test_compare_distributions_insufficient_data(self):
        """Test comparison with insufficient data."""
        comparer = laserplane.DistributionComparer()

        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        result = comparer.compare_distributions(data1, data2, test='ks')

        assert result['test'] is None
        assert result['statistic'] is None
        assert result['p_value'] is None

    def test_compare_distributions_identical_data(self):
        """Test comparison with identical data."""
        comparer = laserplane.DistributionComparer()

        np.random.seed(42)
        data1 = np.random.normal(10.0, 2.0, 100)
        data2 = data1.copy()  # Identical data

        result = comparer.compare_distributions(data1, data2, test='ks')

        assert result['p_value'] > 0.05  # Should not be significant
        assert not result['significant']


class TestMomentAnalyzer:
    """Test MomentAnalyzer class."""

    def test_compute_moments_basic(self):
        """Test basic moment computation."""
        analyzer = laserplane.MomentAnalyzer()

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        moments = analyzer.compute_moments(data)

        assert moments['mean'] == 5.5
        assert moments['variance'] == pytest.approx(9.166666666666666, rel=1e-10)  # Sample variance (ddof=1)
        assert moments['std'] == pytest.approx(np.sqrt(9.166666666666666), rel=1e-10)
        assert moments['median'] == 5.5
        assert isinstance(moments['skewness'], float)
        assert isinstance(moments['kurtosis'], float)

    def test_compute_moments_with_nan(self):
        """Test moment computation with NaN values."""
        analyzer = laserplane.MomentAnalyzer()

        data = np.array([1, 2, np.nan, 4, 5])

        moments = analyzer.compute_moments(data)

        assert moments['mean'] == 3.0  # Mean of [1, 2, 4, 5]
        assert moments['variance'] == pytest.approx(3.3333333333333335, rel=1e-10)  # Sample variance of [1, 2, 4, 5]
        assert np.isfinite(moments['mean'])

    def test_compute_quantiles(self):
        """Test quantile computation."""
        analyzer = laserplane.MomentAnalyzer()

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        quantiles = analyzer.compute_quantiles(data, quantiles=[0.25, 0.5, 0.75])

        assert quantiles['q0.25'] == 3.25
        assert quantiles['q0.50'] == 5.5
        assert quantiles['q0.75'] == 7.75

    def test_compute_confidence_intervals(self):
        """Test confidence interval computation."""
        analyzer = laserplane.MomentAnalyzer()

        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 100)

        ci = analyzer.compute_confidence_intervals(data, confidence_level=0.95)

        assert 'mean_ci' in ci
        assert 'median_ci' in ci
        assert 'std_ci' in ci

        mean_ci = ci['mean_ci']
        assert len(mean_ci) == 2
        assert mean_ci[0] < mean_ci[1]  # Lower < upper
        assert mean_ci[0] <= 10.0 <= mean_ci[1]  # Mean should be within CI

    def test_estimate_mode(self):
        """Test mode estimation."""
        analyzer = laserplane.MomentAnalyzer()

        # Unimodal data
        data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])

        mode = analyzer._estimate_mode(data)

        assert abs(mode - 3.0) < 0.1  # Should be close to most frequent value

    def test_moments_empty_data(self):
        """Test moment computation with empty data."""
        analyzer = laserplane.MomentAnalyzer()

        data = np.array([])

        moments = analyzer.compute_moments(data)

        assert all(np.isnan(v) for v in moments.values() if isinstance(v, float))


class TestLaserPlaneAnalyzer:
    """Test LaserPlaneAnalyzer class."""

    def create_test_jump_rope(self):
        """Create test JumpRope model for analyzer tests."""
        # Create test data
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        # Generate trajectories
        model.generate_trajectories(n_samples=50, x0=10.0)

        return model

    def test_laser_plane_analyzer_initialization(self):
        """Test LaserPlaneAnalyzer initialization."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        assert analyzer.jump_rope == model
        assert analyzer.fitter is not None
        assert analyzer.comparer is not None
        assert analyzer.moment_analyzer is not None

    def test_analyze_cross_section(self):
        """Test cross-section analysis."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        result = analyzer.analyze_cross_section(time_point=3.0, n_bootstrap=100)

        assert isinstance(result, laserplane.CrossSectionResult)
        assert result.time_point == 3.0
        assert len(result.data) == 50  # Number of trajectories
        assert result.distribution_fit is not None
        assert result.moments is not None
        assert result.quantiles is not None
        assert result.goodness_of_fit is not None
        assert result.confidence_intervals is not None

    def test_analyze_cross_section_multiple_times(self):
        """Test cross-section analysis at multiple time points."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        time_points = [1.0, 2.0, 3.0, 4.0, 5.0]

        for time_point in time_points:
            result = analyzer.analyze_cross_section(time_point)

            assert result.time_point == time_point
            assert len(result.data) == 50
            assert np.isfinite(result.moments['mean'])

    def test_compare_distributions(self):
        """Test distribution comparison."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        # Create condition data for comparison
        np.random.seed(42)
        condition_data = {
            'condition1': np.random.normal(15.0, 2.0, 50),
            'condition2': np.random.normal(16.0, 2.5, 50)
        }

        comparison = analyzer.compare_distributions(
            time_point=3.0,
            condition_data=condition_data,
            test='ks'
        )

        assert isinstance(comparison, laserplane.DistributionComparison)
        assert comparison.time_point == 3.0
        assert comparison.distribution1_name == 'reference'
        assert comparison.distribution2_name == ['condition1', 'condition2']
        assert isinstance(comparison.significant_differences, list)

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval computation."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        # Generate test data
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 100)

        ci = analyzer._bootstrap_confidence_intervals(data, n_bootstrap=200)

        assert 'mean_ci' in ci
        assert 'median_ci' in ci
        assert 'std_ci' in ci

        mean_ci = ci['mean_ci']
        assert len(mean_ci) == 2
        assert mean_ci[0] < mean_ci[1]
        assert mean_ci[0] <= 10.0 <= mean_ci[1]  # Mean should be within CI

    def test_assess_goodness_of_fit(self):
        """Test goodness of fit assessment."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 100)

        # Fit normal distribution (with aic)
        distribution_fit = {
            'distribution': 'normal',
            'parameters': (10.0, 2.0),
            'log_likelihood': -200.0,
            'aic': -150.0
        }

        gof = analyzer._assess_goodness_of_fit(data, distribution_fit)

        assert 'aic' in gof
        assert 'bic' in gof
        assert 'ks_statistic' in gof
        assert 'ks_p_value' in gof
        assert np.isfinite(gof['ks_statistic'])

    def test_generate_summary_report(self):
        """Test summary report generation."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        time_points = [1.0, 3.0, 5.0]

        report = analyzer.generate_summary_report(time_points)

        assert isinstance(report, str)
        assert 'time_point' in report
        assert 'n_samples' in report
        assert 'mean' in report
        assert 'distribution' in report

    def test_cross_section_analysis_with_different_distributions(self):
        """Test cross-section analysis with different data distributions."""
        # Create test data with different distributions
        for dist_name in ['normal', 'lognormal', 'gamma']:
            if dist_name == 'normal':
                data = pd.DataFrame({
                    'time': [1, 2, 3, 4, 5] * 10,
                    'phenotype1': np.random.normal(10, 2, 50)
                })
            elif dist_name == 'lognormal':
                data = pd.DataFrame({
                    'time': [1, 2, 3, 4, 5] * 10,
                    'phenotype1': np.random.lognormal(0, 0.5, 50)
                })
            elif dist_name == 'gamma':
                data = pd.DataFrame({
                    'time': [1, 2, 3, 4, 5] * 10,
                    'phenotype1': np.random.gamma(2, 2, 50)
                })

            ts_data = datacore.TimeSeriesData(
                data=data,
                time_column='time',
                phenotype_columns=['phenotype1']
            )

            data_core = datacore.DataCore([ts_data])

            model = jumprope.JumpRope.fit(
                data_core,
                model_type='jump-diffusion',
                time_points=np.array([1, 2, 3, 4, 5])
            )

            model.generate_trajectories(n_samples=30, x0=10.0)

            analyzer = laserplane.LaserPlaneAnalyzer(model)

            result = analyzer.analyze_cross_section(time_point=3.0)

            assert result.distribution_fit is not None
            assert result.goodness_of_fit['aic'] is not None
            assert np.isfinite(result.moments['mean'])

    def test_cross_section_analysis_edge_cases(self):
        """Test cross-section analysis edge cases."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        # Test with small sample size
        small_model = self.create_test_jump_rope()
        small_model.trajectories = small_model.trajectories[:5]  # Reduce to 5 trajectories

        small_analyzer = laserplane.LaserPlaneAnalyzer(small_model)

        result = small_analyzer.analyze_cross_section(time_point=3.0)

        assert result is not None
        assert len(result.data) == 5
        assert np.isfinite(result.moments['mean'])

    def test_bootstrap_with_small_sample(self):
        """Test bootstrap with small sample size."""
        model = self.create_test_jump_rope()

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        # Small dataset
        data = np.array([1, 2, 3, 4, 5])

        ci = analyzer._bootstrap_confidence_intervals(data, n_bootstrap=50)

        assert 'mean_ci' in ci
        assert 'median_ci' in ci
        assert 'std_ci' in ci

        # Should handle small data gracefully
        assert len(ci['mean_ci']) == 2
        assert ci['mean_ci'][0] <= ci['mean_ci'][1]
