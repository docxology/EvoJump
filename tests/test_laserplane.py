"""
Test suite for LaserPlane module.

This module tests the cross-sectional analysis functionality of the LaserPlane module
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope, laserplane
from scipy import stats
from scipy.stats import beta, uniform, kstest
from conftest import make_growth_frame



class TestDistributionFitter:
    """Test DistributionFitter class."""

    def test_fit_distribution_normal(self):
        """Test fitting normal distribution."""
        fitter = laserplane.DistributionFitter()

        # Generate normal data
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

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
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

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

        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

        with pytest.raises(ValueError, match="Unsupported distribution"):
            fitter.fit_distribution(data, distribution='invalid_distribution')

    def test_fit_distribution_uniform(self):
        """Test fitting uniform distribution: fitted params round-trip through KS."""
        fitter = laserplane.DistributionFitter()

        rng = np.random.default_rng(42)
        data = rng.uniform(0.0, 10.0, 300)

        result = fitter.fit_distribution(data, distribution='uniform')

        assert result['distribution'] == 'uniform'
        assert result['parameters'][0] == pytest.approx(data.min(), abs=1e-12)
        assert result['parameters'][1] == pytest.approx(data.max() - data.min(), abs=1e-12)
        assert np.isfinite(result['aic'])

        # The fitted uniform must not be rejected on the data it was fitted to
        ks_stat, ks_p = kstest(data, uniform(*result['parameters']).cdf)
        assert ks_p > 0.05

    def test_fit_distribution_beta(self):
        """Test fitting beta distribution on data in (0, 1)."""
        fitter = laserplane.DistributionFitter()

        rng = np.random.default_rng(43)
        data = rng.beta(2.0, 5.0, 200)

        result = fitter.fit_distribution(data, distribution='beta')

        assert result['distribution'] == 'beta'
        assert result['parameters'] is not None
        assert np.isfinite(result['aicc'])

        # Log-likelihood must be computed on the scaled data the parameters
        # were estimated on, with the change-of-variables Jacobian so the
        # value is comparable with models fitted on the original scale.
        lo, hi = result['scale']
        fit_data = result['fit_data']
        expected_ll = beta.logpdf(fit_data, *result['parameters']).sum() \
            - len(fit_data) * np.log(hi - lo)
        assert result['log_likelihood'] == pytest.approx(expected_ll, rel=1e-9)

    def test_fit_distribution_lognormal_with_nonpositive_values(self):
        """Lognormal fit on data containing non-positive values must have finite ICs."""
        fitter = laserplane.DistributionFitter()

        rng = np.random.default_rng(44)
        data = np.concatenate([rng.lognormal(0.0, 0.5, 100), [0.0, -1.0]])

        result = fitter.fit_distribution(data, distribution='lognormal')

        assert result['distribution'] == 'lognormal'
        assert np.isfinite(result['aic'])
        assert np.isfinite(result['bic'])
        assert np.isfinite(result['aicc'])
        assert result['n_fit'] == len(data) - 2  # fitted on the positive subset


    @pytest.mark.parametrize('dist_name', ['lognormal', 'gamma'])
    def test_fit_positive_support_distribution_without_enough_positive_values(self, dist_name):
        """Fewer than four positive observations yields the no-fit sentinel."""
        fitter = laserplane.DistributionFitter()

        data = np.array([-3.0, -1.0, -2.0, -0.5])

        result = fitter.fit_distribution(data, dist_name)

        assert result == {'distribution': None, 'parameters': None, 'aic': np.inf}

class TestDistributionComparer:
    """Test DistributionComparer class."""

    def test_compare_distributions_kolmogorov_smirnov(self):
        """Test Kolmogorov-Smirnov test."""
        comparer = laserplane.DistributionComparer()

        rng = np.random.default_rng(42)
        data1 = rng.normal(10.0, 2.0, 100)
        data2 = rng.normal(10.5, 2.0, 100)

        result = comparer.compare_distributions(data1, data2, test='ks')

        assert result['test'] == 'kolmogorov_smirnov'
        assert result['statistic'] is not None
        assert result['p_value'] is not None
        assert isinstance(result['significant'], (bool, np.bool_))

    def test_compare_distributions_mann_whitney(self):
        """Test Mann-Whitney U test."""
        comparer = laserplane.DistributionComparer()

        rng = np.random.default_rng(42)
        data1 = rng.normal(10.0, 2.0, 50)
        data2 = rng.normal(12.0, 2.0, 50)

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

        rng = np.random.default_rng(42)
        data1 = rng.normal(10.0, 2.0, 100)
        data2 = data1.copy()  # Identical data

        result = comparer.compare_distributions(data1, data2, test='ks')

        assert result['p_value'] > 0.05  # Should not be significant
        assert not result['significant']

    def test_compare_distributions_rng_reproducible(self):
        """Permutation-based tests must be reproducible for a fixed rng."""
        comparer = laserplane.DistributionComparer()

        rng = np.random.default_rng(42)
        data1 = rng.normal(10.0, 2.0, 60)
        data2 = rng.normal(10.5, 2.0, 60)

        result1 = comparer.compare_distributions(
            data1, data2, test='cramer', rng=np.random.default_rng(123))
        result2 = comparer.compare_distributions(
            data1, data2, test='cramer', rng=np.random.default_rng(123))

        assert result1['p_value'] == result2['p_value']
        assert 0.0 <= result1['p_value'] <= 1.0

    def test_compare_distributions_auto_selects_ks(self):
        """test='auto' must route to the two-sample Kolmogorov-Smirnov test."""
        comparer = laserplane.DistributionComparer()

        rng = np.random.default_rng(60)
        data1 = rng.normal(0.0, 1.0, 60)
        data2 = rng.normal(0.5, 1.0, 60)

        result = comparer.compare_distributions(data1, data2, test='auto')

        assert result['test'] == 'kolmogorov_smirnov'
        assert np.isfinite(result['statistic'])
        assert 0.0 <= result['p_value'] <= 1.0

    def test_compare_distributions_unknown_test_raises(self):
        """An unsupported test name must raise instead of silently routing."""
        comparer = laserplane.DistributionComparer()

        rng = np.random.default_rng(61)
        data1 = rng.normal(0.0, 1.0, 60)
        data2 = rng.normal(0.0, 1.0, 60)

        with pytest.raises(ValueError, match="Unsupported test"):
            comparer.compare_distributions(data1, data2, test='permutation')

    def test_anderson_falls_back_to_permutation_on_degenerate_input(self):
        """Identical observations break anderson_ksamp; the permutation
        fallback must take over and report an uninformative p-value."""
        comparer = laserplane.DistributionComparer()

        result = comparer.compare_distributions(
            np.full(10, 2.5), np.full(10, 2.5), 'anderson',
            rng=np.random.default_rng(0))

        assert result['method'] == 'anderson_ksamp_permutation'
        assert result['p_value'] == 1.0  # every relabeling of identical data yields the same statistic
        assert not result['significant']
        assert np.isfinite(result['statistic'])



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

        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

        ci = analyzer.compute_confidence_intervals(data, confidence_level=0.95)

        assert 'mean_ci' in ci
        assert 'median_ci' in ci
        assert 'std_ci' in ci

        mean_ci = ci['mean_ci']
        assert len(mean_ci) == 2
        assert mean_ci[0] < mean_ci[1]  # Lower < upper
        assert mean_ci[0] <= 10.0 <= mean_ci[1]  # Mean should be within CI

        # median_ci must be a real confidence interval for the median (the
        # exact order-statistic interval: ranks 40 and 61 for n=100), not the
        # central 95% range of the data.
        median_ci = ci['median_ci']
        assert median_ci[0] < median_ci[1]
        assert median_ci[0] <= np.median(data) <= median_ci[1]
        sorted_data = np.sort(data)
        assert median_ci[0] == sorted_data[39]
        assert median_ci[1] == sorted_data[60]
        assert median_ci[0] > np.quantile(data, 0.025)  # narrower than data range

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

    def test_compute_quantiles_all_nan_returns_nan_map(self):
        """Quantiles of all-NaN data are NaN, keyed by the default grid."""
        analyzer = laserplane.MomentAnalyzer()

        quantiles = analyzer.compute_quantiles(np.array([np.nan, np.nan]))

        assert set(quantiles) == {'q0.05', 'q0.25', 'q0.50', 'q0.75', 'q0.95'}
        assert all(np.isnan(v) for v in quantiles.values())

    def test_compute_confidence_intervals_single_observation_returns_nan(self):
        """With fewer than two observations every confidence interval is undefined."""
        analyzer = laserplane.MomentAnalyzer()

        ci = analyzer.compute_confidence_intervals(np.array([3.0]))

        assert set(ci) == {'mean_ci', 'median_ci', 'std_ci'}
        for lo, hi in ci.values():
            assert np.isnan(lo) and np.isnan(hi)

    def test_estimate_mode_nan_data_returns_nan(self):
        """Mode estimation on non-finite data must return NaN, not crash."""
        analyzer = laserplane.MomentAnalyzer()

        assert np.isnan(analyzer._estimate_mode(np.array([np.nan, np.nan])))



def build_jump_rope(frame=None, time_points=None, n_samples=50, seed=42,
                    generate=True):
    """Build a JumpRope model on a synthetic growth frame.

    Uses the shared conftest frame builder unless a custom frame is given;
    simulated trajectories are seeded so every derived test is deterministic.
    Set ``generate=False`` for a fitted model without trajectories.
    """
    if frame is None:
        frame = make_growth_frame(n_points=5, phenotype_cols=("phenotype1",),
                                  seed=seed)
    if time_points is None:
        time_points = frame['time'].dropna().unique()
    ts_data = datacore.TimeSeriesData(
        data=frame,
        time_column='time',
        phenotype_columns=[c for c in frame.columns if c != 'time']
    )

    data_core = datacore.DataCore([ts_data])

    model = jumprope.JumpRope.fit(
        data_core,
        model_type='jump-diffusion',
        time_points=np.asarray(time_points, dtype=float)
    )

    if generate:
        model.generate_trajectories(n_samples=n_samples, x0=10.0, seed=seed)

    return model


@pytest.fixture
def laser_model():
    """Default seeded JumpRope model with 50 simulated trajectories."""
    return build_jump_rope()




CROSS_SECTION_FRAME_BUILDERS = {
    'normal': lambda rng: rng.normal(10.0, 2.0, 50),
    'lognormal': lambda rng: rng.lognormal(0.0, 0.5, 50),
    'gamma': lambda rng: rng.gamma(2.0, 2.0, 50),
}


class TestLaserPlaneAnalyzer:
    """Test LaserPlaneAnalyzer class."""
    def test_laser_plane_analyzer_initialization(self, laser_model):
        """Test LaserPlaneAnalyzer initialization."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        assert analyzer.jump_rope == laser_model
        assert isinstance(analyzer.fitter, laserplane.DistributionFitter)
        assert isinstance(analyzer.comparer, laserplane.DistributionComparer)
        assert isinstance(analyzer.moment_analyzer, laserplane.MomentAnalyzer)

    def test_analyze_cross_section(self, laser_model):
        """Test cross-section analysis."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        result = analyzer.analyze_cross_section(time_point=3.0, n_bootstrap=100)

        assert isinstance(result, laserplane.CrossSectionResult)
        assert result.time_point == 3.0
        assert len(result.data) == 50  # Number of trajectories
        assert result.distribution_fit is not None
        assert result.moments is not None
        assert result.quantiles is not None
        assert result.goodness_of_fit is not None
        assert result.confidence_intervals is not None

    def test_analyze_cross_section_multiple_times(self, laser_model):
        """Test cross-section analysis at multiple time points."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        time_points = [0.0, 1.0, 2.0, 3.0, 4.0]

        for time_point in time_points:
            result = analyzer.analyze_cross_section(time_point)

            assert result.time_point == time_point
            assert len(result.data) == 50
            assert np.isfinite(result.moments['mean'])

    def test_compare_distributions(self, laser_model):
        """Test distribution comparison populates statistics, p-values and effect sizes."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        # Create condition data for comparison: condition2 sits ~3 sd above
        # the reference cross-section at t=3, so it must come out significant.
        rng = np.random.default_rng(42)
        condition_data = {
            'condition1': rng.normal(15.0, 2.0, 50),
            'condition2': rng.normal(21.0, 2.0, 50)
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

        # The comparison must report per-condition statistics, not empty dicts
        assert set(comparison.p_values) == {'condition1', 'condition2'}
        assert set(comparison.test_statistics) == {'condition1', 'condition2'}
        assert all(np.isfinite(v) for v in comparison.test_statistics.values())
        assert all(np.isfinite(v) for v in comparison.p_values.values())
        assert set(comparison.effect_sizes) == {'condition1', 'condition2'}
        assert all(np.isfinite(v) for v in comparison.effect_sizes.values())

        # The clearly shifted condition must be flagged significant
        assert comparison.p_values['condition2'] < 0.05
        assert 'condition2' in comparison.significant_differences
        assert comparison.effect_sizes['condition2'] > 0  # shifted upward

    def test_bootstrap_confidence_intervals(self, laser_model):
        """Test bootstrap confidence interval computation."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        # Generate test data
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

        ci = analyzer._bootstrap_confidence_intervals(
            data, n_bootstrap=200, rng=np.random.default_rng(7))

        assert 'mean_ci' in ci
        assert 'median_ci' in ci
        assert 'std_ci' in ci

        mean_ci = ci['mean_ci']
        assert len(mean_ci) == 2
        assert mean_ci[0] < mean_ci[1]
        assert mean_ci[0] <= 10.0 <= mean_ci[1]  # Mean should be within CI

    def test_assess_goodness_of_fit(self, laser_model):
        """Test goodness of fit assessment."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        # Generate normal data
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

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

    def test_assess_goodness_of_fit_no_distribution(self, laser_model):
        """GOF sentinel values are returned when no distribution was fitted."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 100)

        gof = analyzer._assess_goodness_of_fit(
            data, {'distribution': None, 'parameters': None, 'aic': np.inf})

        assert gof == {'aic': np.inf, 'bic': np.inf,
                       'ks_statistic': np.nan, 'ks_p_value': np.nan}

    def test_generate_summary_report(self, laser_model):
        """Test summary report generation."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        time_points = [0.0, 2.0, 4.0]

        report = analyzer.generate_summary_report(time_points)

        assert isinstance(report, str)
        assert 'time_point' in report
        assert 'n_samples' in report
        assert 'mean' in report
        assert 'distribution' in report

    def test_generate_summary_report_writes_csv(self, laser_model, tmp_path):
        """output_file must receive a CSV whose rows match the report."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        out = tmp_path / "summary.csv"
        report = analyzer.generate_summary_report([0.0, 2.0], output_file=out)

        assert out.exists()
        df = pd.read_csv(out)
        assert list(df['time_point']) == [0.0, 2.0]
        assert (df['n_samples'] == 50).all()
        assert np.isfinite(df['aic']).all()
        expected_cols = {'time_point', 'n_samples', 'mean', 'std',
                         'distribution', 'aic'}
        assert expected_cols.issubset(df.columns)
        assert all(col in report for col in expected_cols)

    @pytest.mark.parametrize('dist_name', sorted(CROSS_SECTION_FRAME_BUILDERS))
    def test_cross_section_analysis_with_different_distributions(self, dist_name):
        """Cross-section analysis works for varied phenotype data shapes."""
        rng = np.random.default_rng(51)
        frame = pd.DataFrame({
            'time': np.repeat(np.arange(1.0, 6.0), 10),
            'phenotype1': CROSS_SECTION_FRAME_BUILDERS[dist_name](rng)
        })

        model = build_jump_rope(frame=frame, time_points=np.arange(1.0, 6.0),
                                n_samples=30, seed=52)

        analyzer = laserplane.LaserPlaneAnalyzer(model)

        result = analyzer.analyze_cross_section(time_point=3.0)

        assert result.distribution_fit is not None
        assert result.goodness_of_fit['aic'] is not None
        assert np.isfinite(result.moments['mean'])

    def test_cross_section_analysis_small_sample(self):
        """Cross-section analysis works with only 5 trajectories."""
        analyzer = laserplane.LaserPlaneAnalyzer(build_jump_rope(n_samples=5, seed=53))

        result = analyzer.analyze_cross_section(time_point=3.0)

        assert isinstance(result, laserplane.CrossSectionResult)
        assert len(result.data) == 5
        assert np.isfinite(result.moments['mean'])

    def test_bootstrap_with_small_sample(self, laser_model):
        """Test bootstrap with small sample size."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        # Small dataset
        data = np.array([1, 2, 3, 4, 5])

        ci = analyzer._bootstrap_confidence_intervals(
            data, n_bootstrap=50, rng=np.random.default_rng(7))

        assert 'mean_ci' in ci
        assert 'median_ci' in ci
        assert 'std_ci' in ci

        # Should handle small data gracefully
        assert len(ci['mean_ci']) == 2
        assert ci['mean_ci'][0] <= ci['mean_ci'][1]

    def test_analyze_cross_section_reproducible_with_seeded_rng(self):
        """Same model and seed must give identical bootstrap intervals."""
        analyzer1 = laserplane.LaserPlaneAnalyzer(build_jump_rope(seed=7))
        analyzer2 = laserplane.LaserPlaneAnalyzer(build_jump_rope(seed=7))

        result1 = analyzer1.analyze_cross_section(
            2.0, n_bootstrap=200, rng=np.random.default_rng(99))
        result2 = analyzer2.analyze_cross_section(
            2.0, n_bootstrap=200, rng=np.random.default_rng(99))

        assert result1.confidence_intervals == result2.confidence_intervals
        mean_ci = result1.confidence_intervals['mean_ci']
        assert mean_ci[0] < mean_ci[1]

    def test_bootstrap_confidence_intervals_insufficient_data_returns_nan(self, laser_model):
        """Fewer than four observations yield all-NaN bootstrap intervals."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        ci = analyzer._bootstrap_confidence_intervals(
            np.array([1.0, 2.0, 3.0]), n_bootstrap=10, rng=np.random.default_rng(7))

        assert set(ci) == {'mean_ci', 'median_ci', 'std_ci'}
        for lo, hi in ci.values():
            assert np.isnan(lo) and np.isnan(hi)

    def test_assess_goodness_of_fit_bad_parameters_yield_nan_ks(self, laser_model):
        """Parameters that cannot build a frozen CDF give NaN KS and finite BIC."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        rng = np.random.default_rng(70)
        data = rng.normal(10.0, 2.0, 100)

        gof = analyzer._assess_goodness_of_fit(
            data,
            {'distribution': 'uniform', 'parameters': ('bad',),
             'aic': 1.0, 'log_likelihood': 0.0})

        assert np.isnan(gof['ks_statistic']) and np.isnan(gof['ks_p_value'])
        assert np.isfinite(gof['bic'])
        assert gof['aic'] == 1.0

    def test_compare_distributions_omits_tiny_condition_from_statistics(self, laser_model):
        """A single-observation condition yields no p-value and no effect size."""
        analyzer = laserplane.LaserPlaneAnalyzer(laser_model)

        comparison = analyzer.compare_distributions(2.0, {'singleton': np.array([5.0])})

        assert comparison.p_values == {}
        assert comparison.effect_sizes == {}
        assert comparison.significant_differences == []

    def test_cohens_d_singleton_group_is_none(self):
        """Cohen's d is undefined for a one-observation group."""
        assert laserplane.LaserPlaneAnalyzer._cohens_d(
            np.array([1.0, 2.0, 3.0]), np.array([7.0])) is None

    def test_cohens_d_zero_pooled_variance_is_none(self):
        """Cohen's d is undefined when the pooled variance is zero."""
        same = np.full(5, 3.0)
        assert laserplane.LaserPlaneAnalyzer._cohens_d(same, same.copy()) is None

    def test_cohens_d_matches_pooled_sd_definition(self):
        """Cohen's d equals the mean shift over the pooled standard deviation."""
        rng = np.random.default_rng(71)
        group1 = rng.normal(0.0, 1.0, 50)
        group2 = rng.normal(1.2, 1.0, 50)

        n1, n2 = len(group1), len(group2)
        pooled = np.sqrt(((n1 - 1) * group1.var(ddof=1) + (n2 - 1) * group2.var(ddof=1))
                         / (n1 + n2 - 2))

        assert laserplane.LaserPlaneAnalyzer._cohens_d(group1, group2) == pytest.approx(
            (group2.mean() - group1.mean()) / pooled)

    def test_generate_summary_report_raises_when_every_time_point_fails(self):
        """Without trajectories every cross-section fails; the reporter must
        surface a RuntimeError instead of returning an empty report."""
        model = build_jump_rope(generate=False)
        analyzer = laserplane.LaserPlaneAnalyzer(model)

        with pytest.raises(RuntimeError, match="no summary report"):
            analyzer.generate_summary_report([0.0, 2.0])
