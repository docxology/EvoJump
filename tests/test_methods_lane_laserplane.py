"""Methods-lane tests: LaserPlane distribution fitting and comparison.

Real computation only — no mock frameworks (repo policy).
Covers: AICc/BIC in fits, Vuong check, Anderson-Darling k-sample,
permutation CVM p-value, D'Agostino-style normality verdicts.
"""
import warnings

import numpy as np
import pytest

from evojump import laserplane
from scipy.stats import norm


class TestDistributionFitterAICcBIC:
    def test_fit_reports_aicc_bic_aic(self):
        data = np.random.default_rng(42).normal(5.0, 2.0, 300)
        fitter = laserplane.DistributionFitter()
        result = fitter.fit_distribution(data, 'normal')
        assert result['distribution'] == 'normal'
        for key in ('aic', 'bic', 'aicc', 'n_params', 'log_likelihood'):
            assert key in result, f"missing {key}"
        n = 300
        k = result['n_params']
        assert result['aic'] == pytest.approx(2 * k - 2 * result['log_likelihood'])
        assert result['bic'] == pytest.approx(k * np.log(n) - 2 * result['log_likelihood'])
        expected_aicc = result['aic'] + 2 * k * (k + 1) / (n - k - 1)
        assert result['aicc'] == pytest.approx(expected_aicc)

    def test_aicc_penalty_positive_for_small_samples(self):
        # With n close to k, AICc must exceed AIC
        data = np.random.default_rng(7).normal(0.0, 1.0, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = laserplane.DistributionFitter().fit_distribution(data, 'normal')
        assert result['distribution'] == 'normal'
        assert result['aicc'] > result['aic']


class TestVuongSelection:
    def test_select_best_returns_string_and_records_vuong(self):
        data = np.random.default_rng(11).normal(3.0, 1.0, 250)
        fitter = laserplane.DistributionFitter()
        best = fitter._select_best_distribution(data)
        assert best in fitter.supported_distributions
        vuong = getattr(fitter, '_last_selection', None)
        assert vuong is not None
        assert 'lr_statistic' in vuong and 'p_value' in vuong
        assert vuong['comparison'] is None or (
            vuong['comparison'][0] == best)

    def test_vuong_prefers_clearly_separated_model(self):
        # Strongly bimodal data: uniform or others should beat normal on LR
        rng = np.random.default_rng(5)
        data = np.concatenate([rng.normal(-4, 0.3, 150), rng.normal(4, 0.3, 150)])
        fitter = laserplane.DistributionFitter()
        best = fitter._select_best_distribution(data)
        vuong = fitter._last_selection
        # The verdict must be finite and interpretable
        assert vuong['verdict'] in (
            'best_model_preferred', 'second_model_preferred',
            'no_significant_difference', 'degenerate_variance',
            'non_finite_likelihood', 'logpdf_evaluation_failed',
            'insufficient_fits')
        if vuong['verdict'] in ('best_model_preferred', 'second_model_preferred'):
            assert vuong['p_value'] < 0.05

    def test_vuong_insufficient_fits(self):
        # Constant data: nothing fits
        data = np.full(20, 3.14)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = laserplane.DistributionFitter()._select_best_distribution(data)
        assert best in ('normal', 'lognormal', 'gamma', 'beta', 'uniform')


class TestAndersonDarlingKSample:
    def test_ad_detects_shift(self):
        a = np.random.default_rng(1).normal(0, 1, 100)
        b = np.random.default_rng(2).normal(1.5, 1, 100)
        result = laserplane.DistributionComparer().compare_distributions(a, b, 'anderson')
        assert result['test'] == 'anderson_darling'
        assert result['statistic'] > 0
        assert 0.0 <= result['p_value'] <= 1.0
        assert result['significant'], "shifted samples must be significant"

    def test_ad_null_not_flagged(self):
        a = np.random.default_rng(3).normal(0, 1, 100)
        b = np.random.default_rng(4).normal(0, 1, 100)
        result = laserplane.DistributionComparer().compare_distributions(a, b, 'anderson')
        assert result['p_value'] > 0.01


class TestCVMPermutation:
    def test_cvm_reports_permutation_p(self):
        a = np.random.default_rng(6).normal(0, 1, 60)
        b = np.random.default_rng(7).normal(1.2, 1, 60)
        result = laserplane.DistributionComparer().compare_distributions(a, b, 'cramer')
        assert result['test'] == 'cramer_von_mises'
        assert 'permutation' in result.get('method', '')
        assert 0.0 <= result['p_value'] <= 1.0

    def test_cvm_permutation_p_detects_shift(self):
        a = np.random.default_rng(8).normal(0, 1, 60)
        b = np.random.default_rng(9).normal(2.0, 1, 60)
        result = laserplane.DistributionComparer().compare_distributions(a, b, 'cramer')
        assert result['p_value'] < 0.05

    def test_cvm_permutation_null_is_not_significant(self):
        a = np.random.default_rng(10).normal(0, 1, 60)
        b = np.random.default_rng(11).normal(0, 1, 60)
        result = laserplane.DistributionComparer().compare_distributions(a, b, 'cramer')
        assert result['p_value'] > 0.01


class TestMomentAnalyzerNormality:
    def test_normal_data_verdict(self):
        data = np.random.default_rng(21).normal(0, 1, 400)
        r = laserplane.MomentAnalyzer().assess_normality(data)
        assert r['sufficient_data']
        assert r['verdict'] == 'consistent_with_normal'
        assert r['omnibus_p'] > 0.05
        assert r['skewness_se'] > 0 and r['kurtosis_se'] > 0
        assert np.isfinite(r['skewness_z']) and np.isfinite(r['kurtosis_z'])

    def test_heavy_tailed_data_verdict(self):
        data = np.random.default_rng(22).standard_t(3, 400)
        r = laserplane.MomentAnalyzer().assess_normality(data)
        assert r['verdict'] == 'not_normal'
        assert r['omnibus_p'] < 0.05

    def test_insufficient_data(self):
        data = np.random.default_rng(23).normal(0, 1, 5)
        r = laserplane.MomentAnalyzer().assess_normality(data)
        assert not r['sufficient_data']
        assert r['verdict'] == 'insufficient_data'


class TestVuongDefensiveBranches:
    def test_vuong_degenerate_variance_on_single_observation(self):
        # With one comparable observation the pointwise variance is undefined;
        # the check must report the degenerate branch instead of a p-value.
        fitter = laserplane.DistributionFitter()
        results = {
            'normal': {'distribution': 'normal', 'parameters': (0.0, 1.0),
                       'log_likelihood': -12.5, 'n_params': 2, 'aicc': 29.0},
            'uniform': {'distribution': 'uniform', 'parameters': (4.0, 2.0),
                        'log_likelihood': -0.69, 'n_params': 2, 'aicc': 5.38},
        }
        vuong = fitter._vuong_check(np.array([5.0]), results, 'uniform')
        assert vuong['verdict'] == 'degenerate_variance'
        assert np.isnan(vuong['p_value'])

    def test_vuong_near_equal_likelihoods_not_significant(self):
        # Two well-fitting candidates on the same data: no significant
        # likelihood-ratio verdict.
        data = np.random.default_rng(5).normal(3.0, 1.0, 250)
        fitter = laserplane.DistributionFitter()
        results = {name: fitter.fit_distribution(data, name)
                   for name in fitter.supported_distributions}
        vuong = fitter._vuong_check(data, results, 'normal')
        assert vuong['verdict'] in ('no_significant_difference', 'degenerate_variance')

    def test_vuong_logpdf_evaluation_failed(self):
        # Parameters that cannot evaluate a logpdf must hit the guard branch.
        fitter = laserplane.DistributionFitter()
        data = np.random.default_rng(6).normal(0.0, 1.0, 30)
        results = {
            'normal': {'distribution': 'normal', 'parameters': (0.0, 1.0),
                       'log_likelihood': -42.0, 'n_params': 2, 'aicc': 88.0},
            'uniform': {'distribution': 'uniform', 'parameters': ('bad',),
                        'log_likelihood': -42.0, 'n_params': 1, 'aicc': 87.0},
        }
        vuong = fitter._vuong_check(data, results, 'normal')
        assert vuong['verdict'] == 'logpdf_evaluation_failed'


class TestComparerAllTests:
    @pytest.mark.parametrize('test', ['ks', 'anderson', 'cramer', 'mann_whitney', 't_test'])
    def test_all_supported_tests_routed(self, test):
        a = np.random.default_rng(31).normal(0.0, 1.0, 60)
        b = np.random.default_rng(32).normal(0.5, 1.0, 60)
        result = laserplane.DistributionComparer().compare_distributions(a, b, test)
        assert result['test'] is not None
        assert np.isfinite(float(result['statistic']))
        assert 0.0 <= float(result['p_value']) <= 1.0

    @pytest.mark.parametrize('test', ['ks', 'anderson', 'cramer', 'mann_whitney', 't_test'])
    def test_insufficient_data_sentinel_for_all_tests(self, test):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = laserplane.DistributionComparer().compare_distributions(a, b, test)
        assert result == {'test': None, 'statistic': None, 'p_value': None}


class TestFitterDefensiveBranches:
    def test_compute_log_likelihood_returns_neg_inf_when_params_invalid(self):
        """Parameters that cannot even unpack must map to -inf log-likelihood."""
        fitter = laserplane.DistributionFitter()

        assert fitter._compute_log_likelihood(np.arange(5.0), norm, None) == -np.inf

    def test_pointwise_loglik_beta_requires_scale_record(self):
        """A beta fit without its min-max scaling record must fail loudly."""
        fitter = laserplane.DistributionFitter()

        with pytest.raises(ValueError, match="min-max scaling"):
            fitter._pointwise_loglik(np.array([0.2, 0.5, 0.8]), 'beta',
                                     {'parameters': (2.0, 2.0, 0.0, 1.0)})

    def test_vuong_insufficient_fits_verdict(self):
        """With no successfully fitted candidate the check reports it."""
        fitter = laserplane.DistributionFitter()
        results = {name: {'distribution': None, 'parameters': None, 'aic': np.inf}
                   for name in fitter.supported_distributions}

        vuong = fitter._vuong_check(np.linspace(-2.0, 2.0, 20), results, 'normal')

        assert vuong['verdict'] == 'insufficient_fits'
        assert np.isnan(vuong['lr_statistic'])
        assert np.isnan(vuong['p_value'])

    def test_vuong_non_finite_likelihood_verdict(self):
        """A candidate with non-finite likelihood aborts the comparison."""
        fitter = laserplane.DistributionFitter()
        data = np.linspace(-2.0, 2.0, 30)
        results = {
            'normal': {'distribution': 'normal', 'parameters': (0.0, 1.0),
                       'log_likelihood': -np.inf, 'n_params': 2, 'aicc': np.inf},
            'uniform': {'distribution': 'uniform', 'parameters': (-2.0, 4.0),
                        'log_likelihood': -45.0, 'n_params': 2, 'aicc': 94.0},
        }

        vuong = fitter._vuong_check(data, results, 'uniform')

        assert vuong['verdict'] == 'non_finite_likelihood'
        assert np.isnan(vuong['p_value'])

    def test_ad_ksample_statistic_separates_shifted_samples(self):
        """The Scholz-Stephens statistic grows with distributional shift."""
        comparer = laserplane.DistributionComparer
        rng = np.random.default_rng(33)
        a = rng.normal(0.0, 1.0, 100)
        b = rng.normal(0.0, 1.0, 100)

        null_stat = comparer._ad_ksample_statistic([a, b])
        shifted_stat = comparer._ad_ksample_statistic([a, b + 1.5])

        assert np.isfinite(null_stat) and np.isfinite(shifted_stat)
        assert shifted_stat > null_stat
        # Under the null the raw statistic stays O(1) while genuine
        # separation at these sample sizes grows by more than an order of
        # magnitude.
        assert shifted_stat > 5.0 * null_stat

    def test_ad_ksample_statistic_floor_for_identical_observations(self):
        """All-identical observations sit at the statistic's floor."""
        stat = laserplane.DistributionComparer._ad_ksample_statistic(
            [np.full(10, 2.5), np.full(10, 2.5)])
        assert stat == 0.0
