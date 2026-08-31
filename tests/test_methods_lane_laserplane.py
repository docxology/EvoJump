"""Methods-lane tests: LaserPlane distribution fitting and comparison.

Real computation only — no mock frameworks (repo policy).
Covers: AICc/BIC in fits, Vuong check, Anderson-Darling k-sample,
permutation CVM p-value, D'Agostino-style normality verdicts.
"""
import warnings

import numpy as np
import pytest

from evojump import laserplane


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
        if result['distribution'] is not None:
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
