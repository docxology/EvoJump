"""
LaserPlane Analyzer: Cross-Sectional Analysis Algorithms

This module implements cross-sectional analysis algorithms for phenotypic distribution
characterization at specific developmental timepoints. Features include distribution fitting,
moment estimation, quantile analysis, and comparative distribution testing across different
developmental stages or genetic backgrounds.

Classes:
    LaserPlaneAnalyzer: Main analyzer for cross-sectional distributions
    DistributionFitter: Fits statistical distributions to cross-sectional data
    DistributionComparer: Compares distributions across conditions
    MomentAnalyzer: Analyzes moments and other distribution characteristics

Examples:
    >>> # Create analyzer
    >>> analyzer = LaserPlaneAnalyzer(jump_rope_model)
    >>> # Analyze distribution at specific time
    >>> results = analyzer.analyze_cross_section(time_point=10.0)
    >>> # Compare distributions
    >>> comparison = analyzer.compare_distributions(time_point=10.0, condition_data={'treatment': data})
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, gamma, beta, uniform, kstest
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import inspect
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionResult:
    """Container for cross-sectional analysis results."""
    time_point: float
    data: np.ndarray
    distribution_fit: Dict[str, Any]
    moments: Dict[str, float]
    quantiles: Dict[str, float]
    goodness_of_fit: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class DistributionComparison:
    """Container for distribution comparison results.

    ``distribution2_name`` lists every condition compared against the
    reference. ``test_statistics``, ``p_values`` and ``effect_sizes`` are
    keyed by condition name; conditions with insufficient data are omitted
    from the statistics dictionaries.
    """
    time_point: float
    distribution1_name: str
    distribution2_name: List[str]
    test_statistics: Dict[str, float]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    significant_differences: List[str]


class DistributionFitter:
    """Fits statistical distributions to cross-sectional data."""

    def __init__(self):
        """Initialize distribution fitter."""
        self.supported_distributions = {
            'normal': norm,
            'lognormal': lognorm,
            'gamma': gamma,
            'beta': beta,
            'uniform': uniform
        }

    def fit_distribution(self,
                        data: np.ndarray,
                        distribution: str = 'auto') -> Dict[str, Any]:
        """
        Fit statistical distribution to data.

        Parameters:
            data: Cross-sectional data
            distribution: Distribution to fit ('auto' for automatic selection)

        Returns:
            Dictionary with fit results
        """
        if len(data) < 4:
            warnings.warn("Insufficient data for distribution fitting")
            return {'distribution': None, 'parameters': None, 'aic': np.inf}

        data = data[~np.isnan(data)]  # Remove NaN values

        if distribution == 'auto':
            distribution = self._select_best_distribution(data)

        if distribution not in self.supported_distributions:
            raise ValueError(f"Unsupported distribution: {distribution}")

        dist_class = self.supported_distributions[distribution]

        try:
            # The information criteria are always computed on the exact data
            # the parameters were estimated on (``fit_data``); evaluating the
            # likelihood on a different array (e.g. the raw data for a fit
            # made on a positive subset or a rescaled copy) yields invalid,
            # often non-finite, log-likelihoods.
            fit_data = data
            fit_meta = {}
            if distribution == 'normal':
                params = dist_class.fit(data)
            elif distribution == 'lognormal':
                # Positive support: fit and evaluate on the positive subset
                data_pos = data[data > 0]
                if len(data_pos) < 4:
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                params = dist_class.fit(data_pos, floc=0)
                fit_data = data_pos
            elif distribution == 'gamma':
                # Positive support: fit and evaluate on the positive subset
                data_pos = data[data > 0]
                if len(data_pos) < 4:
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                params = dist_class.fit(data_pos, floc=0)
                fit_data = data_pos
            elif distribution == 'beta':
                # The beta density is defined on the open interval (0, 1), so
                # the data is min-max scaled; the change-of-variables Jacobian
                # (-n log(range)) keeps the likelihood comparable to models
                # fitted on the original scale.
                if np.min(data) == np.max(data):
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                scale_min = float(np.min(data))
                scale_max = float(np.max(data))
                data_scaled = (data - scale_min) / (scale_max - scale_min)
                # Min-max scaling puts the sample endpoints exactly at 0 and
                # 1, where the beta density is undefined; nudge them just
                # inside the open interval.
                eps = 32 * np.finfo(float).eps
                data_scaled = np.clip(data_scaled, eps, 1.0 - eps)
                params = dist_class.fit(data_scaled, floc=0, fscale=1)
                fit_data = data_scaled
                fit_meta = {'scale': (scale_min, scale_max)}
            elif distribution == 'uniform':
                if np.min(data) == np.max(data):
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                # scipy uniform.fit raises when loc and scale are both fixed;
                # the MLE is exactly (min, range), so set it directly.
                params = (np.min(data), np.max(data) - np.min(data))

            # Calculate information criteria for model comparison
            n_fit = len(fit_data)
            log_likelihood = self._compute_log_likelihood(fit_data, dist_class, params)
            if distribution == 'beta':
                # Change-of-variables Jacobian: the likelihood stored here is
                # the density of the transformed-beta model on the ORIGINAL
                # data, so it is comparable with the other candidates' AICc.
                log_likelihood = log_likelihood - n_fit * np.log(scale_max - scale_min)
            n_params = len(params)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n_fit) - 2 * log_likelihood
            aicc = np.inf if n_fit - n_params - 1 <= 0 else aic + 2 * n_params * (n_params + 1) / (n_fit - n_params - 1)

            result = {
                'distribution': distribution,
                'parameters': params,
                'aic': aic,
                'bic': bic,
                'aicc': aicc,
                'n_params': n_params,
                'log_likelihood': log_likelihood,
                'n_fit': n_fit,
                'fit_data': fit_data,
            }
            result.update(fit_meta)
            return result

        except Exception as e:
            logger.warning(f"Distribution fitting failed for {distribution}: {e}")
            return {'distribution': None, 'parameters': None, 'aic': np.inf}

    def _select_best_distribution(self, data: np.ndarray) -> str:
        """Select best fitting distribution using AICc."""
        best_distribution = 'normal'
        best_aicc = np.inf
        results = {}

        for dist_name in self.supported_distributions.keys():
            result = self.fit_distribution(data, dist_name)
            results[dist_name] = result

            if result.get('aicc', np.inf) < best_aicc:
                best_aicc = result['aicc']
                best_distribution = dist_name

        logger.info(f"Selected best distribution: {best_distribution} (AICc: {best_aicc})")
        self._last_selection = self._vuong_check(data, results, best_distribution)
        return best_distribution

    def _vuong_check(self, data: np.ndarray, results: Dict[str, Any], best_distribution: str) -> Dict[str, Any]:
        """Vuong-style likelihood-ratio check between the top two candidates.

        For non-nested models the LR statistic (with an Akaike-style
        small-sample adjustment) is referred to a standard normal, using the
        pointwise variance of the log-likelihood difference. Reports a
        p-value for the null that both models are equally close to the true
        data-generating process.
        """
        fitted = {k: v for k, v in results.items() if v.get('distribution') is not None}
        if len(fitted) < 2:
            return {'comparison': None, 'lr_statistic': np.nan,
                    'p_value': np.nan, 'verdict': 'insufficient_fits'}

        ranked = sorted(fitted.items(), key=lambda kv: kv[1]['aicc'])
        (name1, fit1), (name2, fit2) = ranked[0], ranked[1]

        if fit1['log_likelihood'] <= -np.inf or fit2['log_likelihood'] <= -np.inf:
            return {'comparison': None, 'lr_statistic': np.nan,
                    'p_value': np.nan, 'verdict': 'non_finite_likelihood'}

        try:
            ll1 = self._pointwise_loglik(data, name1, fit1)
            ll2 = self._pointwise_loglik(data, name2, fit2)
        except Exception:
            return {'comparison': None, 'lr_statistic': np.nan,
                    'p_value': np.nan, 'verdict': 'logpdf_evaluation_failed'}

        # Models with restricted support (lognormal, gamma, beta on its
        # min-max transform) assign -inf to observations outside that support;
        # compare the candidates on the observations where both are finite.
        common = np.isfinite(ll1) & np.isfinite(ll2)
        if not np.any(common):
            return {'comparison': None, 'lr_statistic': np.nan,
                    'p_value': np.nan, 'verdict': 'non_finite_likelihood'}
        ll1 = ll1[common]
        ll2 = ll2[common]
        n = len(ll1)

        lr_raw = float(np.sum(ll1 - ll2))
        # Akaike-style small-sample adjustment for non-nested comparison
        lr = lr_raw - (fit1['n_params'] - fit2['n_params'])
        diff = ll1 - ll2
        omega2 = float(np.var(diff, ddof=1)) if n > 1 else 0.0

        if omega2 <= 0 or not np.isfinite(omega2):
            statistic, p_value = np.nan, np.nan
            verdict = 'degenerate_variance'
        else:
            statistic = lr / (np.sqrt(n) * np.sqrt(omega2))
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))
            verdict = 'best_model_preferred' if p_value < 0.05 and lr > 0 else (
                'second_model_preferred' if p_value < 0.05 else 'no_significant_difference')

        return {
            'comparison': (name1, name2),
            'lr_statistic': statistic,
            'p_value': p_value,
            'verdict': verdict
        }

    def _pointwise_loglik(self, data: np.ndarray, name: str, fit: Dict[str, Any]) -> np.ndarray:
        """Pointwise log-likelihood of a fitted model on the original data.

        Models fitted on a restricted or rescaled copy of the data are mapped
        back to the original observations: positive-support models
        (lognormal, gamma) assign -inf to non-positive points, and the beta
        model fitted on min-max-scaled data includes the change-of-variables
        Jacobian so its density is comparable on the original scale.
        """
        dist = self.supported_distributions[name]
        params = fit['parameters']
        data = np.asarray(data, dtype=float)
        if name == 'beta':
            scale = fit.get('scale')
            if scale is None:
                raise ValueError("beta fit is missing its min-max scaling record")
            lo, hi = scale
            scaled = (data - lo) / (hi - lo)
            with np.errstate(divide='ignore', invalid='ignore'):
                ll = dist.logpdf(scaled, *params) - np.log(hi - lo)
            return np.where((scaled > 0.0) & (scaled < 1.0), ll, -np.inf)
        if name in ('lognormal', 'gamma'):
            with np.errstate(divide='ignore', invalid='ignore'):
                ll = dist.logpdf(np.where(data > 0, data, 1.0), *params)
            return np.where(data > 0, ll, -np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.asarray(dist.logpdf(data, *params), dtype=float)

    def _compute_log_likelihood(self,
                              data: np.ndarray,
                              dist_class,
                              params) -> float:
        """Compute log-likelihood for fitted distribution."""
        try:
            log_likelihood = np.sum(dist_class.logpdf(data, *params))
            return log_likelihood
        except:
            return -np.inf


class DistributionComparer:
    """Compares distributions across different conditions."""

    def __init__(self):
        """Initialize distribution comparer."""
        self.supported_tests = {
            'ks': self._kolmogorov_smirnov_test,
            'anderson': self._anderson_darling_test,
            'cramer': self._cramer_von_mises_test,
            'mann_whitney': self._mann_whitney_test,
            't_test': self._t_test
        }

    def compare_distributions(self,
                            data1: np.ndarray,
                            data2: np.ndarray,
                            test: str = 'auto',
                            rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Compare two distributions using statistical tests.

        Parameters:
            data1: First dataset
            data2: Second dataset
            test: Statistical test to use
            rng: Optional NumPy generator seeding permutation p-values

        Returns:
            Dictionary with test results
        """
        if len(data1) < 4 or len(data2) < 4:
            warnings.warn("Insufficient data for distribution comparison")
            return {'test': None, 'statistic': None, 'p_value': None}

        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]

        if test == 'auto':
            test = self._select_comparison_test(data1, data2)

        if test not in self.supported_tests:
            raise ValueError(f"Unsupported test: {test}")

        test_func = self.supported_tests[test]
        if 'rng' in inspect.signature(test_func).parameters:
            result = test_func(data1, data2, rng=rng)
        else:
            result = test_func(data1, data2)

        return result

    def _select_comparison_test(self, data1: np.ndarray, data2: np.ndarray) -> str:
        """Return the comparison test used for ``test='auto'``.

        Auto-selection currently always uses the two-sample
        Kolmogorov-Smirnov test.
        """
        return 'ks'

    def _kolmogorov_smirnov_test(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test."""
        statistic, p_value = kstest(data1, data2)
        return {
            'test': 'kolmogorov_smirnov',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def _anderson_darling_test(self, data1: np.ndarray, data2: np.ndarray,
                               rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Scholz-Stephens Anderson-Darling k-sample test.

        Uses scipy's anderson_ksamp (midrank variant) asymptotic p-value;
        falls back to a permutation p-value when the asymptotic one is
        unavailable or out of range.
        """
        samples = [data1, data2]
        try:
            result = stats.anderson_ksamp(samples, variant='midrank')
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            method_note = 'anderson_ksamp_asymptotic'
        except (ValueError, RuntimeError):
            statistic = self._ad_ksample_statistic(samples)
            p_value = self._permutation_p_value(samples, self._ad_ksample_statistic,
                                                rng=rng)
            method_note = 'anderson_ksamp_permutation'

        return {
            'test': 'anderson_darling',
            'statistic': statistic,
            'p_value': p_value,
            'method': method_note,
            'significant': p_value < 0.05
        }

    @staticmethod
    def _ad_ksample_statistic(samples: List[np.ndarray]) -> float:
        """Anderson-Darling k-sample statistic (Scholz-Stephens eq. 7, midrank).

        Matches the statistic computed by ``scipy.stats.anderson_ksamp`` for
        the midrank variant, so the permutation fallback (taken when
        anderson_ksamp itself refuses degenerate input) tests the same
        quantity as the asymptotic path. It grows with distributional
        separation and is minimal when all samples share one distribution.
        """
        k = len(samples)
        Z = np.sort(np.concatenate(samples))
        N = Z.size
        Zstar = np.unique(Z)
        if Zstar.size < 2:
            # Every observation identical: no label split is distinguishable
            # and the statistic attains its floor.
            return 0.0
        Z_left = Z.searchsorted(Zstar, 'left')
        lj = 1.0 if N == Zstar.size else Z.searchsorted(Zstar, 'right') - Z_left
        Bj = Z_left + lj / 2.0
        A2akN = 0.0
        for sample in samples:
            s = np.sort(sample)
            s_right = s.searchsorted(Zstar, side='right').astype(float)
            fij = s_right - s.searchsorted(Zstar, side='left')
            Mij = s_right - fij / 2.0
            inner = lj / float(N) * (N * Mij - Bj * len(s)) ** 2 \
                / (Bj * (N - Bj) - N * lj / 4.0)
            A2akN += inner.sum() / len(s)
        return float(A2akN * (N - 1.0) / N)

    @staticmethod
    def _permutation_p_value(samples: List[np.ndarray], statistic_fn,
                             n_permutations: int = 2000,
                             rng: Optional[np.random.Generator] = None) -> float:
        """Permutation p-value: shuffle group labels, recompute the statistic."""
        if rng is None:
            rng = np.random.default_rng(0)
        observed = statistic_fn(samples)
        pooled = np.concatenate(samples)
        sizes = [len(s) for s in samples]
        count_ge = 0
        for _ in range(n_permutations):
            permuted = rng.permutation(pooled)
            parts = []
            start = 0
            for size in sizes:
                parts.append(permuted[start:start + size])
                start += size
            if statistic_fn(parts) >= observed - 1e-12:
                count_ge += 1
        return float((count_ge + 1) / (n_permutations + 1))

    def _cramer_von_mises_test(self, data1: np.ndarray, data2: np.ndarray,
                               rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Perform Cramer-von Mises test with a permutation p-value.

        scipy has no analytic two-sample CVM p-value, so an empirical
        p-value over 2000 label shuffles is reported (honest and cheap).
        """
        statistic = self._cvm_statistic([data1, data2])
        p_value = self._permutation_p_value([data1, data2], self._cvm_statistic, rng=rng)

        return {
            'test': 'cramer_von_mises',
            'statistic': statistic,
            'p_value': p_value,
            'method': '2000-permutation empirical',
            'significant': p_value < 0.05
        }

    @staticmethod
    def _cvm_statistic(samples: List[np.ndarray]) -> float:
        """Two-sample Cramér-von Mises statistic (CDF-difference form)."""
        data1, data2 = samples[0], samples[1]
        n1, n2 = len(data1), len(data2)
        n = n1 + n2
        combined = np.sort(np.concatenate([data1, data2]))
        f1 = np.searchsorted(np.sort(data1), combined, side='right') / n1
        f2 = np.searchsorted(np.sort(data2), combined, side='right') / n2
        return float((n1 * n2 / n**2) * np.sum((f1 - f2) ** 2) / n)

    def _mann_whitney_test(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform Mann-Whitney U test."""
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        return {
            'test': 'mann_whitney',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def _t_test(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform t-test."""
        statistic, p_value = stats.ttest_ind(data1, data2)
        return {
            'test': 't_test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


class MomentAnalyzer:
    """Analyzes moments and distribution characteristics."""

    def __init__(self):
        """Initialize moment analyzer."""
        pass

    def compute_moments(self, data: np.ndarray) -> Dict[str, float]:
        """Compute statistical moments of the data."""
        data = data[~np.isnan(data)]

        if len(data) < 1:
            return {
                'mean': np.nan,
                'variance': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan
            }

        moments = {
            'mean': np.mean(data),
            'variance': np.var(data, ddof=1),
            'std': np.std(data, ddof=1),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'median': np.median(data),
            'mode': self._estimate_mode(data)
        }

        return moments

    def assess_normality(self, data: np.ndarray,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """D'Agostino-style normality assessment from skew/kurtosis z-scores.

        Reports skewness and excess kurtosis with classical large-sample
        standard errors, the D'Agostino (1970) transformed skewness z, an
        Anscombe-Glynn style kurtosis z, and the D'Agostino-Pearson K^2
        omnibus verdict (chi-square, 2 df).
        """
        data = data[~np.isnan(data)]
        n = len(data)
        if n < 8:
            return {'n': n, 'sufficient_data': False,
                    'verdict': 'insufficient_data'}

        skew = float(stats.skew(data, bias=False))
        kurt = float(stats.kurtosis(data, bias=False))  # excess kurtosis

        se_skew = float(np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
        se_kurt = float(np.sqrt(24.0 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))))

        # D'Agostino (1970) transform of skewness to near-normality
        b = 3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
        w2 = -1.0 + np.sqrt(2.0 * (b - 1.0))
        delta = 1.0 / np.sqrt(0.5 * np.log(w2))
        y = skew * np.sqrt((w2 - 1.0) * (n + 1) * (n + 3) / (12.0 * (n - 2)))
        z_skew = float(delta * np.log(y + np.sqrt(y ** 2 + 1.0))) if w2 > 1 and np.isfinite(y) else skew / se_skew

        z_kurt = kurt / se_kurt

        p_skew = 2.0 * (1.0 - stats.norm.cdf(abs(z_skew)))
        p_kurt = 2.0 * (1.0 - stats.norm.cdf(abs(z_kurt)))

        k2 = z_skew ** 2 + z_kurt ** 2
        p_omnibus = float(stats.chi2.sf(k2, df=2))

        return {
            'n': n,
            'sufficient_data': True,
            'skewness': skew,
            'skewness_se': se_skew,
            'skewness_z': z_skew,
            'skewness_p': p_skew,
            'excess_kurtosis': kurt,
            'kurtosis_se': se_kurt,
            'kurtosis_z': z_kurt,
            'kurtosis_p': p_kurt,
            'omnibus_k2': k2,
            'omnibus_p': p_omnibus,
            'verdict': 'not_normal' if p_omnibus < alpha else 'consistent_with_normal',
            'alpha': alpha
        }

    def compute_quantiles(self, data: np.ndarray, quantiles: List[float] = None) -> Dict[str, float]:
        """Compute quantiles of the data."""
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        data = data[~np.isnan(data)]

        if len(data) < 1:
            return {f'q{q:.2f}': np.nan for q in quantiles}

        quantile_values = np.quantile(data, quantiles)
        quantile_dict = {f'q{q:.2f}': val for q, val in zip(quantiles, quantile_values)}

        return quantile_dict

    def compute_confidence_intervals(self,
                                   data: np.ndarray,
                                   confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for distribution parameters.

        ``median_ci`` is the exact distribution-free order-statistic
        confidence interval for the median (binomial coverage over the
        order statistics), not a central range of the data itself.
        """
        data = data[~np.isnan(data)]

        if len(data) < 2:
            return {
                'mean_ci': (np.nan, np.nan),
                'median_ci': (np.nan, np.nan),
                'std_ci': (np.nan, np.nan)
            }

        # Confidence interval for mean
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        mean_ci = (mean - t_value * std / np.sqrt(n), mean + t_value * std / np.sqrt(n))

        # Exact distribution-free confidence interval for the median: the
        # number of observations at or below the median is Binomial(n, 0.5),
        # so choose order-statistic ranks whose binomial coverage reaches the
        # requested confidence level.
        sorted_data = np.sort(data)
        alpha = 1.0 - confidence_level
        lower_rank = max(int(stats.binom.ppf(alpha / 2.0, n, 0.5)), 1)
        upper_rank = min(int(stats.binom.ppf(1.0 - alpha / 2.0, n, 0.5)) + 1, n)
        median_ci = (sorted_data[lower_rank - 1], sorted_data[upper_rank - 1])

        # Confidence interval for standard deviation
        chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, n - 1)
        chi2_upper = stats.chi2.ppf((1 + confidence_level) / 2, n - 1)
        std_ci = (
            std * np.sqrt((n - 1) / chi2_upper),
            std * np.sqrt((n - 1) / chi2_lower)
        )

        return {
            'mean_ci': mean_ci,
            'median_ci': median_ci,
            'std_ci': std_ci
        }

    def _estimate_mode(self, data: np.ndarray) -> float:
        """Estimate mode of the data using kernel density estimation."""
        try:
            # Simple histogram-based mode estimation
            hist, bin_edges = np.histogram(data, bins=30)
            mode_idx = np.argmax(hist)
            mode = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
            return mode
        except:
            return np.nan


class LaserPlaneAnalyzer:
    """Main analyzer for cross-sectional distributions."""

    def __init__(self, jump_rope_model):
        """Initialize analyzer with JumpRope model."""
        self.jump_rope = jump_rope_model
        self.fitter = DistributionFitter()
        self.comparer = DistributionComparer()
        self.moment_analyzer = MomentAnalyzer()

        logger.info("Initialized LaserPlane Analyzer")

    def analyze_cross_section(self,
                            time_point: float,
                            n_bootstrap: int = 1000,
                            rng: Optional[np.random.Generator] = None) -> CrossSectionResult:
        """
        Analyze cross-sectional distribution at specific time point.

        Parameters:
            time_point: Time point for analysis
            n_bootstrap: Number of bootstrap samples for confidence intervals
            rng: Optional NumPy generator for reproducible bootstrap intervals

        Returns:
            CrossSectionResult with analysis results
        """
        logger.info(f"Analyzing cross-section at time point {time_point}")

        # Get cross-sectional data
        time_idx = np.argmin(np.abs(self.jump_rope.time_points - time_point))
        actual_time = self.jump_rope.time_points[time_idx]

        cross_section_data = self.jump_rope.compute_cross_sections(time_idx)

        # Fit distribution
        distribution_fit = self.fitter.fit_distribution(cross_section_data)

        # Compute moments
        moments = self.moment_analyzer.compute_moments(cross_section_data)

        # Compute quantiles
        quantiles = self.moment_analyzer.compute_quantiles(cross_section_data)

        # Compute confidence intervals using bootstrap
        confidence_intervals = self._bootstrap_confidence_intervals(
            cross_section_data, n_bootstrap, rng=rng
        )

        # Assess goodness of fit
        goodness_of_fit = self._assess_goodness_of_fit(
            cross_section_data, distribution_fit
        )

        result = CrossSectionResult(
            time_point=actual_time,
            data=cross_section_data,
            distribution_fit=distribution_fit,
            moments=moments,
            quantiles=quantiles,
            goodness_of_fit=goodness_of_fit,
            confidence_intervals=confidence_intervals
        )

        logger.info(f"Cross-section analysis completed for time point {actual_time}")
        return result

    def compare_distributions(self,
                           time_point: float,
                           condition_data: Dict[str, np.ndarray],
                           test: str = 'auto',
                           rng: Optional[np.random.Generator] = None) -> DistributionComparison:
        """
        Compare distributions across different conditions at a time point.

        Parameters:
            time_point: Time point for comparison
            condition_data: Dictionary of condition names to data arrays
            test: Statistical test to use
            rng: Optional NumPy generator seeding permutation p-values

        Returns:
            DistributionComparison with comparison results
        """
        logger.info(f"Comparing distributions at time point {time_point}")

        # Get reference cross-section
        time_idx = np.argmin(np.abs(self.jump_rope.time_points - time_point))
        reference_data = np.asarray(self.jump_rope.compute_cross_sections(time_idx), dtype=float)
        reference_data = reference_data[~np.isnan(reference_data)]

        comparison_results = {}

        for condition_name, data in condition_data.items():
            comparison = self.comparer.compare_distributions(reference_data, data, test, rng=rng)
            comparison_results[condition_name] = comparison

        # Aggregate per-condition results into the comparison record
        test_statistics = {}
        p_values = {}
        effect_sizes = {}
        significant_differences = []

        for condition_name, comparison in comparison_results.items():
            if comparison.get('statistic') is not None:
                test_statistics[condition_name] = float(comparison['statistic'])
            p_value = comparison.get('p_value')
            if p_value is not None:
                p_values[condition_name] = float(p_value)
                if p_value < 0.05:
                    significant_differences.append(condition_name)
            condition_arr = np.asarray(condition_data[condition_name], dtype=float)
            condition_arr = condition_arr[~np.isnan(condition_arr)]
            d = self._cohens_d(reference_data, condition_arr)
            if d is not None:
                effect_sizes[condition_name] = d

        result = DistributionComparison(
            time_point=time_point,
            distribution1_name='reference',
            distribution2_name=list(condition_data.keys()),
            test_statistics=test_statistics,
            p_values=p_values,
            effect_sizes=effect_sizes,
            significant_differences=significant_differences
        )

        logger.info(f"Distribution comparison completed for time point {time_point}")
        return result

    @staticmethod
    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> Optional[float]:
        """Pooled-standard-deviation Cohen's d between two samples."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return None
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled == 0:
            return None
        return float((np.mean(group2) - np.mean(group1)) / pooled)

    def _bootstrap_confidence_intervals(self,
                                      data: np.ndarray,
                                      n_bootstrap: int,
                                      rng: Optional[np.random.Generator] = None) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals (seeded generator)."""
        if rng is None:
            rng = np.random.default_rng(0)
        if len(data) < 4:
            return {
                'mean_ci': (np.nan, np.nan),
                'median_ci': (np.nan, np.nan),
                'std_ci': (np.nan, np.nan)
            }

        bootstrap_means = []
        bootstrap_medians = []
        bootstrap_stds = []

        for _ in range(n_bootstrap):
            bootstrap_sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
            bootstrap_medians.append(np.median(bootstrap_sample))
            bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))

        # Compute confidence intervals
        mean_ci = (np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5))
        median_ci = (np.percentile(bootstrap_medians, 2.5), np.percentile(bootstrap_medians, 97.5))
        std_ci = (np.percentile(bootstrap_stds, 2.5), np.percentile(bootstrap_stds, 97.5))

        return {
            'mean_ci': mean_ci,
            'median_ci': median_ci,
            'std_ci': std_ci
        }

    def _assess_goodness_of_fit(self,
                              data: np.ndarray,
                              distribution_fit: Dict[str, Any]) -> Dict[str, float]:
        """Assess goodness of fit for the distribution."""
        if distribution_fit['distribution'] is None:
            return {'aic': np.inf, 'bic': np.inf, 'ks_statistic': np.nan, 'ks_p_value': np.nan}

        dist_name = distribution_fit['distribution']
        params = distribution_fit['parameters']
        dist_class = self.fitter.supported_distributions[dist_name]
        # The KS test is evaluated on the same data the parameters were
        # estimated on (the positive subset for lognormal/gamma, the min-max
        # scaled data for beta); the frozen-CDF form is used because the
        # name+args form is broken on scipy >= 1.15 for some distributions.
        ks_data = np.asarray(distribution_fit.get('fit_data', data), dtype=float)
        try:
            ks_statistic, ks_p_value = kstest(ks_data, dist_class(*params).cdf)
        except Exception:
            ks_statistic, ks_p_value = np.nan, np.nan

        # BIC calculation
        n_params = len(params)
        n_samples = len(ks_data)
        log_likelihood = distribution_fit.get('log_likelihood', 0)
        bic = n_params * np.log(n_samples) - 2 * log_likelihood

        return {
            'aic': distribution_fit['aic'],
            'bic': bic,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value
        }

    def generate_summary_report(self,
                              time_points: List[float],
                              output_file: Optional[Path] = None) -> str:
        """Generate summary report of cross-sectional analyses."""
        results = []

        for time_point in time_points:
            try:
                analysis_result = self.analyze_cross_section(time_point)

                result_summary = {
                    'time_point': analysis_result.time_point,
                    'n_samples': len(analysis_result.data),
                    'mean': analysis_result.moments['mean'],
                    'std': analysis_result.moments['std'],
                    'distribution': analysis_result.distribution_fit.get('distribution', 'unknown'),
                    'aic': analysis_result.goodness_of_fit['aic']
                }

                results.append(result_summary)

            except Exception as e:
                logger.warning(f"Failed to analyze time point {time_point}: {e}")
                continue

        if time_points and not results:
            raise RuntimeError(
                f"Cross-section analysis failed for all {len(time_points)} "
                "time points; no summary report can be generated "
                "(see logged warnings for the underlying errors)."
            )

        # Create summary DataFrame
        summary_df = pd.DataFrame(results)

        if output_file:
            summary_df.to_csv(output_file, index=False)
            logger.info(f"Summary report saved to {output_file}")

        return summary_df.to_string()

