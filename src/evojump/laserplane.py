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
    >>> comparison = analyzer.compare_distributions(time_point=10.0, condition='treatment')
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, gamma, beta, uniform, kstest, anderson
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
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
    """Container for distribution comparison results."""
    time_point: float
    distribution1_name: str
    distribution2_name: str
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
            # Fit distribution
            if distribution == 'normal':
                params = dist_class.fit(data)
            elif distribution == 'lognormal':
                # Ensure positive values for lognormal
                data_pos = data[data > 0]
                if len(data_pos) < 4:
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                params = dist_class.fit(data_pos, floc=0)
            elif distribution == 'gamma':
                # Ensure positive values for gamma
                data_pos = data[data > 0]
                if len(data_pos) < 4:
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                params = dist_class.fit(data_pos, floc=0)
            elif distribution == 'beta':
                # Scale data to [0,1] for beta distribution
                if np.min(data) == np.max(data):
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                data_scaled = (data - np.min(data)) / (np.max(data) - np.min(data))
                if np.any((data_scaled <= 0) | (data_scaled >= 1)):
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                params = dist_class.fit(data_scaled, floc=0, fscale=1)
            elif distribution == 'uniform':
                if np.min(data) == np.max(data):
                    return {'distribution': None, 'parameters': None, 'aic': np.inf}
                params = dist_class.fit(data, floc=np.min(data), fscale=np.max(data)-np.min(data))

            # Calculate AIC for model comparison
            log_likelihood = self._compute_log_likelihood(data, dist_class, params)
            n_params = len(params)
            aic = 2 * n_params - 2 * log_likelihood

            return {
                'distribution': distribution,
                'parameters': params,
                'aic': aic,
                'log_likelihood': log_likelihood
            }

        except Exception as e:
            logger.warning(f"Distribution fitting failed for {distribution}: {e}")
            return {'distribution': None, 'parameters': None, 'aic': np.inf}

    def _select_best_distribution(self, data: np.ndarray) -> str:
        """Select best fitting distribution using AIC."""
        best_distribution = 'normal'
        best_aic = np.inf
        results = {}

        for dist_name in self.supported_distributions.keys():
            result = self.fit_distribution(data, dist_name)
            results[dist_name] = result

            if result['aic'] < best_aic:
                best_aic = result['aic']
                best_distribution = dist_name

        logger.info(f"Selected best distribution: {best_distribution} (AIC: {best_aic})")
        return best_distribution

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
                            test: str = 'auto') -> Dict[str, Any]:
        """
        Compare two distributions using statistical tests.

        Parameters:
            data1: First dataset
            data2: Second dataset
            test: Statistical test to use

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
        result = test_func(data1, data2)

        return result

    def _select_comparison_test(self, data1: np.ndarray, data2: np.ndarray) -> str:
        """Select appropriate test based on data characteristics."""
        # For now, default to Kolmogorov-Smirnov test
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

    def _anderson_darling_test(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform Anderson-Darling test."""
        # Anderson-Darling test for k-samples (approximate)
        result1 = anderson(data1, dist='norm')
        result2 = anderson(data2, dist='norm')

        # Combine test statistics
        combined_stat = result1.statistic + result2.statistic
        # Approximate p-value (this is a rough approximation)
        p_value = 1 - norm.cdf(combined_stat)

        return {
            'test': 'anderson_darling',
            'statistic': combined_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def _cramer_von_mises_test(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform Cramer-von Mises test."""
        # Approximate implementation
        n1, n2 = len(data1), len(data2)
        combined = np.concatenate([data1, data2])
        ranks = stats.rankdata(combined)

        # Calculate test statistic
        statistic = (n1 * n2 / (n1 + n2)**2) * np.sum((ranks[:n1] - np.arange(1, n1+1))**2)

        # Approximate p-value
        p_value = 1 - norm.cdf(statistic)

        return {
            'test': 'cramer_von_mises',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

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
        """Compute confidence intervals for distribution parameters."""
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

        # Confidence interval for median (approximate)
        median = np.median(data)
        median_ci = (np.quantile(data, 0.025), np.quantile(data, 0.975))

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
                            n_bootstrap: int = 1000) -> CrossSectionResult:
        """
        Analyze cross-sectional distribution at specific time point.

        Parameters:
            time_point: Time point for analysis
            n_bootstrap: Number of bootstrap samples for confidence intervals

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
            cross_section_data, n_bootstrap
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
                           test: str = 'auto') -> DistributionComparison:
        """
        Compare distributions across different conditions at a time point.

        Parameters:
            time_point: Time point for comparison
            condition_data: Dictionary of condition names to data arrays
            test: Statistical test to use

        Returns:
            DistributionComparison with comparison results
        """
        logger.info(f"Comparing distributions at time point {time_point}")

        # Get reference cross-section
        time_idx = np.argmin(np.abs(self.jump_rope.time_points - time_point))
        reference_data = self.jump_rope.compute_cross_sections(time_idx)

        comparison_results = {}

        for condition_name, data in condition_data.items():
            comparison = self.comparer.compare_distributions(reference_data, data, test)
            comparison_results[condition_name] = comparison

        # Aggregate results
        all_tests = list(comparison_results.values())[0]['test'] if comparison_results else None
        test_statistics = {}
        p_values = {}
        significant_differences = []

        for condition_name, result in comparison_results.items():
            if result['p_value'] is not None and result['p_value'] < 0.05:
                significant_differences.append(condition_name)

        result = DistributionComparison(
            time_point=time_point,
            distribution1_name='reference',
            distribution2_name=list(condition_data.keys()),
            test_statistics=test_statistics,
            p_values=p_values,
            effect_sizes={},
            significant_differences=significant_differences
        )

        logger.info(f"Distribution comparison completed for time point {time_point}")
        return result

    def _bootstrap_confidence_intervals(self,
                                      data: np.ndarray,
                                      n_bootstrap: int) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals."""
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
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
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

        # Kolmogorov-Smirnov test
        try:
            ks_statistic, ks_p_value = kstest(data, dist_class.name, args=params)
        except:
            ks_statistic, ks_p_value = np.nan, np.nan

        # BIC calculation
        n_params = len(params)
        n_samples = len(data)
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

        # Create summary DataFrame
        summary_df = pd.DataFrame(results)

        if output_file:
            summary_df.to_csv(output_file, index=False)
            logger.info(f"Summary report saved to {output_file}")

        return summary_df.to_string()

