"""
Analytics Engine: Comprehensive Statistical Analysis

This module provides comprehensive statistical analysis framework supporting both classical
and modern analytical approaches. Includes time series analysis, multivariate statistics,
machine learning algorithms, predictive modeling, Bayesian analysis, network analysis,
causal inference, dimensionality reduction, survival analysis, spectral analysis,
nonlinear dynamics, information theory, robust statistics, and spatial analysis
capabilities adapted for developmental and evolutionary data.

Classes:
    AnalyticsEngine: Main analytics engine
    TimeSeriesAnalyzer: Time series analysis methods
    MultivariateAnalyzer: Multivariate statistical methods
    PredictiveModeler: Predictive modeling and machine learning
    ChangePointDetector: Detects changes in developmental trajectories
    BayesianAnalyzer: Bayesian inference methods
    NetworkAnalyzer: Graph theory and network analysis
    CausalInference: Causal relationship discovery
    DimensionalityReducer: Advanced dimensionality reduction methods
    SurvivalAnalyzer: Survival and timing analysis
    SpectralAnalyzer: Frequency domain analysis
    NonlinearDynamics: Chaos theory and attractor analysis
    InformationTheory: Entropy and mutual information analysis
    RobustStatistics: Outlier-resistant statistical methods
    SpatialAnalyzer: Spatial developmental pattern analysis

Examples:
    >>> # Create analytics engine
    >>> engine = AnalyticsEngine(data)
    >>> # Perform time series analysis
    >>> ts_results = engine.analyze_time_series()
    >>> # Run predictive modeling
    >>> predictions = engine.predictive_modeling(target='adult_phenotype')
    >>> # Perform Bayesian analysis
    >>> bayes_results = engine.bayesian_analysis()
    >>> # Analyze developmental networks
    >>> network_results = engine.network_analysis()
    >>> # Discover causal relationships
    >>> causal_results = engine.causal_inference()
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import scipy.linalg as linalg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import networkx as nx
from networkx.algorithms import community
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesResult:
    """Container for time series analysis results."""
    trend_analysis: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    change_points: List[Dict[str, Any]]
    forecasts: Dict[str, np.ndarray]
    model_fit: Dict[str, Any]


@dataclass
class PredictiveModelResult:
    """Container for predictive modeling results."""
    model_name: str
    predictions: np.ndarray
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    model_parameters: Dict[str, Any]


class TimeSeriesAnalyzer:
    """Time series analysis methods for developmental data."""

    def __init__(self, data: pd.DataFrame, time_column: str = 'time'):
        """Initialize time series analyzer."""
        self.data = data
        self.time_column = time_column
        self.phenotype_columns = [col for col in data.columns if col != time_column]

    def analyze_trends(self, method: str = 'linear') -> Dict[str, Any]:
        """
        Analyze trends in time series data.

        Parameters:
            method: Trend analysis method

        Returns:
            Dictionary with trend analysis results
        """
        results = {}

        for col in self.phenotype_columns:
            series_data = self.data[col].dropna()

            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(series_data):
                logger.warning(f"Skipping non-numeric column: {col}")
                continue

            if method == 'linear':
                # Linear regression
                time_numeric = np.arange(len(series_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_numeric, series_data.values
                )

                results[col] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_error': std_err,
                    'significant': p_value < 0.05
                }

            elif method == 'polynomial':
                # Polynomial fitting
                time_numeric = np.arange(len(series_data))
                coeffs = np.polyfit(time_numeric, series_data.values, 2)
                poly_fit = np.poly1d(coeffs)

                # Compute R-squared
                y_pred = poly_fit(time_numeric)
                ss_res = np.sum((series_data.values - y_pred)**2)
                ss_tot = np.sum((series_data.values - np.mean(series_data.values))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                results[col] = {
                    'coefficients': coeffs,
                    'polynomial_fit': poly_fit,
                    'r_squared': r_squared,
                    'degree': 2
                }

        return results

    def detect_seasonality(self, period: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect seasonality in time series data.

        Parameters:
            period: Expected period length

        Returns:
            Dictionary with seasonality analysis results
        """
        results = {}

        for col in self.phenotype_columns:
            series_data = self.data[col].dropna()

            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(series_data):
                logger.warning(f"Skipping non-numeric column in seasonality: {col}")
                continue

            # Resolve the period per column: a user-supplied period applies
            # to every column, but auto-detection must never reuse the
            # previous column's result.
            col_period = period
            if col_period is None:
                # Auto-detect period using autocorrelation
                autocorr = np.correlate(series_data, series_data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]

                # Find peaks in autocorrelation
                peaks, _ = find_peaks(autocorr, height=0.2)
                if len(peaks) > 0:
                    col_period = int(peaks[0])


            if col_period and col_period < len(series_data):
                # Perform seasonal decomposition
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomposition = seasonal_decompose(
                        series_data, model='additive', period=col_period
                    )

                    results[col] = {
                        'period': col_period,
                        'seasonal_strength': np.var(decomposition.seasonal) / np.var(series_data),
                        'trend_strength': np.var(decomposition.trend) / np.var(series_data),
                        'residual_strength': np.var(decomposition.resid) / np.var(series_data),
                        'seasonal_detected': True
                    }
                except Exception:
                    results[col] = {
                        'period': col_period,
                        'seasonal_detected': False,
                        'error': 'Decomposition failed'
                    }
            else:
                results[col] = {
                    'seasonal_detected': False,
                    'reason': 'Insufficient data or invalid period'
                }

        return results

    def detect_change_points(self, method: str = 'cusum') -> List[Dict[str, Any]]:
        """
        Detect change points in time series.

        Parameters:
            method: Change point detection method ('cusum' or 'variance')

        Returns:
            List of detected change points
        """
        if method == 'cusum':
            # Delegate to the shared CUSUM implementation so both entry
            # points report identical results.
            detector = ChangePointDetector(self.data, self.time_column)
            return detector._statistical_change_detection()
        elif method == 'variance':
            return self._variance_change_detection()
        else:
            raise ValueError(f"Unsupported change point method: {method}")

    def _variance_change_detection(self, alpha: float = 0.05,
                                   min_segment: int = 5) -> List[Dict[str, Any]]:
        """Variance-based change detection with a split-sample F-test gate.

        For every candidate split, compares the pooled variances of the two
        segments with a two-sided F-test (larger-variance numerator) and
        reports the most significant split at level `alpha`. The reported
        confidence is 1 - p_value, so the method stays quiet on constant-
        variance series instead of always firing (the old behavior appended
        the argmax window with a hardcoded confidence of 0.8).
        """
        change_points = []

        for col in self.phenotype_columns:
            series_data = self.data[col].dropna()

            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(series_data):
                continue

            if len(series_data) < 2 * min_segment:
                continue

            x = series_data.to_numpy(dtype=float)
            n = len(x)

            n_splits = (n - 2 * min_segment) + 1
            best = None  # (f_stat, p_value, split_idx)
            for split in range(min_segment, n - min_segment + 1):
                left = x[:split]
                right = x[split:]
                var_l = float(np.var(left, ddof=1))
                var_r = float(np.var(right, ddof=1))
                if var_l <= 0 or var_r <= 0:
                    continue
                if var_l >= var_r:
                    f_stat, df1, df2 = var_l / var_r, len(left) - 1, len(right) - 1
                else:
                    f_stat, df1, df2 = var_r / var_l, len(right) - 1, len(left) - 1
                # Two-sided: double the one-sided tail of the larger ratio,
                # then Bonferroni-correct for scanning all candidate splits
                # (without the correction, the minimum p over ~n splits
                # fires on homoscedastic noise).
                p_value = min(1.0, 2.0 * stats.f.sf(f_stat, df1, df2) * n_splits)
                if best is None or p_value < best[1]:
                    best = (f_stat, p_value, split)

            if best is None or best[1] >= alpha:
                continue

            f_stat, p_value, split = best
            time_val = self.data[self.time_column].iloc[split] if self.time_column in self.data.columns else split
            change_points.append({
                'variable': col,
                'time_index': split,
                'time_value': time_val,
                'variance_ratio': f_stat,
                'p_value': p_value,
                'confidence': 1.0 - p_value,
                'method': 'variance'
            })

        return change_points

    def forecast(self, forecast_steps: int = 10, method: str = 'arima') -> Dict[str, np.ndarray]:
        """
        Forecast future values.

        Parameters:
            forecast_steps: Number of steps to forecast
            method: Forecasting method

        Returns:
            Dictionary with forecasts for each variable
        """
        forecasts = {}

        for col in self.phenotype_columns:
            series_data = self.data[col].dropna()

            if len(series_data) < 10:
                forecasts[col] = np.full(forecast_steps, np.nan)
                continue

            try:
                if method == 'arima':
                    # ARIMA model
                    model = ARIMA(series_data, order=(1, 1, 1))
                    model_fit = model.fit()
                    forecast_result = model_fit.forecast(steps=forecast_steps)

                    forecasts[col] = forecast_result.values

                elif method == 'exponential_smoothing':
                    # Simple exponential smoothing
                    alpha = 0.3  # Smoothing parameter
                    forecast_values = [series_data.iloc[-1]]  # Start with last value

                    for _ in range(forecast_steps - 1):
                        next_val = alpha * series_data.iloc[-1] + (1 - alpha) * forecast_values[-1]
                        forecast_values.append(next_val)

                    forecasts[col] = np.array(forecast_values)

                else:
                    forecasts[col] = np.full(forecast_steps, series_data.iloc[-1])

            except Exception as e:
                logger.warning(f"Forecasting failed for {col}: {e}")
                forecasts[col] = np.full(forecast_steps, np.nan)

        return forecasts


class MultivariateAnalyzer:
    """Multivariate statistical methods for developmental data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize multivariate analyzer."""
        self.data = data
        self.n_samples = len(data)
        self.variables = data.columns.tolist()

    def principal_component_analysis(self, n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform principal component analysis.

        Parameters:
            n_components: Number of components to retain

        Returns:
            Dictionary with PCA results
        """
        if self.n_samples < 3:
            raise ValueError("Insufficient data for PCA")

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Perform PCA
        if n_components is None:
            n_components = min(self.n_samples, len(self.variables))

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)

        # Compute explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        return {
            'pca_components': pca_result,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'loadings': pca.components_,
            'eigenvalues': pca.explained_variance_,
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_
        }

    def canonical_correlation_analysis(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform canonical correlation analysis.

        Parameters:
            data1: First dataset
            data2: Second dataset

        Returns:
            Dictionary with CCA results
        """
        if len(data1) != len(data2):
            raise ValueError("Datasets must have same number of samples")

        # Standardize data
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        scaled_data1 = scaler1.fit_transform(data1)
        scaled_data2 = scaler2.fit_transform(data2)

        # Compute canonical correlations
        n_samples = scaled_data1.shape[0]
        n_vars1 = scaled_data1.shape[1]
        n_vars2 = scaled_data2.shape[1]

        # Compute covariance matrices
        cov_matrix = np.cov(np.hstack([scaled_data1, scaled_data2]).T)

        # Split covariance matrix
        cov11 = cov_matrix[:n_vars1, :n_vars1]
        cov12 = cov_matrix[:n_vars1, n_vars1:]
        cov21 = cov_matrix[n_vars1:, :n_vars1]
        cov22 = cov_matrix[n_vars1:, n_vars1:]

        # Compute canonical correlations
        try:
            # Canonical correlations are the singular values of
            # cov11^{-1/2} @ cov12 @ cov22^{-1/2}. Squaring them gives the
            # eigenvalues of the symmetric matrix
            #   M = cov11^{-1/2} cov12 cov22^{-1} cov21 cov11^{-1/2},
            # which is the correct generalized eigenproblem for CCA (the
            # cov22^{-1} factor is essential; without it the "eigenvalues"
            # are not squared canonical correlations). pinv-style guards
            # below tolerate rank-deficient blocks.
            inv_sqrt11 = self._inv_psd_sqrt(cov11)
            inv_sqrt22 = self._inv_psd_sqrt(cov22)
            B = inv_sqrt11 @ cov12 @ inv_sqrt22
            eigvals, eigvecs = linalg.eigh(B @ B.T)

            # Eigenvalues are squared canonical correlations; clip tiny
            # negative round-off and sort descending.
            eigvals = np.clip(eigvals, 0.0, 1.0)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            canonical_correlations = np.sqrt(eigvals)

            # X-side canonical coefficients: a_k = cov11^{-1/2} v_k
            canonical_variables_1 = inv_sqrt11 @ eigvecs
            # Y-side canonical coefficients: b_k = cov22^{-1/2} cov21 a_k / r_k
            canonical_variables_2 = inv_sqrt22 @ cov21 @ canonical_variables_1
            nonzero = canonical_correlations > 1e-12
            canonical_variables_2[:, nonzero] /= canonical_correlations[nonzero]

            return {
                'canonical_correlations': canonical_correlations,
                'canonical_variables_1': canonical_variables_1,
                'canonical_variables_2': canonical_variables_2,
                'eigenvalues': eigvals,
                'scaler1_mean': scaler1.mean_,
                'scaler1_scale': scaler1.scale_,
                'scaler2_mean': scaler2.mean_,
                'scaler2_scale': scaler2.scale_
            }

        except Exception as e:
            logger.warning(f"CCA failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def _inv_psd_sqrt(matrix: np.ndarray) -> np.ndarray:
        """Inverse symmetric square root of a PSD matrix (eigh-based).

        Directions with numerically zero eigenvalues map to zero, which is
        the Moore-Penrose behavior needed for singular covariance blocks.
        """
        sym = (matrix + matrix.T) / 2.0
        vals, vecs = linalg.eigh(sym)
        tol = np.finfo(float).eps * max(vals.max(), 1.0) * len(vals)
        inv_sqrt_vals = np.where(vals > tol, 1.0 / np.sqrt(np.where(vals > tol, vals, 1.0)), 0.0)
        return (vecs * inv_sqrt_vals) @ vecs.T


    def cluster_analysis(self, n_clusters: int = 3, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform cluster analysis.

        Parameters:
            n_clusters: Number of clusters
            method: Clustering method

        Returns:
            Dictionary with clustering results
        """
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.data)

            return {
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters
            }

        else:
            raise ValueError(f"Unsupported clustering method: {method}")


class PredictiveModeler:
    """Predictive modeling and machine learning for developmental data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize predictive modeler."""
        self.data = data
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'support_vector': SVR(),
            'neural_network': MLPRegressor(random_state=42, max_iter=1000)
        }

    def train_predictive_model(self,
                              target_variable: str,
                              feature_variables: List[str],
                              model_name: str = 'random_forest',
                              test_size: float = 0.2) -> PredictiveModelResult:
        """
        Train predictive model.

        Parameters:
            target_variable: Variable to predict
            feature_variables: Feature variables
            model_name: Name of model to use
            test_size: Fraction of data for testing

        Returns:
            PredictiveModelResult with model results
        """
        if target_variable not in self.data.columns:
            raise ValueError(f"Target variable {target_variable} not found in data")

        missing_features = [var for var in feature_variables if var not in self.data.columns]
        if missing_features:
            raise ValueError(f"Feature variables not found: {missing_features}")

        # Prepare data
        model_data = self.data.dropna(subset=[target_variable] + feature_variables)
        X = model_data[feature_variables]
        y = model_data[target_variable]

        if len(model_data) < 10:
            raise ValueError("Insufficient data for model training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        # Train model
        model = self.models[model_name]
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Compute metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_variables, model.feature_importances_))

        return PredictiveModelResult(
            model_name=model_name,
            predictions=y_pred_test,
            performance_metrics={
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            feature_importance=feature_importance,
            model_parameters=model.get_params()
        )

    def cross_validate_model(self,
                           target_variable: str,
                           feature_variables: List[str],
                           model_name: str = 'random_forest',
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.

        Parameters:
            target_variable: Variable to predict
            feature_variables: Feature variables
            model_name: Name of model to use
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        model_data = self.data.dropna(subset=[target_variable] + feature_variables)
        X = model_data[feature_variables]
        y = model_data[target_variable]

        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]

        # Perform cross-validation
        mse_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

        return {
            'mean_mse': -np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores)
        }


class ChangePointDetector:
    """Detects changes in developmental trajectories."""

    def __init__(self, data: pd.DataFrame, time_column: str = 'time'):
        """Initialize change point detector."""
        self.data = data
        self.time_column = time_column
        self.phenotype_columns = [col for col in data.columns if col != time_column]

    def detect_changes(self, method: str = 'statistical', **kwargs) -> List[Dict[str, Any]]:
        """
        Detect change points in developmental trajectories.

        Parameters:
            method: Detection method
            **kwargs: Additional parameters

        Returns:
            List of detected change points
        """
        if method == 'statistical':
            return self._statistical_change_detection(**kwargs)
        elif method == 'bayesian':
            return self._bayesian_change_detection(**kwargs)
        elif method == 'information':
            return self._information_criterion_change_detection(**kwargs)
        else:
            raise ValueError(f"Unsupported detection method: {method}")

    def _statistical_change_detection(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Statistical change point detection."""
        change_points = []

        for col in self.phenotype_columns:
            series_data = self.data[col].dropna()

            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(series_data):
                continue

            if len(series_data) < 10:
                continue

            # Compute differences
            differences = np.abs(np.diff(series_data))

            # Find significant changes
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)

            if std_diff > 0:
                z_scores = (differences - mean_diff) / std_diff
                significant_changes = np.where(z_scores > threshold)[0]

                for change_idx in significant_changes:
                    time_val = self.data[self.time_column].iloc[change_idx] if self.time_column in self.data.columns else change_idx
                    change_points.append({
                        'variable': col,
                        'time_index': change_idx,
                        'time_value': time_val,
                        'change_magnitude': differences[change_idx],
                        'z_score': z_scores[change_idx],
                        'method': 'statistical'
                    })

        return change_points

    def _bayesian_change_detection(self,
                                   expected_run_length: float = 25.0,
                                   hazard: Optional[float] = None,
                                   mu0: float = 0.0,
                                   kappa0: float = 1.0,
                                   alpha0: float = 1.0,
                                   beta0: float = 1.0,
                                   threshold: float = 0.5,
                                   **kwargs) -> List[Dict[str, Any]]:
        """Bayesian online change-point detection (Adams & MacKay 2007).

        Normal-conjugate BOCPD per phenotype column: Student-t predictive
        likelihood, hazard h = 1/expected_run_length (or explicit `hazard`),
        run-length posterior over indices. Change points are reported where
        the run-length posterior mass at zero (changepoint probability) is
        >= `threshold` and is a local maximum. The CUSUM path in
        _statistical_change_detection remains available and unchanged.
        """
        change_points = []
        for col in self.phenotype_columns:
            series = self.data[col].dropna()
            if not pd.api.types.is_numeric_dtype(series):
                continue
            x = series.to_numpy(dtype=float)
            if len(x) < 10:
                continue

            h = hazard if (hazard is not None and hazard > 0) else 1.0 / max(expected_run_length, 1.0)

            # Sufficient statistics for runs of length j: index j holds the
            # NIG posterior after j observations of the current segment.
            R = np.zeros(len(x) + 1)
            R[0] = 1.0
            mu_t = np.full(len(x) + 1, mu0)
            kappa_t = np.full(len(x) + 1, kappa0)
            alpha_t = np.full(len(x) + 1, alpha0)
            beta_t = np.full(len(x) + 1, beta0)

            cp_probs = np.zeros(len(x))
            for t_idx, obs in enumerate(x):
                # Predictive Student-t log-likelihood over active runs 0..t_idx
                df = 2.0 * alpha_t[:t_idx + 1]
                scale2 = beta_t[:t_idx + 1] * (kappa_t[:t_idx + 1] + 1.0) / (alpha_t[:t_idx + 1] * kappa_t[:t_idx + 1])
                z = (obs - mu_t[:t_idx + 1]) / np.sqrt(scale2)
                log_pred = stats.t.logpdf(z, df) - 0.5 * np.log(scale2)

                pred = np.exp(log_pred - log_pred.max())
                # Changepoint: new run starts from the prior, so its predictive
                # is the prior Student-t (NOT the grown runs' predictives).
                s2_prior = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0)
                z_prior = (obs - mu0) / np.sqrt(s2_prior)
                log_prior_pred = stats.t.logpdf(z_prior, 2.0 * alpha0) - 0.5 * np.log(s2_prior)
                pred_prior = np.exp(log_prior_pred - log_pred.max())
                cp_prob = float(h * np.sum(R[:t_idx + 1]) * pred_prior)
                growth = (1.0 - h) * R[:t_idx + 1] * pred

                denom = cp_prob + np.sum(growth)
                if denom <= 0 or not np.isfinite(denom):
                    R[:t_idx + 2] = 0.0
                    R[t_idx + 1] = 1.0
                else:
                    R[t_idx + 1] = cp_prob / denom
                    R[:t_idx + 1] = growth / denom
                # Normalized posterior mass at run length 0 = P(changepoint now)
                cp_probs[t_idx] = float(R[t_idx + 1])

                # Shift registers: grown run j+1 = old run j absorbing obs
                # (NIG conjugate update); fresh run 0 stays at the prior.
                kappa_old = kappa_t[:t_idx + 1].copy()
                mu_old = mu_t[:t_idx + 1].copy()
                alpha_old = alpha_t[:t_idx + 1].copy()
                beta_old = beta_t[:t_idx + 1].copy()
                kappa_t[1:t_idx + 2] = kappa_old + 1.0
                mu_t[1:t_idx + 2] = (kappa_old * mu_old + obs) / (kappa_old + 1.0)
                alpha_t[1:t_idx + 2] = alpha_old + 0.5
                beta_t[1:t_idx + 2] = (beta_old + 0.5 * kappa_old * (obs - mu_old) ** 2 / (kappa_old + 1.0))
                mu_t[0] = mu0
                kappa_t[0] = kappa0
                alpha_t[0] = alpha0
                beta_t[0] = beta0

            # Report local maxima of changepoint probability above threshold
            for idx in range(1, len(cp_probs) - 1):
                if (cp_probs[idx] >= threshold
                        and cp_probs[idx] >= cp_probs[idx - 1]
                        and cp_probs[idx] >= cp_probs[idx + 1]):
                    time_val = (self.data[self.time_column].iloc[idx]
                                if self.time_column in self.data.columns else idx)
                    change_points.append({
                        'variable': col,
                        'time_index': idx,
                        'time_value': time_val,
                        'changepoint_probability': float(cp_probs[idx]),
                        'method': 'bocpd'
                    })

        return change_points

    def _information_criterion_change_detection(self,
                                                min_segment: int = 5,
                                                **kwargs) -> List[Dict[str, Any]]:
        """BIC-based mean-shift change detection.

        For each phenotype column, compares the BIC of a single Gaussian
        against the best two-segment Gaussian (independent mean and variance
        per segment) and reports the best split whenever the two-segment
        model wins. Segments shorter than `min_segment` are never proposed.
        """
        change_points = []
        for col in self.phenotype_columns:
            series = self.data[col].dropna()
            if not pd.api.types.is_numeric_dtype(series):
                continue
            x = series.to_numpy(dtype=float)
            n = len(x)
            if n < 2 * min_segment + 2:
                continue

            def _gaussian_bic(seg: np.ndarray) -> float:
                m = len(seg)
                if m < 2:
                    return np.inf
                mu = float(np.mean(seg))
                var = float(np.var(seg)) + 1e-12
                loglik = -0.5 * m * (np.log(2.0 * np.pi * var) + 1.0)
                return 2.0 * np.log(m) - 2.0 * loglik

            bic_single = _gaussian_bic(x)
            best_k, best_bic = None, bic_single
            for k in range(min_segment, n - min_segment + 1):
                bic_split = _gaussian_bic(x[:k]) + _gaussian_bic(x[k:])
                if bic_split < best_bic:
                    best_bic, best_k = bic_split, k
            if best_k is None:
                continue

            time_val = (self.data[self.time_column].iloc[best_k]
                        if self.time_column in self.data.columns else best_k)
            change_points.append({
                'variable': col,
                'time_index': best_k,
                'time_value': time_val,
                'bic_single_segment': float(bic_single),
                'bic_two_segment': float(best_bic),
                'bic_improvement': float(bic_single - best_bic),
                'method': 'information'
            })

        return change_points


# Advanced Analytic Classes
@dataclass
class BayesianResult:
    """Container for Bayesian analysis results."""
    posterior_samples: np.ndarray
    credible_intervals: Dict[str, Tuple[float, float]]
    model_evidence: float
    convergence_diagnostics: Dict[str, float]
    predictive_distributions: Dict[str, Any]


@dataclass
class NetworkResult:
    """Container for network analysis results."""
    graph: Any
    centrality_measures: Dict[str, float]
    community_structure: Dict[str, Any]
    path_analysis: Dict[str, Any]
    network_metrics: Dict[str, float]


@dataclass
class CausalResult:
    """Container for causal inference results."""
    causal_graph: Any
    causal_effects: Dict[str, float]
    confounding_analysis: Dict[str, Any]
    mediation_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, float]


@dataclass
class DimensionalityResult:
    """Container for dimensionality reduction results."""
    embeddings: np.ndarray
    explained_variance: np.ndarray
    reconstruction_error: float
    intrinsic_dimensionality: int
    manifold_structure: Dict[str, Any]


@dataclass
class SurvivalResult:
    """Container for survival analysis results."""
    survival_function: np.ndarray
    hazard_function: np.ndarray
    cumulative_hazard: np.ndarray
    median_survival_time: float
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class SpectralResult:
    """Container for spectral analysis results."""
    power_spectrum: np.ndarray
    frequency_peaks: np.ndarray
    spectral_entropy: float
    dominant_frequencies: np.ndarray
    coherence_matrix: np.ndarray


@dataclass
class NonlinearResult:
    """Container for nonlinear dynamics results."""
    lyapunov_exponents: np.ndarray
    correlation_dimensions: np.ndarray
    attractor_properties: Dict[str, Any]
    chaos_quantifiers: Dict[str, float]
    recurrence_properties: Dict[str, Any]


@dataclass
class InformationResult:
    """Container for information theory results."""
    entropy_measures: Dict[str, float]
    mutual_information: np.ndarray
    transfer_entropy: np.ndarray
    complexity_measures: Dict[str, float]
    information_flow: Dict[str, Any]


@dataclass
class RobustResult:
    """Container for robust statistics results."""
    robust_estimates: Dict[str, float]
    outlier_analysis: Dict[str, Any]
    influence_measures: Dict[str, float]
    breakdown_properties: Dict[str, Any]
    efficiency_comparison: Dict[str, float]


@dataclass
class SpatialResult:
    """Container for spatial analysis results."""
    spatial_autocorrelation: float
    moran_statistics: Dict[str, float]
    spatial_clusters: Dict[str, Any]
    distance_matrices: np.ndarray
    spatial_patterns: Dict[str, Any]


class BayesianAnalyzer:
    """Bayesian inference methods for developmental data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize Bayesian analyzer."""
        self.data = data
        self.prior_parameters = self._set_default_priors()

    def _set_default_priors(self) -> Dict[str, Any]:
        """Set default prior distributions."""
        return {
            'location': {'mean': 0.0, 'precision': 0.001},
            'scale': {'shape': 1.0, 'rate': 1.0},
            'correlation': {'concentration': 1.0}
        }

    def bayesian_linear_regression(self,
                                 x_data: np.ndarray,
                                 y_data: np.ndarray,
                                 n_samples: int = 1000,
                                 seed: Optional[int] = None) -> BayesianResult:
        """
        Perform Bayesian linear regression.

        Parameters:
            x_data: Independent variable data
            y_data: Dependent variable data
            n_samples: Number of posterior samples
            seed: Optional RNG seed for reproducible posterior draws

        Returns:
            BayesianResult with posterior analysis
        """
        # Simple Bayesian linear regression implementation
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)

        # Compute sufficient statistics
        n = len(x_data)
        s_xx = np.sum((x_data - x_mean) ** 2)
        s_xy = np.sum((x_data - x_mean) * (y_data - y_mean))

        # Conjugate Normal-Inverse-Gamma posterior for the slope.
        # Prior: beta ~ N(m0, 1/precision), residual variance tau ~ InvGamma(a0, b0).
        m0 = self.prior_parameters['location']['mean']
        prec0 = self.prior_parameters['location']['precision']
        a0 = self.prior_parameters['scale']['shape']
        b0 = self.prior_parameters['scale']['rate']

        # Analytic posterior for beta with unknown variance (conjugate update)
        post_prec = prec0 + s_xx
        post_mean = (prec0 * m0 + s_xy) / post_prec
        post_a = a0 + n / 2.0
        # RSS about the fitted regression line (centered residuals).
        # NOTE: post_b must use the same centered decomposition; using
        # uncentered sum(y**2) with centered sufficient stats silently
        # inflates the posterior scale.
        slope_hat = (s_xy / s_xx) if s_xx > 0 else 0.0
        rss = np.sum((y_data - y_mean - slope_hat * (x_data - x_mean)) ** 2)
        # Conjugate update in centered form:
        # b_n = b0 + 0.5 * (Tyy + prec0 * m0^2 - post_prec * post_mean^2)
        # with Tyy the centered sum of squares of y (consistent with the
        # centered Sxy/Sxx sufficient statistics above).
        t_yy = np.sum((y_data - y_mean) ** 2)
        post_b = b0 + 0.5 * (t_yy + prec0 * m0 ** 2 - post_prec * post_mean ** 2)
        post_b = max(post_b, 1e-12)

        rng = np.random.default_rng(seed)
        # Sample (tau, beta) jointly from the exact posterior
        tau_samples = rng.gamma(shape=post_a, scale=1.0 / post_b, size=n_samples)
        beta_samples = rng.normal(post_mean, np.sqrt(1.0 / (post_prec * tau_samples)))

        posterior_samples = beta_samples

        # Credible intervals
        credible_intervals = {
            '95%': (np.percentile(posterior_samples, 2.5), np.percentile(posterior_samples, 97.5)),
            '90%': (np.percentile(posterior_samples, 5), np.percentile(posterior_samples, 95))
        }

        # Diagnostics: these are analytic-draw (iid) samples, so R-hat is
        # undefined; report split R-hat computed honestly across the draws.
        halves = np.array_split(beta_samples, 2)
        m = np.mean([h.mean() for h in halves])
        W = np.mean([h.var(ddof=1) for h in halves])
        B = n_samples / 2 * np.var([h.mean() for h in halves], ddof=1)
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        r_hat = float(np.sqrt(var_hat / W)) if W > 0 else 1.0
        convergence_diagnostics = {
            'r_hat': r_hat,
            'effective_sample_size': float(n_samples),  # iid draws by construction
            'posterior_mean': float(post_mean),
            'posterior_var': float(np.var(beta_samples)),
        }

        # Exact NIG log marginal likelihood (conjugate evidence):
        # log p(y) = lgamma(a_n) - lgamma(a_0) + a_0*log(b_0) - a_n*log(b_n)
        #            + 0.5*log(prec_0) - 0.5*log(post_prec)
        #            - (n/2)*log(2*pi)
        # (centered-data form; slope-only model with known intercept at y_mean)
        from math import lgamma
        log_evidence = (lgamma(post_a) - lgamma(a0)
                        + a0 * np.log(max(b0, 1e-300)) - post_a * np.log(max(post_b, 1e-300))
                        + 0.5 * np.log(prec0) - 0.5 * np.log(post_prec)
                        - n / 2.0 * np.log(2 * np.pi))

        return BayesianResult(
            posterior_samples=posterior_samples,
            credible_intervals=credible_intervals,
            model_evidence=float(log_evidence),
            convergence_diagnostics=convergence_diagnostics,
            predictive_distributions={}
        )

    def bayesian_model_comparison(self,
                                 model1_likelihood: float,
                                 model2_likelihood: float,
                                 model1_complexity: int,
                                 model2_complexity: int) -> Dict[str, float]:
        """
        Compare Bayesian models using BIC and AIC.

        Parameters:
            model1_likelihood: Log-likelihood of first model
            model2_likelihood: Log-likelihood of second model
            model1_complexity: Number of parameters in first model
            model2_complexity: Number of parameters in second model

        Returns:
            Dictionary with model comparison metrics
        """
        n_samples = len(self.data)

        # BIC calculation
        bic1 = model1_complexity * np.log(n_samples) - 2 * model1_likelihood
        bic2 = model2_complexity * np.log(n_samples) - 2 * model2_likelihood

        # AIC calculation
        aic1 = 2 * model1_complexity - 2 * model1_likelihood
        aic2 = 2 * model2_complexity - 2 * model2_likelihood

        # Model probabilities (simplified)
        delta_bic = bic2 - bic1
        model1_prob = 1 / (1 + np.exp(delta_bic))
        model2_prob = 1 - model1_prob

        return {
            'bic_model1': bic1,
            'bic_model2': bic2,
            'aic_model1': aic1,
            'aic_model2': aic2,
            'model1_probability': model1_prob,
            'model2_probability': model2_prob,
            'preferred_model': 'model1' if bic1 < bic2 else 'model2'
        }


class NetworkAnalyzer:
    """Graph theory and network analysis methods."""

    def __init__(self, data: pd.DataFrame):
        """Initialize network analyzer."""
        self.data = data
        self.graph = None
        self.distance_matrix = None

    def construct_correlation_network(self,
                                    threshold: float = 0.7,
                                    method: str = 'pearson') -> NetworkResult:
        """
        Construct network from correlation matrix.

        Parameters:
            threshold: Correlation threshold for edge creation
            method: Correlation method

        Returns:
            NetworkResult with network analysis
        """
        # Compute correlation matrix
        numeric_data = self.data.select_dtypes(include=[np.number])
        if method == 'pearson':
            corr_matrix = np.corrcoef(numeric_data.values.T)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")

        # Create network graph
        G = nx.Graph()

        # Add nodes
        for i, col in enumerate(numeric_data.columns):
            G.add_node(col)

        # Add edges based on correlation threshold
        n_vars = len(numeric_data.columns)
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if abs(corr_matrix[i, j]) > threshold:
                    G.add_edge(
                        numeric_data.columns[i],
                        numeric_data.columns[j],
                        weight=abs(corr_matrix[i, j])
                    )

        # Persist the constructed graph for downstream analyses
        self.graph = G

        # Compute centrality measures
        centrality_measures = {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000)
        }

        # Community detection
        try:
            communities = community.greedy_modularity_communities(G)
            community_structure = {
                'communities': list(communities),
                'modularity': community.modularity(G, communities),
                'num_communities': len(communities)
            }
        except:
            community_structure = {'error': 'Community detection failed'}

        # Network metrics
        network_metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'average_degree': np.mean([d for n, d in G.degree()]),
            'connected_components': nx.number_connected_components(G)
        }

        return NetworkResult(
            graph=G,
            centrality_measures=centrality_measures,
            community_structure=community_structure,
            path_analysis={},
            network_metrics=network_metrics
        )

    def shortest_path_analysis(self, source: str, target: str) -> Dict[str, Any]:
        """Compute shortest paths between two variables in the stored graph.

        Requires construct_correlation_network() to have been called first.
        Returns edge-weighted path (distance = ``1/abs(correlation)``), plus
        unweighted hop count. Raises ValueError if nodes are absent or
        disconnected.
        """
        if self.graph is None:
            raise ValueError("No graph available. Call construct_correlation_network() first.")
        if source not in self.graph or target not in self.graph:
            raise ValueError(f"Nodes {source} and/or {target} not in graph")

        # weighted by inverse correlation strength
        weighted = self.graph.copy()
        for u, v, d in weighted.edges(data=True):
            d['distance'] = 1.0 / max(d.get('weight', 1e-12), 1e-12)

        try:
            path = nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"No path between {source} and {target}") from exc

        try:
            weighted_path = nx.shortest_path(weighted, source, target, weight='distance')
            weighted_length = nx.shortest_path_length(weighted, source, target, weight='distance')
        except nx.NetworkXNoPath:
            weighted_path, weighted_length = None, None

        return {
            'path': path,
            'path_length': len(path) - 1,
            'weighted_path': weighted_path,
            'weighted_path_length': weighted_length,
            'all_shortest_paths': list(nx.all_shortest_paths(self.graph, source, target)),
        }


class CausalInference:
    """Causal relationship discovery methods."""

    def __init__(self, data: pd.DataFrame):
        """Initialize causal inference analyzer."""
        self.data = data
        self.causal_graph = None

    def granger_causality_test(self,
                              cause_var: str,
                              effect_var: str,
                              max_lag: int = 5) -> Dict[str, Any]:
        """
        Perform Granger causality test.

        Parameters:
            cause_var: Potential causal variable
            effect_var: Potential effect variable
            max_lag: Maximum lag to test

        Returns:
            Dictionary with Granger causality test results
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Prepare data
            data = self.data[[cause_var, effect_var]].dropna()

            if len(data) < max_lag * 2:
                return {'error': 'Insufficient data for Granger causality test'}

            # Perform Granger causality test
            gc_results = grangercausalitytests(data[[effect_var, cause_var]], maxlag=max_lag)

            # Extract results
            f_tests = []
            p_values = []
            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    f_test = gc_results[lag][0]['ssr_ftest']
                    f_tests.append(f_test[0])  # F-statistic
                    p_values.append(f_test[1])  # p-value

            return {
                'granger_causality': {
                    'f_statistics': f_tests,
                    'p_values': p_values,
                    'lags_tested': list(range(1, max_lag + 1))
                },
                'causal_direction': cause_var + ' -> ' + effect_var,
                'significant_causality': any(p < 0.05 for p in p_values),
                'best_lag': np.argmin(p_values) + 1 if p_values else None
            }

        except Exception as e:
            return {'error': f'Granger causality test failed: {e}'}


class DimensionalityReducer:
    """Advanced dimensionality reduction methods."""

    def __init__(self, data: pd.DataFrame):
        """Initialize dimensionality reducer."""
        self.data = data
        self.embeddings = None

    def fast_ica(self, n_components: int = 2) -> DimensionalityResult:
        """
        Perform FastICA dimensionality reduction.

        Parameters:
            n_components: Number of components

        Returns:
            DimensionalityResult with ICA analysis
        """
        try:
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.data.select_dtypes(include=[np.number]))

            # Perform FastICA
            ica = FastICA(n_components=n_components, random_state=42)
            ica_components = ica.fit_transform(data_scaled)

            # Compute explained variance (approximate)
            explained_variance = np.var(ica_components, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance)

            # Reconstruction error
            reconstructed = ica.inverse_transform(ica_components)
            reconstruction_error = np.mean((data_scaled - reconstructed) ** 2)

            return DimensionalityResult(
                embeddings=ica_components,
                explained_variance=explained_variance_ratio,
                reconstruction_error=reconstruction_error,
                intrinsic_dimensionality=self._estimate_intrinsic_dimension(data_scaled),
                manifold_structure={'algorithm': 'FastICA', 'components': n_components}
            )

        except Exception as e:
            logger.warning(f"FastICA failed: {e}")
            return DimensionalityResult(
                embeddings=np.array([]),
                explained_variance=np.array([]),
                reconstruction_error=1.0,
                intrinsic_dimensionality=0,
                manifold_structure={'error': str(e)}
            )

    def tsne_analysis(self,
                     n_components: int = 2,
                     perplexity: float = 30.0,
                     learning_rate: float = 200.0) -> DimensionalityResult:
        """
        Perform t-SNE dimensionality reduction.

        Parameters:
            n_components: Number of components
            perplexity: Perplexity parameter
            learning_rate: Learning rate

        Returns:
            DimensionalityResult with t-SNE analysis
        """
        try:
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.data.select_dtypes(include=[np.number]))

            # Perform t-SNE
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                random_state=42
            )
            tsne_embeddings = tsne.fit_transform(data_scaled)

            # t-SNE doesn't provide explained variance, so approximate
            explained_variance = np.ones(n_components) / n_components
            reconstruction_error = 1.0  # t-SNE is not reconstructive

            return DimensionalityResult(
                embeddings=tsne_embeddings,
                explained_variance=explained_variance,
                reconstruction_error=reconstruction_error,
                intrinsic_dimensionality=self._estimate_intrinsic_dimension(data_scaled),
                manifold_structure={
                    'algorithm': 't-SNE',
                    'perplexity': perplexity,
                    'learning_rate': learning_rate
                }
            )

        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")
            return DimensionalityResult(
                embeddings=np.array([]),
                explained_variance=np.array([]),
                reconstruction_error=1.0,
                intrinsic_dimensionality=0,
                manifold_structure={'error': str(e)}
            )

    def _estimate_intrinsic_dimension(self, data: np.ndarray, k: int = 10) -> int:
        """
        Estimate intrinsic dimensionality using nearest neighbors.

        Parameters:
            data: Input data
            k: Number of nearest neighbors

        Returns:
            Estimated intrinsic dimension
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            # Fit nearest neighbors
            nn = NearestNeighbors(n_neighbors=k+1).fit(data)
            distances, _ = nn.kneighbors(data)

            # Use average distance to k-th nearest neighbor
            avg_distances = np.mean(distances[:, -1])

            # Estimate intrinsic dimension (simplified)
            # This is a basic implementation - more sophisticated methods exist
            if avg_distances > 0:
                return max(1, min(data.shape[1], int(np.log(data.shape[0]) / np.log(1/avg_distances))))
            else:
                return data.shape[1]

        except Exception as e:
            logger.warning(f"Intrinsic dimension estimation failed: {e}")
            return data.shape[1]


class AnalyticsEngine:
    """Main analytics engine for comprehensive statistical analysis."""

    def __init__(self, data: Union[pd.DataFrame, 'datacore.DataCore'], time_column: str = 'time'):
        """Initialize analytics engine."""
        from . import datacore
        if isinstance(data, datacore.DataCore):
            # Extract data from DataCore
            combined_data = []
            for ts in data.time_series_data:
                combined_data.append(ts.data)
            self.data = pd.concat(combined_data, ignore_index=True)
        else:
            self.data = data

        self.time_column = time_column
        self.ts_analyzer = TimeSeriesAnalyzer(self.data, time_column)
        self.mv_analyzer = MultivariateAnalyzer(self.data)
        self.predictive_modeler = PredictiveModeler(self.data)
        self.change_detector = ChangePointDetector(self.data, time_column)

        # Initialize advanced analyzers
        self.bayesian_analyzer = BayesianAnalyzer(self.data)
        self.network_analyzer = NetworkAnalyzer(self.data)
        self.causal_inferencer = CausalInference(self.data)

        logger.info("Initialized Analytics Engine")

    def analyze_time_series(self) -> TimeSeriesResult:
        """
        Perform comprehensive time series analysis.

        Returns:
            TimeSeriesResult with analysis results
        """
        logger.info("Performing time series analysis")

        # Trend analysis
        trends = self.ts_analyzer.analyze_trends()

        # Seasonality analysis
        seasonality = self.ts_analyzer.detect_seasonality()

        # Change point detection
        change_points = self.ts_analyzer.detect_change_points()

        # Forecasting
        forecasts = self.ts_analyzer.forecast()

        # Model fit assessment
        model_fit = {
            'stationarity': self._test_stationarity(),
            'autocorrelation': self._analyze_autocorrelation()
        }

        result = TimeSeriesResult(
            trend_analysis=trends,
            seasonality_analysis=seasonality,
            change_points=change_points,
            forecasts=forecasts,
            model_fit=model_fit
        )

        logger.info("Time series analysis completed")
        return result

    def analyze_multivariate(self) -> Dict[str, Any]:
        """
        Perform multivariate analysis.

        Returns:
            Dictionary with multivariate analysis results
        """
        logger.info("Performing multivariate analysis")

        results = {
            'principal_components': self.mv_analyzer.principal_component_analysis(),
            'cluster_analysis': self.mv_analyzer.cluster_analysis(),
            'correlation_analysis': self._correlation_analysis()
        }

        logger.info("Multivariate analysis completed")
        return results

    def predictive_modeling(self,
                           target_variable: str,
                           feature_variables: Optional[List[str]] = None,
                           models: Optional[List[str]] = None) -> Dict[str, PredictiveModelResult]:
        """
        Perform predictive modeling.

        Parameters:
            target_variable: Variable to predict
            feature_variables: Feature variables (auto-selected if None)
            models: Models to train (all models if None)

        Returns:
            Dictionary with model results
        """
        logger.info(f"Performing predictive modeling for {target_variable}")

        if feature_variables is None:
            # Auto-select features
            feature_variables = [col for col in self.data.columns
                               if col != target_variable and col != self.time_column]

        if models is None:
            models = list(self.predictive_modeler.models.keys())

        results = {}

        for model_name in models:
            try:
                model_result = self.predictive_modeler.train_predictive_model(
                    target_variable, feature_variables, model_name
                )
                results[model_name] = model_result

                logger.info(f"Trained {model_name} model: R² = {model_result.performance_metrics['test_r2']:.3f}")

            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue

        logger.info("Predictive modeling completed")
        return results

    def detect_changes(self, method: str = 'statistical') -> List[Dict[str, Any]]:
        """
        Detect changes in developmental trajectories.

        Parameters:
            method: Detection method

        Returns:
            List of detected change points
        """
        logger.info(f"Detecting changes using {method} method")

        change_points = self.change_detector.detect_changes(method=method)

        logger.info(f"Detected {len(change_points)} change points")
        return change_points

    def _test_stationarity(self) -> Dict[str, bool]:
        """Test for stationarity in time series."""
        stationarity = {}

        for col in self.ts_analyzer.phenotype_columns:
            series_data = self.data[col].dropna()

            try:
                # Augmented Dickey-Fuller test
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(series_data)
                stationarity[col] = result[1] < 0.05  # p-value < 0.05 indicates stationarity
            except:
                stationarity[col] = False

        return stationarity

    def _analyze_autocorrelation(self) -> Dict[str, Any]:
        """Analyze autocorrelation in time series."""
        autocorrelation = {}

        for col in self.ts_analyzer.phenotype_columns:
            series_data = self.data[col].dropna()

            try:
                # Compute autocorrelation function
                from statsmodels.tsa.stattools import acf
                acf_values = acf(series_data, nlags=min(20, len(series_data)//2))

                autocorrelation[col] = {
                    'autocorrelation_values': acf_values,
                    'lags': list(range(len(acf_values)))
                }
            except:
                autocorrelation[col] = {'error': 'ACF computation failed'}

        return autocorrelation

    def _correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis."""
        # Compute correlation matrix (excluding time column)
        numeric_data = self.data.select_dtypes(include=[np.number])
        if self.time_column in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=[self.time_column])
        correlation_matrix = numeric_data.corr()

        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.7:  # Threshold for high correlation
                    high_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        return {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'mean_correlation': np.mean(np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]))
        }
    def bayesian_analysis(self,
                         x_variable: str,
                         y_variable: str,
                         n_samples: int = 1000,
                         seed: Optional[int] = None) -> BayesianResult:
        """
        Perform Bayesian analysis.

        Parameters:
            x_variable: Independent variable name
            y_variable: Dependent variable name
            n_samples: Number of posterior samples
            seed: Optional RNG seed passed through to the regression sampler

        Returns:
            BayesianResult with Bayesian analysis
        """
        logger.info(f"Performing Bayesian analysis: {x_variable} -> {y_variable}")

        # Drop rows jointly so the x/y pairs stay aligned
        sub = self.data[[x_variable, y_variable]].dropna()
        x_data = sub[x_variable].to_numpy(dtype=float)
        y_data = sub[y_variable].to_numpy(dtype=float)

        if len(x_data) < 10:
            logger.warning("Insufficient data for Bayesian analysis")
            return BayesianResult(
                posterior_samples=np.array([]),
                credible_intervals={},
                model_evidence=0.0,
                convergence_diagnostics={},
                predictive_distributions={}
            )

        return self.bayesian_analyzer.bayesian_linear_regression(x_data, y_data, n_samples, seed=seed)

    def network_analysis(self,
                        correlation_threshold: float = 0.7,
                        method: str = 'pearson') -> NetworkResult:
        """
        Perform network analysis.

        Parameters:
            correlation_threshold: Threshold for edge creation
            method: Correlation method

        Returns:
            NetworkResult with network analysis
        """
        logger.info(f"Performing network analysis with threshold {correlation_threshold}")

        return self.network_analyzer.construct_correlation_network(correlation_threshold, method)

    def causal_inference(self,
                        cause_variable: str,
                        effect_variable: str,
                        max_lag: int = 5) -> Dict[str, Any]:
        """
        Perform causal inference analysis.

        Parameters:
            cause_variable: Potential causal variable
            effect_variable: Potential effect variable
            max_lag: Maximum lag for Granger causality

        Returns:
            Dictionary with causal inference results
        """
        logger.info(f"Performing causal inference: {cause_variable} -> {effect_variable}")

        return self.causal_inferencer.granger_causality_test(cause_variable, effect_variable, max_lag)

    def advanced_dimensionality_reduction(self,
                                        method: str = 'fastica',
                                        n_components: int = 2,
                                        **kwargs) -> DimensionalityResult:
        """
        Perform advanced dimensionality reduction.

        Parameters:
            method: Dimensionality reduction method ('fastica', 'tsne', 'nmf')
            n_components: Number of components
            **kwargs: Additional method parameters

        Returns:
            DimensionalityResult with dimensionality reduction analysis
        """
        logger.info(f"Performing advanced dimensionality reduction using {method}")

        # Create dimensionality reducer instance
        reducer = DimensionalityReducer(self.data)

        if method.lower() == 'fastica':
            return reducer.fast_ica(n_components)
        elif method.lower() == 'tsne':
            perplexity = kwargs.get('perplexity', 30.0)
            learning_rate = kwargs.get('learning_rate', 200.0)
            return reducer.tsne_analysis(n_components, perplexity, learning_rate)
        else:
            logger.warning(f"Unsupported method {method}, using FastICA")
            return reducer.fast_ica(n_components)

    def survival_analysis(self,
                         time_column: str,
                         event_column: str,
                         group_column: Optional[str] = None) -> SurvivalResult:
        """
        Perform survival analysis.

        Parameters:
            time_column: Column with survival times
            event_column: Column with event indicators
            group_column: Optional grouping variable

        Returns:
            SurvivalResult with survival analysis
        """
        logger.info("Performing survival analysis")

        # Simple Kaplan-Meier survival analysis implementation
        if time_column not in self.data.columns or event_column not in self.data.columns:
            return SurvivalResult(
                survival_function=np.array([]),
                hazard_function=np.array([]),
                cumulative_hazard=np.array([]),
                median_survival_time=np.nan,
                confidence_intervals={}
            )

        # Basic survival function estimation.
        # Drop rows jointly so each (time, event) pair stays aligned even
        # when only one of the two columns has a NaN.
        sub = self.data[[time_column, event_column]].dropna()
        times = sub[time_column]
        events = sub[event_column]

        bad_events = sorted({str(v) for v in pd.unique(events)} - {'0', '1', 'False', 'True', '0.0', '1.0'})
        if bad_events:
            raise ValueError(
                f"Event column must contain only 0/1 indicators; got {bad_events}"
            )

        if len(times) < 2:
            return SurvivalResult(
                survival_function=np.array([]),
                hazard_function=np.array([]),
                cumulative_hazard=np.array([]),
                median_survival_time=np.nan,
                confidence_intervals={}
            )

        # Kaplan-Meier survival curve with Nelson-Aalen cumulative hazard
        unique_times = np.sort(times.unique())
        survival_probs = np.ones(len(unique_times))
        nelson_aalen = np.zeros(len(unique_times))
        prev = 1.0
        prev_h = 0.0
        for i, t in enumerate(unique_times):
            at_risk = np.sum(times >= t)
            events_at_t = np.sum((times == t) & (events == 1))
            if at_risk > 0 and events_at_t > 0:
                prev *= (1 - events_at_t / at_risk)
                prev_h += events_at_t / at_risk
            survival_probs[i] = prev
            nelson_aalen[i] = prev_h

        # KM median: first time where S(t) <= 0.5 (NaN if never reached)
        below = np.where(survival_probs <= 0.5)[0]
        median_survival = float(unique_times[below[0]]) if len(below) > 0 else np.nan

        # Greenwood-style standard errors for pointwise CIs
        var_cum = np.zeros(len(unique_times))
        for i, t in enumerate(unique_times):
            at_risk = np.sum(times >= t)
            events_at_t = np.sum((times == t) & (events == 1))
            if at_risk > events_at_t and at_risk > 0:
                var_cum[i] = var_cum[i-1] if i > 0 else 0.0
                var_cum[i] += events_at_t / (at_risk * (at_risk - events_at_t))
        se = np.where(survival_probs > 0, survival_probs * np.sqrt(var_cum), 0.0)
        confidence_intervals = {
            'lower': (survival_probs - 1.96 * se).clip(0, 1),
            'upper': (survival_probs + 1.96 * se).clip(0, 1),
        }

        return SurvivalResult(
            survival_function=survival_probs,
            hazard_function=np.diff(np.concatenate([[0.0], nelson_aalen])),
            cumulative_hazard=nelson_aalen,
            median_survival_time=median_survival,
            confidence_intervals=confidence_intervals
        )

    def spectral_analysis(self,
                         signal_column: str,
                         sampling_frequency: float = 1.0,
                         coherence_column: Optional[str] = None) -> SpectralResult:
        """
        Perform spectral analysis.

        Parameters:
            signal_column: Column with signal data
            sampling_frequency: Sampling frequency
            coherence_column: Optional second column; when given, the
                magnitude-squared coherence between the two signals is
                stored in `coherence_matrix` as an (n_frequencies, 2) array
                of [frequency, coherence]. Without it `coherence_matrix`
                is empty (coherence is a two-signal quantity).

        Returns:
            SpectralResult with spectral analysis
        """
        logger.info(f"Performing spectral analysis on {signal_column}")

        if signal_column not in self.data.columns:
            return SpectralResult(
                power_spectrum=np.array([]),
                frequency_peaks=np.array([]),
                spectral_entropy=0.0,
                dominant_frequencies=np.array([]),
                coherence_matrix=np.array([])
            )

        # Simple power spectral analysis
        signal_data = self.data[signal_column].dropna().values

        if len(signal_data) < 10:
            return SpectralResult(
                power_spectrum=np.array([]),
                frequency_peaks=np.array([]),
                spectral_entropy=0.0,
                dominant_frequencies=np.array([]),
                coherence_matrix=np.array([])
            )

        # Compute power spectrum using Welch's method
        from scipy.signal import welch
        frequencies, power_spectrum = welch(signal_data, fs=sampling_frequency, nperseg=min(256, len(signal_data)//4))

        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(power_spectrum, height=np.percentile(power_spectrum, 75))

        # Compute spectral entropy
        power_normalized = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-12))

        # Get dominant frequencies
        dominant_frequencies = frequencies[peaks] if len(peaks) > 0 else np.array([])

        # Magnitude-squared coherence against a second column, when requested
        coherence_matrix = np.array([])
        if coherence_column is not None and coherence_column in self.data.columns:
            paired = self.data[[signal_column, coherence_column]].dropna()
            if len(paired) >= 10:
                coh_freqs, coh = signal.coherence(
                    paired[signal_column].to_numpy(dtype=float),
                    paired[coherence_column].to_numpy(dtype=float),
                    fs=sampling_frequency,
                )
                coherence_matrix = np.column_stack([coh_freqs, coh])
        return SpectralResult(
            power_spectrum=np.column_stack([frequencies, power_spectrum]),
            frequency_peaks=np.column_stack([frequencies[peaks], power_spectrum[peaks]]) if len(peaks) > 0 else np.array([]),
            spectral_entropy=spectral_entropy,
            dominant_frequencies=dominant_frequencies,
            coherence_matrix=coherence_matrix
        )

    def nonlinear_dynamics_analysis(self,
                                  time_series_column: str,
                                  embedding_dim: int = 3,
                                  tau: int = 1) -> Dict[str, Any]:
        """
        Perform nonlinear dynamics analysis.

        Parameters:
            time_series_column: Column with time series data
            embedding_dim: Embedding dimension
            tau: Time delay

        Returns:
            Dictionary with nonlinear dynamics analysis
        """
        logger.info(f"Performing nonlinear dynamics analysis on {time_series_column}")

        if time_series_column not in self.data.columns:
            return {'error': f'Column {time_series_column} not found'}

        time_series = self.data[time_series_column].dropna().values

        if len(time_series) < embedding_dim * 10:
            return {'error': 'Insufficient data for nonlinear dynamics analysis'}

        # Largest Lyapunov exponent via the Rosenstein (1993) method on a
        # time-delay-embedded attractor. Neighbor search is forward-only
        # (j >= i), which halves the pair cost; pairs closer than the
        # Theiler window are excluded so temporally adjacent segments of
        # the same trajectory are not mistaken for distinct neighbors.
        def _embed(series: np.ndarray, m: int, delay: int) -> np.ndarray:
            n = len(series) - (m - 1) * delay
            return np.array([series[i:i + (m - 1) * delay + 1:delay] for i in range(n)])

        embedded = _embed(time_series, embedding_dim, tau)
        n_vec = len(embedded)
        theiler = embedding_dim  # conservative Theiler window

        nearest = np.full(n_vec, -1)
        for i in range(n_vec):
            dists = np.linalg.norm(embedded[i:] - embedded[i], axis=1)
            dists[:min(theiler, len(dists))] = np.inf
            if np.isfinite(dists).any() and np.isfinite(dists[dists < np.inf]).any():
                nearest[i] = i + int(np.argmin(dists))

        diverge_t = min(50, n_vec // 2)
        ln_divs = []
        for i in range(n_vec):
            j = nearest[i]
            if j < 0 or i + diverge_t >= n_vec or j + diverge_t >= n_vec:
                continue
            d0 = np.linalg.norm(embedded[i] - embedded[j])
            if d0 <= 0:
                continue
            traj = [np.log(np.linalg.norm(embedded[i + k] - embedded[j + k]) / d0)
                    for k in range(min(diverge_t, n_vec - max(i, j)))
                    if np.linalg.norm(embedded[i + k] - embedded[j + k]) > 0]
            if traj:
                ln_divs.append(traj)

        if ln_divs:
            min_len = min(len(t) for t in ln_divs)
            mean_ln_div = np.mean([t[:min_len] for t in ln_divs], axis=0)
            # Rosenstein: fit the slope only over the initial part of the
            # divergence curve, where exponential growth holds; the tail
            # saturates at the attractor's finite size and would bias the
            # estimate downward.
            n_fit = max(2, min_len // 2)
            k_fit = np.arange(n_fit)
            slope = np.polyfit(k_fit, mean_ln_div[:n_fit], 1)[0]
            largest_lyapunov = float(slope)
        else:
            largest_lyapunov = np.nan

        # Correlation dimension via Grassberger-Procaccia slopes (real computation)
        radii = np.logspace(-2, 0, 10) * np.std(time_series)
        corr_sums = []
        for r in radii:
            count = 0
            total = 0
            for i in range(0, n_vec, max(1, n_vec // 200)):
                dists = np.linalg.norm(embedded[i+1:] - embedded[i], axis=1)
                count += np.sum(dists < r)
                total += len(dists)
            corr_sums.append(np.log(max(count / max(total, 1), 1e-300)))
        corr_sums = np.array(corr_sums)
        log_radii = np.log(radii)
        # local slopes as correlation-dimension estimates
        local_slopes = np.diff(corr_sums) / np.diff(log_radii)

        return {
            'largest_lyapunov_exponent': largest_lyapunov,
            'correlation_dimensions': local_slopes,
            'attractor_properties': {'embedding_dim': embedding_dim, 'tau': tau},
            'chaos_quantifiers': {'lyapunov_exponent': largest_lyapunov},
            'recurrence_properties': {}
        }

    def information_theory_analysis(self, data_column: str) -> Dict[str, float]:
        """
        Perform information theory analysis.

        Parameters:
            data_column: Column to analyze

        Returns:
            Dictionary with information theory measures
        """
        logger.info(f"Performing information theory analysis on {data_column}")

        if data_column not in self.data.columns:
            return {'shannon_entropy': 0.0, 'normalized_entropy': 0.0}

        data_values = self.data[data_column].dropna().values

        if len(data_values) < 2:
            return {'shannon_entropy': 0.0, 'normalized_entropy': 0.0}

        # Simple Shannon entropy computation
        hist, bin_edges = np.histogram(data_values, bins=min(30, len(np.unique(data_values))))
        probs = hist / np.sum(hist)
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # Normalized entropy (entropy divided by maximum possible entropy)
        max_entropy = np.log2(len(probs))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy
        }

    def robust_statistical_analysis(self, data_column: str) -> Dict[str, float]:
        """
        Perform robust statistical analysis.

        Parameters:
            data_column: Column to analyze

        Returns:
            Dictionary with robust statistical measures
        """
        logger.info(f"Performing robust statistical analysis on {data_column}")

        if data_column not in self.data.columns:
            return {
                'location_estimates': {},
                'scale_estimates': {},
                'robust_location_preferred': np.nan,
                'robust_scale_preferred': np.nan
            }

        data_values = self.data[data_column].dropna().values

        if len(data_values) < 4:
            return {
                'location_estimates': {},
                'scale_estimates': {},
                'robust_location_preferred': np.nan,
                'robust_scale_preferred': np.nan
            }

        # Robust location estimates
        location_estimates = {
            'median': np.median(data_values),
            'trimmed_mean': np.mean(data_values[1:-1]) if len(data_values) > 2 else np.mean(data_values),
            'huber_estimator': self._huber_estimate(data_values),
            'tukey_biweight': self._tukey_biweight_estimate(data_values)
        }

        # Robust scale estimates
        mad = np.median(np.abs(data_values - np.median(data_values)))
        scale_estimates = {
            'mad': mad,
            'mad_normalized': mad * 1.4826,  # Consistent with normal distribution
            'iqr': np.subtract(*np.percentile(data_values, [75, 25])),
            'sn_scale': self._sn_scale_estimate(data_values)
        }

        return {
            'location_estimates': location_estimates,
            'scale_estimates': scale_estimates,
            'robust_location_preferred': location_estimates.get('trimmed_mean', np.nan),
            'robust_scale_preferred': scale_estimates.get('mad_normalized', np.nan)
        }

    def _huber_estimate(self, data: np.ndarray, k: float = 1.345,
                        max_iter: int = 100, tol: float = 1e-8) -> float:
        """Huber M-estimator of location via IRLS with a fixed robust scale.

        Scale is held at the normalized MAD (standard practice for
        location-only M-estimation); psi(u) = u for |u| <= k, k*sign(u)
        otherwise, with u = (x - mu) / scale.
        """
        x = np.asarray(data, dtype=float)
        mu = float(np.median(x))
        scale = float(np.median(np.abs(x - mu))) * 1.4826
        if scale <= 0:
            return float(mu)
        for _ in range(max_iter):
            u = (x - mu) / scale
            w = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))
            mu_new = float(np.sum(w * x) / np.sum(w))
            done = abs(mu_new - mu) < tol
            mu = mu_new
            if done:
                break
        return float(mu)

    def _tukey_biweight_estimate(self, data: np.ndarray, c: float = 4.685,
                                 max_iter: int = 100, tol: float = 1e-8) -> float:
        """Tukey biweight M-estimator of location via IRLS.

        u = (x - mu) / (c * scale) with scale the normalized MAD;
        w(u) = (1 - u^2)^2 for |u| < 1 and 0 otherwise (redescending psi).
        """
        x = np.asarray(data, dtype=float)
        mu = float(np.median(x))
        scale = float(np.median(np.abs(x - mu))) * 1.4826
        if scale <= 0:
            return float(mu)
        for _ in range(max_iter):
            u = (x - mu) / (c * scale)
            w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
            denom = float(np.sum(w))
            if denom <= 0:
                break
            mu_new = float(np.sum(w * x) / denom)
            done = abs(mu_new - mu) < tol
            if done:
                break
        return float(mu)

    def _sn_scale_estimate(self, data: np.ndarray) -> float:
        """Rousseeuw-Croux Sn scale estimator.

        Sn = 1.1926 * median_i { median_{j != i} |x_i - x_j| }; the 1.1926
        factor makes Sn consistent with sigma for Gaussian data. Unlike the
        MAD, Sn stays efficient even for asymmetric contamination.
        """
        x = np.asarray(data, dtype=float)
        n = len(x)
        if n < 2:
            return 0.0
        abs_diff = np.abs(x[:, None] - x[None, :])
        np.fill_diagonal(abs_diff, np.nan)
        inner = np.nanmedian(abs_diff, axis=1)
        return float(1.1926 * np.median(inner))

    def spatial_analysis(self,
                        value_column: str,
                        spatial_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform spatial analysis.

        Parameters:
            value_column: Column with values to analyze
            spatial_weights: Optional n x n spatial weights matrix W. When
                given, Moran's I is I = (n / S0) * (z' W z) / (z' z) with z
                the centered values and S0 = sum(W). When omitted, a
                linear-adjacency default is used (w_ij = 1 for adjacent
                observations in sequence order) and the result reports
                'weights_kind': 'linear_adjacency'.

        Returns:
            Dictionary with spatial analysis results
        """
        logger.info(f"Performing spatial analysis on {value_column}")

        if value_column not in self.data.columns:
            return {'morans_i': np.nan, 'spatial_autocorrelation': 'No data'}

        data_values = self.data[value_column].dropna().values

        if len(data_values) < 4:
            return {'morans_i': np.nan, 'spatial_autocorrelation': 'Insufficient data'}

        # Moran's I on centered values with an explicit weights matrix.
        n = len(data_values)
        if spatial_weights is not None:
            W = np.asarray(spatial_weights, dtype=float)
            if W.shape != (n, n):
                raise ValueError(
                    f"spatial_weights must be ({n}, {n}) to match the "
                    f"{n} non-NaN observations of {value_column!r}"
                )
            weights_kind = 'supplied'
        else:
            W = np.zeros((n, n))
            adj = np.arange(n - 1)
            W[adj, adj + 1] = 1.0
            W[adj + 1, adj] = 1.0
            weights_kind = 'linear_adjacency'

        z = data_values - np.mean(data_values)
        S0 = float(W.sum())
        denom = float(z @ z)
        if S0 <= 0 or denom <= 0:
            morans_i = np.nan
        else:
            morans_i = float((n / S0) * (z @ (W @ z)) / denom)

        return {
            'morans_i': morans_i,
            'spatial_autocorrelation': 'Positive' if morans_i > 0.1 else 'Negative' if morans_i < -0.1 else 'None',
            'weights_kind': weights_kind
        }

    def comprehensive_analysis_report(self,
                                      bayesian_columns: Optional[Tuple[str, str]] = None,
                                      causal_columns: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all analytic methods.

        Parameters:
            bayesian_columns: Optional (x, y) column pair for the Bayesian
                regression section. Bayesian analysis on the first two
                arbitrary numeric columns is meaningless, so the section is
                reported as 'not_analyzed' unless an explicit pair is given.
            causal_columns: Optional (cause, effect) column pair for the
                Granger causality section; same 'not_analyzed' default.

        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info("Generating comprehensive analysis report")

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'n_samples': len(self.data),
                'n_variables': len(self.data.columns),
                'time_column': self.time_column
            }
        }

        # Time series analysis
        try:
            report['time_series'] = self.analyze_time_series()
        except Exception as e:
            report['time_series'] = {'error': str(e)}

        # Multivariate analysis
        try:
            report['multivariate'] = self.analyze_multivariate()
        except Exception as e:
            report['multivariate'] = {'error': str(e)}

        # Bayesian analysis needs an explicitly chosen (x, y) pair; running
        # it on the first two arbitrary numeric columns produced results
        # with no interpretation.
        if bayesian_columns is not None:
            try:
                report['bayesian'] = self.bayesian_analysis(*bayesian_columns)
            except Exception as e:
                report['bayesian'] = {'error': str(e)}
        else:
            report['bayesian'] = {
                'not_analyzed': 'Pass bayesian_columns=(x, y) to run Bayesian regression on a chosen pair'
            }

        # Network analysis
        try:
            report['network'] = self.network_analysis()
        except Exception as e:
            report['network'] = {'error': str(e)}

        # Same rationale as the Bayesian section: Granger causality is only
        # meaningful for a hypothesized cause -> effect pair.
        if causal_columns is not None:
            try:
                report['causal'] = self.causal_inference(*causal_columns)
            except Exception as e:
                report['causal'] = {'error': str(e)}
        else:
            report['causal'] = {
                'not_analyzed': 'Pass causal_columns=(cause, effect) to run Granger causality on a chosen pair'
            }

        # Information theory
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                report['information_theory'] = self.information_theory_analysis(numeric_cols[0])
            else:
                report['information_theory'] = {'error': 'No numeric columns'}
        except Exception as e:
            report['information_theory'] = {'error': str(e)}

        # Robust statistics
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                report['robust_statistics'] = self.robust_statistical_analysis(numeric_cols[0])
            else:
                report['robust_statistics'] = {'error': 'No numeric columns'}
        except Exception as e:
            report['robust_statistics'] = {'error': str(e)}

        logger.info("Comprehensive analysis report completed")
        return report
    
    def wavelet_analysis(self,
                        column: str,
                        wavelet: str = 'morl',
                        scales: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform wavelet analysis to identify time-frequency patterns.
        
        Parameters:
            column: Column name to analyze
            wavelet: Wavelet type ('morl', 'cgau1', 'gaus1', 'mexh')
            scales: Scales for wavelet transform
        
        Returns:
            Dictionary with wavelet coefficients and analysis
        """
        logger.info(f"Performing wavelet analysis on {column}")
        
        import pywt
        
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data")
        
        signal = self.data[column].dropna().values
        
        # Set scales if not provided
        if scales is None:
            scales = np.arange(1, min(128, len(signal) // 4))
        
        # Continuous wavelet transform
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
        
        # Compute power spectrum
        power = np.abs(coefficients) ** 2
        
        # Find dominant frequencies
        avg_power = np.mean(power, axis=1)
        dominant_scale_idx = np.argmax(avg_power)
        dominant_scale = scales[dominant_scale_idx]
        
        # Identify time-localized events
        threshold = np.percentile(power, 95)
        events = np.where(power > threshold)
        
        return {
            'coefficients': coefficients,
            'scales': scales,
            'frequencies': frequencies,
            'power_spectrum': power,
            'dominant_scale': float(dominant_scale),
            'avg_power': avg_power,
            'n_events': len(events[0]),
            'event_locations': {'time': events[1].tolist(), 'scale': events[0].tolist()}
        }
    
    def copula_analysis(self,
                       column1: str,
                       column2: str,
                       copula_type: str = 'gaussian') -> Dict[str, Any]:
        """
        Analyze dependence structure using copulas.
        
        Parameters:
            column1: First column name
            column2: Second column name
            copula_type: Type of copula ('gaussian', 'student', 'clayton', 'frank')
        
        Returns:
            Dictionary with copula parameters and dependence measures
        """
        logger.info(f"Performing copula analysis between {column1} and {column2}")
        
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError("Columns not found in data")

        # Drop rows jointly so the (column1, column2) pairs stay aligned
        sub = self.data[[column1, column2]].dropna()
        data1 = sub[column1].to_numpy(dtype=float)
        data2 = sub[column2].to_numpy(dtype=float)
        tau, tau_pval = stats.kendalltau(data1, data2)
        if len(data1) < 3 or not np.isfinite(tau):
            raise ValueError("Insufficient variation between columns to fit a copula")
        
        # Transform to uniform margins using empirical CDF
        from scipy.stats import rankdata
        u1 = rankdata(data1) / (len(data1) + 1)
        u2 = rankdata(data2) / (len(data2) + 1)
        
        # Compute dependence measures
        # Kendall's tau
        tau, tau_pval = stats.kendalltau(data1, data2)
        
        # Spearman's rho
        rho, rho_pval = stats.spearmanr(data1, data2)
        
        # Tail dependence (empirical)
        threshold = 0.95
        upper_tail_prob = np.mean((u1 > threshold) & (u2 > threshold))
        lower_tail_prob = np.mean((u1 < (1 - threshold)) & (u2 < (1 - threshold)))
        
        # Fit copula parameters
        degrees_of_freedom = None
        if copula_type == 'gaussian':
            # Gaussian copula: correlation of the inverse-normal-transformed
            # uniforms (maximum-likelihood for Gaussian margins)
            from scipy.stats import norm
            z1 = norm.ppf(u1)
            z2 = norm.ppf(u2)
            copula_param = np.corrcoef(z1, z2)[0, 1]
        elif copula_type == 'student':
            # Bivariate Student-t copula: rho from Kendall's tau
            # (tau = (2/pi) arcsin(rho) for the t copula) and nu by the
            # method of moments on excess kurtosis (nu = 4 + 6/excess,
            # valid for nu > 4; clamped otherwise).
            copula_param = float(np.sin(np.pi * tau / 2.0))
            kurt = 0.5 * (float(stats.kurtosis(data1)) + float(stats.kurtosis(data2)))
            nu = 4.0 + 6.0 / kurt if kurt > 1e-12 else 100.0
            degrees_of_freedom = float(min(max(nu, 2.1), 100.0))
        elif copula_type == 'clayton':
            # Clayton copula parameter (method of moments using Kendall's
            # tau); Clayton only admits positive dependence.
            if tau <= 0:
                raise ValueError(
                    "Clayton copula requires positive dependence (Kendall's tau > 0); "
                    f"got tau = {tau:.3f}"
                )
            copula_param = 2 * tau / (1 - tau)
        elif copula_type == 'frank':
            # Frank copula parameter: exact inversion of
            # tau = 1 - 4/theta * (1 - Debye1(theta)) for theta > 0
            # (sign-flipped for negative tau).
            copula_param = self._frank_theta_from_tau(tau)
        else:
            raise ValueError(
                f"Unsupported copula type: {copula_type!r}; "
                "expected one of 'gaussian', 'student', 'clayton', 'frank'"
            )

        result = {
            'copula_type': copula_type,
            'copula_parameter': float(copula_param),
            'kendall_tau': float(tau),
            'kendall_tau_pvalue': float(tau_pval),
            'spearman_rho': float(rho),
            'spearman_rho_pvalue': float(rho_pval),
            'upper_tail_dependence': float(upper_tail_prob),
            'lower_tail_dependence': float(lower_tail_prob),
            'dependence_class': 'positive' if tau > 0.1 else 'negative' if tau < -0.1 else 'independent'
        }
        if degrees_of_freedom is not None:
            result['degrees_of_freedom'] = degrees_of_freedom
        return result

    @staticmethod
    def _frank_theta_from_tau(tau: float) -> float:
        """Solve tau = 1 - 4/theta * (1 - Debye1(theta)) for theta > 0.

        The map theta -> tau is strictly increasing from 0 (theta -> 0+) to
        1 (theta -> inf), so brentq on a wide positive bracket is exact.
        Negative tau is handled by symmetry (theta -> -theta).
        """
        from scipy.optimize import brentq

        if not np.isfinite(tau):
            raise ValueError("Kendall's tau is not finite; cannot fit Frank copula")
        if abs(tau) < 1e-12:
            return 0.0

        target = abs(tau)

        def _debye1(x: float) -> float:
            """Debye function D1(x) = (1/x) * integral_0^x t/(e^t - 1) dt.

            scipy.special does not ship it in every version, so compute the
            integral directly: quadrature on (0, x) for moderate x, and the
            total integral pi^2/6 minus the exponentially small tail for
            large x.
            """
            from scipy.integrate import quad
            if x <= 0:
                return 1.0
            integrand = lambda t: t / np.expm1(t)
            if x < 5.0:
                integral, _ = quad(integrand, 0.0, x, limit=200)
                return integral / x
            total, _ = quad(integrand, x, np.inf, limit=200)
            return (np.pi ** 2 / 6.0 - total) / x

        def _tau_of_theta(theta: float) -> float:
            # Debye D1 is evaluated at theta itself (not 1/theta): D1 -> 1
            # as theta -> 0 gives tau -> 0, and D1 -> 0 as theta -> inf
            # gives tau -> 1, matching the Frank copula's tau range.
            return 1.0 - 4.0 / theta * (1.0 - _debye1(theta))

        lo, hi = 1e-8, 1e12
        theta = brentq(lambda th: _tau_of_theta(th) - target, lo, hi, xtol=1e-12, rtol=1e-14)
        return float(np.sign(tau) * theta)
        
    
    def extreme_value_analysis(self,
                              column: str,
                              threshold: Optional[float] = None,
                              block_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze extreme values using peaks-over-threshold and block maxima methods.
        
        Parameters:
            column: Column name to analyze
            threshold: Threshold for peaks-over-threshold method (if None, auto-select)
            block_size: Size of blocks for block maxima method (if None, use n/10)
        
        Returns:
            Dictionary with extreme value parameters and return levels
        """
        logger.info(f"Performing extreme value analysis on {column}")
        
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data")
        
        data = self.data[column].dropna().values
        n = len(data)
        
        # Peaks-over-threshold (POT) method
        if threshold is None:
            # Use 90th percentile as threshold
            threshold = np.percentile(data, 90)
        
        exceedances = data[data > threshold] - threshold
        n_exceedances = len(exceedances)
        
        # Define return periods globally
        return_periods = np.array([10, 50, 100, 500])
        
        # Fit Generalized Pareto Distribution to exceedances
        if n_exceedances > 10:
            from scipy.stats import genpareto
            shape, loc, scale = genpareto.fit(exceedances)
            
            # Compute return levels
            zeta = n_exceedances / n  # Exceedance rate
            return_levels = threshold + (scale / shape) * (
                (return_periods * zeta) ** shape - 1
            )
        else:
            shape, loc, scale = 0.0, 0.0, 1.0
            return_levels = np.array([threshold] * 4)
        
        # Block maxima method
        if block_size is None:
            block_size = max(1, n // 10)
        
        n_blocks = n // block_size
        block_maxima = np.array([
            np.max(data[i * block_size:(i + 1) * block_size])
            for i in range(n_blocks)
        ])
        
        # Fit Generalized Extreme Value (GEV) distribution
        from scipy.stats import genextreme
        gev_shape, gev_loc, gev_scale = genextreme.fit(block_maxima)
        
        # Compute GEV-based return levels
        gev_return_levels = genextreme.isf(1 / return_periods, gev_shape, gev_loc, gev_scale)
        
        # Extreme value index (Hill estimator)
        sorted_data = np.sort(data)[::-1]
        k = min(int(np.sqrt(n)), 100)  # Number of order statistics to use
        hill_estimator = np.mean(np.log(sorted_data[:k])) - np.log(sorted_data[k])
        
        return {
            'pot_method': {
                'threshold': float(threshold),
                'n_exceedances': int(n_exceedances),
                'shape_parameter': float(shape),
                'scale_parameter': float(scale),
                'return_levels': {
                    f'{int(rp)}_year': float(rl)
                    for rp, rl in zip(return_periods, return_levels)
                }
            },
            'block_maxima_method': {
                'block_size': int(block_size),
                'n_blocks': int(n_blocks),
                'gev_shape': float(gev_shape),
                'gev_location': float(gev_loc),
                'gev_scale': float(gev_scale),
                'return_levels': {
                    f'{int(rp)}_year': float(rl)
                    for rp, rl in zip(return_periods, gev_return_levels)
                }
            },
            'hill_estimator': float(hill_estimator),
            'tail_index': float(1 / hill_estimator) if hill_estimator > 0 else np.inf
        }
    
    def regime_switching_analysis(self,
                                 column: str,
                                 n_regimes: int = 2) -> Dict[str, Any]:
        """
        Identify regime switches in time series data.
        
        Parameters:
            column: Column name to analyze
            n_regimes: Number of regimes to identify
        
        Returns:
            Dictionary with regime information and transition probabilities
        """
        logger.info(f"Performing regime switching analysis on {column}")
        
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data")
        
        data = self.data[column].dropna().values
        n = len(data)
        
        # Use K-means clustering on windowed statistics
        window_size = min(20, n // 10)
        features = []
        
        for i in range(n - window_size + 1):
            window = data[i:i + window_size]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                np.percentile(window, 75) - np.percentile(window, 25)
            ])
        
        features = np.array(features)
        
        # Normalize features; constant windows give zero std, which would
        # otherwise poison the features with NaN and crash K-means.
        std = features.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        features_normalized = (features - features.mean(axis=0)) / std

        # K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(features_normalized)

        # Map each observation to the label of the window whose midpoint is
        # nearest. (Forward-filling each window's label over its span let
        # later windows overwrite earlier ones, smearing labels backwards.)
        full_regime_labels = regime_labels[
            np.clip(np.arange(n) - window_size // 2, 0, len(regime_labels) - 1)
        ]
        
        # Compute regime statistics
        regime_stats = []
        for regime in range(n_regimes):
            regime_data = data[full_regime_labels == regime]
            if len(regime_data) > 0:
                regime_stats.append({
                    'regime_id': int(regime),
                    'mean': float(np.mean(regime_data)),
                    'std': float(np.std(regime_data)),
                    'duration_pct': float(len(regime_data) / n * 100),
                    'n_observations': int(len(regime_data))
                })
        
        # Compute transition matrix between the per-observation labels
        # (the reported labels), not the window-level ones
        transitions = np.zeros((n_regimes, n_regimes))
        for a, b in zip(full_regime_labels[:-1], full_regime_labels[1:]):
            transitions[a, b] += 1
        
        # Normalize to get probabilities
        transition_probs = transitions / transitions.sum(axis=1, keepdims=True)
        transition_probs = np.nan_to_num(transition_probs)
        
        # Identify regime switches
        switches = np.where(np.diff(full_regime_labels) != 0)[0]
        
        return {
            'n_regimes': int(n_regimes),
            'regime_labels': full_regime_labels.tolist(),
            'regime_statistics': regime_stats,
            'transition_matrix': transitions.tolist(),
            'transition_probabilities': transition_probs.tolist(),
            'n_switches': int(len(switches)),
            'switch_timepoints': switches.tolist()
        }
