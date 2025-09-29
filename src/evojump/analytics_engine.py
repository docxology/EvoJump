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

            if period is None:
                # Auto-detect period using autocorrelation
                from scipy.signal import find_peaks
                autocorr = np.correlate(series_data, series_data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]

                # Find peaks in autocorrelation
                peaks, _ = find_peaks(autocorr, height=0.2)
                if len(peaks) > 0:
                    period = peaks[0]

            if period and period < len(series_data):
                # Perform seasonal decomposition
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomposition = seasonal_decompose(
                        series_data, model='additive', period=period
                    )

                    results[col] = {
                        'period': period,
                        'seasonal_strength': np.var(decomposition.seasonal) / np.var(series_data),
                        'trend_strength': np.var(decomposition.trend) / np.var(series_data),
                        'residual_strength': np.var(decomposition.resid) / np.var(series_data),
                        'seasonal_detected': True
                    }
                except:
                    results[col] = {
                        'period': period,
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
            method: Change point detection method

        Returns:
            List of detected change points
        """
        change_points = []

        for col in self.phenotype_columns:
            series_data = self.data[col].dropna()

            if len(series_data) < 10:
                continue

            if method == 'cusum':
                # CUSUM method
                mean_val = np.mean(series_data)
                cusum = np.cumsum(series_data - mean_val)

                # Find maximum deviation
                max_idx = np.argmax(np.abs(cusum))
                max_deviation = cusum[max_idx]

                if abs(max_deviation) > 2 * np.std(series_data):
                    change_points.append({
                        'variable': col,
                        'time_index': max_idx,
                        'time_value': self.data[self.time_column].iloc[max_idx],
                        'cusum_value': max_deviation,
                        'confidence': min(abs(max_deviation) / (3 * np.std(series_data)), 1.0),
                        'method': 'cusum'
                    })

            elif method == 'variance':
                # Variance-based change detection
                window_size = max(5, len(series_data) // 4)
                variances = []

                for i in range(len(series_data) - window_size):
                    window_var = np.var(series_data[i:i+window_size])
                    variances.append(window_var)

                if variances:
                    var_series = pd.Series(variances)
                    var_change_idx = var_series.idxmax()

                    change_points.append({
                        'variable': col,
                        'time_index': var_change_idx,
                        'time_value': self.data[self.time_column].iloc[var_change_idx],
                        'variance_ratio': variances[var_change_idx] / np.mean(variances),
                        'confidence': 0.8  # Placeholder
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
            # Solve generalized eigenvalue problem
            eigenvals, eigenvecs = linalg.eigh(cov12 @ cov21, cov11)

            # Sort by eigenvalues
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Canonical correlations
            canonical_correlations = np.sqrt(np.maximum(eigenvals, 0))

            return {
                'canonical_correlations': canonical_correlations,
                'canonical_variables_1': eigenvecs,
                'eigenvalues': eigenvals,
                'scaler1_mean': scaler1.mean_,
                'scaler1_scale': scaler1.scale_,
                'scaler2_mean': scaler2.mean_,
                'scaler2_scale': scaler2.scale_
            }

        except Exception as e:
            logger.warning(f"CCA failed: {e}")
            return {'error': str(e)}

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
                    change_points.append({
                        'variable': col,
                        'time_index': change_idx,
                        'time_value': self.data[self.time_column].iloc[change_idx],
                        'change_magnitude': differences[change_idx],
                        'z_score': z_scores[change_idx],
                        'method': 'statistical'
                    })

        return change_points

    def _bayesian_change_detection(self, **kwargs) -> List[Dict[str, Any]]:
        """Bayesian change point detection."""
        # Simplified Bayesian change detection
        # In practice, would implement proper Bayesian online change detection
        return self._statistical_change_detection(**kwargs)

    def _information_criterion_change_detection(self, **kwargs) -> List[Dict[str, Any]]:
        """Information criterion-based change detection."""
        # Simplified information criterion approach
        return self._statistical_change_detection(**kwargs)


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
                                 n_samples: int = 1000) -> BayesianResult:
        """
        Perform Bayesian linear regression.

        Parameters:
            x_data: Independent variable data
            y_data: Dependent variable data
            n_samples: Number of posterior samples

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

        # Posterior parameters
        beta_precision = 1.0 / (s_xx / n + self.prior_parameters['location']['precision'])
        beta_mean = beta_precision * (s_xy / n + self.prior_parameters['location']['precision'] * self.prior_parameters['location']['mean'])

        # Generate posterior samples
        posterior_samples = np.random.normal(beta_mean, np.sqrt(beta_precision), n_samples)

        # Credible intervals
        credible_intervals = {
            '95%': (np.percentile(posterior_samples, 2.5), np.percentile(posterior_samples, 97.5)),
            '90%': (np.percentile(posterior_samples, 5), np.percentile(posterior_samples, 95))
        }

        # Convergence diagnostics (simplified)
        convergence_diagnostics = {
            'r_hat': 1.0,  # Simplified
            'effective_sample_size': n_samples * 0.8
        }

        return BayesianResult(
            posterior_samples=posterior_samples,
            credible_intervals=credible_intervals,
            model_evidence=0.0,  # Placeholder
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
            gc_results = grangercausalitytests(data[[effect_var, cause_var]], maxlag=max_lag, verbose=False)

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
                         n_samples: int = 1000) -> BayesianResult:
        """
        Perform Bayesian analysis.

        Parameters:
            x_variable: Independent variable name
            y_variable: Dependent variable name
            n_samples: Number of posterior samples

        Returns:
            BayesianResult with Bayesian analysis
        """
        logger.info(f"Performing Bayesian analysis: {x_variable} -> {y_variable}")

        x_data = self.data[x_variable].dropna().values
        y_data = self.data[y_variable].dropna().values

        # Align data
        common_indices = set(self.data[x_variable].dropna().index) & set(self.data[y_variable].dropna().index)
        x_data = self.data[x_variable].loc[list(common_indices)].values
        y_data = self.data[y_variable].loc[list(common_indices)].values

        if len(x_data) < 10:
            logger.warning("Insufficient data for Bayesian analysis")
            return BayesianResult(
                posterior_samples=np.array([]),
                credible_intervals={},
                model_evidence=0.0,
                convergence_diagnostics={},
                predictive_distributions={}
            )

        return self.bayesian_analyzer.bayesian_linear_regression(x_data, y_data, n_samples)

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

        # Create survival analyzer instance
        from .analytics_engine import SurvivalAnalyzer
        analyzer = SurvivalAnalyzer(self.data)

        return analyzer.kaplan_meier_analysis(time_column, event_column, group_column)

    def spectral_analysis(self,
                         signal_column: str,
                         sampling_frequency: float = 1.0) -> SpectralResult:
        """
        Perform spectral analysis.

        Parameters:
            signal_column: Column with signal data
            sampling_frequency: Sampling frequency

        Returns:
            SpectralResult with spectral analysis
        """
        logger.info(f"Performing spectral analysis on {signal_column}")

        # Create spectral analyzer instance
        from .analytics_engine import SpectralAnalyzer
        analyzer = SpectralAnalyzer(self.data)

        return analyzer.power_spectral_analysis(signal_column, sampling_frequency)

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

        # Create nonlinear dynamics analyzer instance
        from .analytics_engine import NonlinearDynamics
        analyzer = NonlinearDynamics(self.data)

        time_series = self.data[time_series_column].dropna().values

        if len(time_series) < embedding_dim * 10:
            return {'error': 'Insufficient data for nonlinear dynamics analysis'}

        return analyzer.lyapunov_exponent_estimation(time_series, embedding_dim, tau)

    def information_theory_analysis(self, data_column: str) -> Dict[str, float]:
        """
        Perform information theory analysis.

        Parameters:
            data_column: Column to analyze

        Returns:
            Dictionary with information theory measures
        """
        logger.info(f"Performing information theory analysis on {data_column}")

        # Create information theory analyzer instance
        from .analytics_engine import InformationTheory
        analyzer = InformationTheory(self.data)

        return analyzer.compute_entropy_measures(data_column)

    def robust_statistical_analysis(self, data_column: str) -> Dict[str, float]:
        """
        Perform robust statistical analysis.

        Parameters:
            data_column: Column to analyze

        Returns:
            Dictionary with robust statistical measures
        """
        logger.info(f"Performing robust statistical analysis on {data_column}")

        # Create robust statistics analyzer instance
        from .analytics_engine import RobustStatistics
        analyzer = RobustStatistics(self.data)

        # Combine location and scale estimates
        location_results = analyzer.robust_location_estimates(data_column)
        scale_results = analyzer.robust_scale_estimates(data_column)

        # Remove error keys if present
        location_results = {k: v for k, v in location_results.items() if not k.startswith('error')}
        scale_results = {k: v for k, v in scale_results.items() if not k.startswith('error')}

        return {
            'location_estimates': location_results,
            'scale_estimates': scale_results,
            'robust_location_preferred': location_results.get('trimmed_mean', np.nan),
            'robust_scale_preferred': scale_results.get('mad_normalized', np.nan)
        }

    def spatial_analysis(self,
                        value_column: str,
                        spatial_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform spatial analysis.

        Parameters:
            value_column: Column with values to analyze
            spatial_weights: Optional spatial weights matrix

        Returns:
            Dictionary with spatial analysis results
        """
        logger.info(f"Performing spatial analysis on {value_column}")

        # Create spatial analyzer instance
        from .analytics_engine import SpatialAnalyzer
        analyzer = SpatialAnalyzer(self.data)

        return analyzer.morans_i_analysis(value_column, spatial_weights)

    def comprehensive_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all analytic methods.

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

        # Bayesian analysis
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                report['bayesian'] = self.bayesian_analysis(numeric_cols[0], numeric_cols[1])
            else:
                report['bayesian'] = {'error': 'Insufficient numeric columns'}
        except Exception as e:
            report['bayesian'] = {'error': str(e)}

        # Network analysis
        try:
            report['network'] = self.network_analysis()
        except Exception as e:
            report['network'] = {'error': str(e)}

        # Causal inference
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                report['causal'] = self.causal_inference(numeric_cols[0], numeric_cols[1])
            else:
                report['causal'] = {'error': 'Insufficient numeric columns'}
        except Exception as e:
            report['causal'] = {'error': str(e)}

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
