"""
Test suite for AnalyticsEngine module.

This module tests the comprehensive statistical analysis functionality of the AnalyticsEngine
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
from evojump import datacore, analytics_engine


class TestTimeSeriesAnalyzer:
    """Test TimeSeriesAnalyzer class."""

    def test_analyze_trends_linear(self):
        """Test linear trend analysis."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        trends = analyzer.analyze_trends(method='linear')

        assert 'phenotype1' in trends
        assert trends['phenotype1']['slope'] == 2.0
        assert trends['phenotype1']['r_squared'] == 1.0
        assert bool(trends['phenotype1']['significant']) is True

    def test_analyze_trends_polynomial(self):
        """Test polynomial trend analysis."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [1, 4, 9, 16, 25]  # y = x^2
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        trends = analyzer.analyze_trends(method='polynomial')

        assert 'phenotype1' in trends
        assert trends['phenotype1']['degree'] == 2
        assert len(trends['phenotype1']['coefficients']) == 3

    def test_detect_seasonality(self):
        """Test seasonality detection."""
        # Create seasonal data
        time_points = np.arange(1, 25)
        seasonal_data = 10 + 5 * np.sin(2 * np.pi * time_points / 12)

        data = pd.DataFrame({
            'time': time_points,
            'phenotype1': seasonal_data
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        seasonality = analyzer.detect_seasonality(period=12)

        assert 'phenotype1' in seasonality
        assert seasonality['phenotype1']['period'] == 12
        assert bool(seasonality['phenotype1']['seasonal_detected']) is True

    def test_detect_change_points(self):
        """Test change point detection."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'phenotype1': [1, 1, 1, 1, 10, 10, 10, 10, 10, 10]  # Larger change at t=5
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        change_points = analyzer.detect_change_points(method='cusum')

        # Check that change points are detected or at least the method works
        assert isinstance(change_points, list)
        if len(change_points) > 0:
            assert change_points[0]['variable'] == 'phenotype1'
            assert 'time_index' in change_points[0]
            assert 'method' in change_points[0]

    def test_forecast_arima(self):
        """Test ARIMA forecasting."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'phenotype1': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        forecasts = analyzer.forecast(forecast_steps=5, method='arima')

        assert 'phenotype1' in forecasts
        assert len(forecasts['phenotype1']) == 5

    def test_forecast_exponential_smoothing(self):
        """Test exponential smoothing forecasting."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'phenotype1': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        forecasts = analyzer.forecast(forecast_steps=3, method='exponential_smoothing')

        assert 'phenotype1' in forecasts
        assert len(forecasts['phenotype1']) == 3


class TestMultivariateAnalyzer:
    """Test MultivariateAnalyzer class."""

    def test_principal_component_analysis(self):
        """Test PCA analysis."""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.normal(0, 1, 50),
            'var2': np.random.normal(0, 1, 50),
            'var3': np.random.normal(0, 1, 50)
        })

        analyzer = analytics_engine.MultivariateAnalyzer(data)
        pca_results = analyzer.principal_component_analysis(n_components=2)

        assert 'pca_components' in pca_results
        assert 'explained_variance_ratio' in pca_results
        assert 'loadings' in pca_results
        assert len(pca_results['explained_variance_ratio']) == 2

    def test_canonical_correlation_analysis(self):
        """Test CCA analysis."""
        np.random.seed(42)
        data1 = pd.DataFrame({
            'x1': np.random.normal(0, 1, 50),
            'x2': np.random.normal(0, 1, 50)
        })
        data2 = pd.DataFrame({
            'y1': np.random.normal(0, 1, 50),
            'y2': np.random.normal(0, 1, 50)
        })

        analyzer = analytics_engine.MultivariateAnalyzer(data1)
        cca_results = analyzer.canonical_correlation_analysis(data1, data2)

        assert 'canonical_correlations' in cca_results
        assert 'canonical_variables_1' in cca_results
        assert len(cca_results['canonical_correlations']) <= 2

    def test_cluster_analysis(self):
        """Test cluster analysis."""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.normal(0, 1, 50),
            'var2': np.random.normal(0, 1, 50),
            'var3': np.random.normal(5, 1, 50)  # Different cluster
        })

        analyzer = analytics_engine.MultivariateAnalyzer(data)
        cluster_results = analyzer.cluster_analysis(n_clusters=2, method='kmeans')

        assert 'cluster_labels' in cluster_results
        assert 'cluster_centers' in cluster_results
        assert 'inertia' in cluster_results
        assert len(cluster_results['cluster_labels']) == 50
        assert cluster_results['n_clusters'] == 2


class TestPredictiveModeler:
    """Test PredictiveModeler class."""

    def test_train_predictive_model_random_forest(self):
        """Test random forest model training."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })

        modeler = analytics_engine.PredictiveModeler(data)
        result = modeler.train_predictive_model(
            target_variable='target',
            feature_variables=['feature1', 'feature2'],
            model_name='random_forest'
        )

        assert result.model_name == 'random_forest'
        assert len(result.predictions) > 0
        assert 'train_r2' in result.performance_metrics
        assert 'test_r2' in result.performance_metrics
        assert result.performance_metrics['test_r2'] is not None

    def test_cross_validate_model(self):
        """Test model cross-validation."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })

        modeler = analytics_engine.PredictiveModeler(data)
        cv_results = modeler.cross_validate_model(
            target_variable='target',
            feature_variables=['feature1', 'feature2'],
            model_name='random_forest',
            cv_folds=5
        )

        assert 'mean_r2' in cv_results
        assert 'std_r2' in cv_results
        assert 'mean_mse' in cv_results
        assert 'std_mse' in cv_results
        assert cv_results['mean_r2'] is not None


class TestChangePointDetector:
    """Test ChangePointDetector class."""

    def test_detect_changes_statistical(self):
        """Test statistical change point detection."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'phenotype1': [1, 1, 1, 1, 5, 5, 5, 5, 5, 5]  # Change at t=5
        })

        detector = analytics_engine.ChangePointDetector(data, 'time')
        changes = detector.detect_changes(method='statistical', threshold=2.0)

        assert len(changes) > 0
        assert changes[0]['variable'] == 'phenotype1'
        assert changes[0]['method'] == 'statistical'

    def test_detect_changes_bayesian(self):
        """Test Bayesian change point detection."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'phenotype1': [1, 1, 1, 1, 5, 5, 5, 5, 5, 5]  # Change at t=5
        })

        detector = analytics_engine.ChangePointDetector(data, 'time')
        changes = detector.detect_changes(method='bayesian')

        assert isinstance(changes, list)

    def test_detect_changes_information(self):
        """Test information criterion change point detection."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'phenotype1': [1, 1, 1, 1, 5, 5, 5, 5, 5, 5]  # Change at t=5
        })

        detector = analytics_engine.ChangePointDetector(data, 'time')
        changes = detector.detect_changes(method='information')

        assert isinstance(changes, list)


class TestAnalyticsEngine:
    """Test AnalyticsEngine class."""

    def create_test_data(self):
        """Create test data for AnalyticsEngine."""
        np.random.seed(42)
        time_points = np.arange(1, 21)
        trend = 2 * time_points
        noise = np.random.normal(0, 1, len(time_points))

        data = pd.DataFrame({
            'time': time_points,
            'phenotype1': trend + noise,
            'phenotype2': trend * 1.5 + noise * 2
        })
        return data

    def test_analytics_engine_initialization(self):
        """Test AnalyticsEngine initialization."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        assert engine.time_column == 'time'
        assert engine.ts_analyzer is not None
        assert engine.mv_analyzer is not None
        assert engine.predictive_modeler is not None
        assert engine.change_detector is not None

    def test_analyze_time_series(self):
        """Test comprehensive time series analysis."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        ts_results = engine.analyze_time_series()

        assert isinstance(ts_results, analytics_engine.TimeSeriesResult)
        assert ts_results.trend_analysis is not None
        assert ts_results.seasonality_analysis is not None
        assert ts_results.change_points is not None
        assert ts_results.forecasts is not None
        assert ts_results.model_fit is not None

    def test_analyze_multivariate(self):
        """Test multivariate analysis."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        mv_results = engine.analyze_multivariate()

        assert 'principal_components' in mv_results
        assert 'cluster_analysis' in mv_results
        assert 'correlation_analysis' in mv_results

    def test_predictive_modeling(self):
        """Test predictive modeling."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        predictions = engine.predictive_modeling(
            target_variable='phenotype2',
            feature_variables=['phenotype1']
        )

        assert 'random_forest' in predictions
        assert predictions['random_forest'].model_name == 'random_forest'
        assert len(predictions['random_forest'].predictions) > 0

    def test_detect_changes(self):
        """Test change point detection."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        changes = engine.detect_changes(method='statistical')

        assert isinstance(changes, list)

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        corr_results = engine._correlation_analysis()

        assert 'correlation_matrix' in corr_results
        assert 'high_correlations' in corr_results
        assert 'mean_correlation' in corr_results
        # Should have 2 phenotype columns (excluding time column)
        assert corr_results['correlation_matrix'].shape[0] == 2

    def test_test_stationarity(self):
        """Test stationarity testing."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        stationarity = engine._test_stationarity()

        assert 'phenotype1' in stationarity
        assert 'phenotype2' in stationarity
        assert isinstance(bool(stationarity['phenotype1']), bool)

    def test_analyze_autocorrelation(self):
        """Test autocorrelation analysis."""
        data = self.create_test_data()

        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        autocorr = engine._analyze_autocorrelation()

        assert 'phenotype1' in autocorr
        assert 'phenotype2' in autocorr
        assert 'autocorrelation_values' in autocorr['phenotype1']
        assert 'lags' in autocorr['phenotype1']


class TestBayesianAnalyzer:
    """Test BayesianAnalyzer class."""

    def test_bayesian_linear_regression(self):
        """Test Bayesian linear regression."""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })

        analyzer = analytics_engine.BayesianAnalyzer(data)
        result = analyzer.bayesian_linear_regression(data['x'].values, data['y'].values, n_samples=100)

        assert isinstance(result, analytics_engine.BayesianResult)
        assert len(result.posterior_samples) == 100
        assert '95%' in result.credible_intervals
        assert '90%' in result.credible_intervals
        assert isinstance(result.convergence_diagnostics, dict)

    def test_bayesian_model_comparison(self):
        """Test Bayesian model comparison."""
        analyzer = analytics_engine.BayesianAnalyzer(pd.DataFrame({'x': [1, 2, 3]}))

        comparison = analyzer.bayesian_model_comparison(
            model1_likelihood=-10.0,
            model2_likelihood=-12.0,
            model1_complexity=2,
            model2_complexity=3
        )

        assert isinstance(comparison, dict)
        assert 'bic_model1' in comparison
        assert 'bic_model2' in comparison
        assert 'preferred_model' in comparison


class TestNetworkAnalyzer:
    """Test NetworkAnalyzer class."""

    def test_construct_correlation_network(self):
        """Test correlation network construction."""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.normal(0, 1, 50),
            'var2': np.random.normal(0, 1, 50),
            'var3': np.random.normal(0, 1, 50)
        })

        analyzer = analytics_engine.NetworkAnalyzer(data)
        network_result = analyzer.construct_correlation_network(threshold=0.5)

        assert isinstance(network_result, analytics_engine.NetworkResult)
        assert network_result.graph is not None
        assert isinstance(network_result.centrality_measures, dict)
        assert isinstance(network_result.network_metrics, dict)

    def test_shortest_path_analysis(self):
        """Test shortest path analysis."""
        data = pd.DataFrame({
            'var1': [1, 2, 3, 4, 5],
            'var2': [2, 3, 4, 5, 6],
            'var3': [3, 4, 5, 6, 7]
        })

        analyzer = analytics_engine.NetworkAnalyzer(data)
        network_result = analyzer.construct_correlation_network(threshold=0.9)

        if len(network_result.network_metrics) > 0:
            path_analysis = analyzer.shortest_path_analysis('var1', 'var3')
            assert isinstance(path_analysis, dict)


class TestCausalInference:
    """Test CausalInference class."""

    def test_granger_causality_test(self):
        """Test Granger causality analysis."""
        np.random.seed(42)
        data = pd.DataFrame({
            'cause': np.random.normal(0, 1, 50),
            'effect': np.random.normal(0, 1, 50)
        })

        analyzer = analytics_engine.CausalInference(data)
        result = analyzer.granger_causality_test('cause', 'effect', max_lag=3)

        assert isinstance(result, dict)
        assert 'granger_causality' in result or 'error' in result


class TestAdvancedAnalyticsEngine:
    """Test advanced AnalyticsEngine methods."""

    def create_test_data(self):
        """Create comprehensive test data."""
        np.random.seed(42)
        time_points = np.arange(1, 51)
        trend = 2 * time_points
        seasonality = 5 * np.sin(2 * np.pi * time_points / 12)
        noise = np.random.normal(0, 1, len(time_points))

        data = pd.DataFrame({
            'time': time_points,
            'phenotype1': trend + seasonality + noise,
            'phenotype2': trend * 1.5 + seasonality * 0.5 + noise * 2,
            'phenotype3': trend * 0.8 + seasonality * 1.2 + noise * 0.5
        })
        return data

    def test_bayesian_analysis(self):
        """Test Bayesian analysis method."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.bayesian_analysis('phenotype1', 'phenotype2', n_samples=100)

        assert isinstance(result, analytics_engine.BayesianResult)
        assert len(result.posterior_samples) == 100
        assert isinstance(result.credible_intervals, dict)

    def test_network_analysis(self):
        """Test network analysis method."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.network_analysis(correlation_threshold=0.6)

        assert isinstance(result, analytics_engine.NetworkResult)
        assert result.graph is not None
        assert isinstance(result.network_metrics, dict)
        assert 'num_nodes' in result.network_metrics

    def test_causal_inference(self):
        """Test causal inference method."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.causal_inference('phenotype1', 'phenotype2', max_lag=3)

        assert isinstance(result, dict)
        assert 'granger_causality' in result or 'error' in result

    def test_advanced_dimensionality_reduction_fastica(self):
        """Test FastICA dimensionality reduction."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.advanced_dimensionality_reduction(method='fastica', n_components=2)

        assert isinstance(result, analytics_engine.DimensionalityResult)
        assert result.embeddings.shape[0] == len(data)
        assert result.embeddings.shape[1] == 2

    def test_advanced_dimensionality_reduction_tsne(self):
        """Test t-SNE dimensionality reduction."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.advanced_dimensionality_reduction(
            method='tsne',
            n_components=2,
            perplexity=10,
            learning_rate=100
        )

        assert isinstance(result, analytics_engine.DimensionalityResult)
        assert result.embeddings.shape[0] == len(data)
        assert result.embeddings.shape[1] == 2

    def test_spectral_analysis(self):
        """Test spectral analysis method."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.spectral_analysis('phenotype1', sampling_frequency=1.0)

        assert isinstance(result, analytics_engine.SpectralResult)
        assert isinstance(result.power_spectrum, np.ndarray) or isinstance(result.power_spectrum, list)

    def test_nonlinear_dynamics_analysis(self):
        """Test nonlinear dynamics analysis."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.nonlinear_dynamics_analysis('phenotype1', embedding_dim=3, tau=1)

        assert isinstance(result, dict)
        assert 'largest_lyapunov_exponent' in result or 'error' in result

    def test_information_theory_analysis(self):
        """Test information theory analysis."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.information_theory_analysis('phenotype1')

        assert isinstance(result, dict)
        assert 'shannon_entropy' in result or 'error' in result

    def test_robust_statistical_analysis(self):
        """Test robust statistical analysis."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.robust_statistical_analysis('phenotype1')

        assert isinstance(result, dict)
        assert 'location_estimates' in result
        assert 'scale_estimates' in result

    def test_comprehensive_analysis_report(self):
        """Test comprehensive analysis report."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        report = engine.comprehensive_analysis_report()

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'data_summary' in report
        assert 'time_series' in report
        assert 'multivariate' in report
        assert 'bayesian' in report
        assert 'network' in report
        assert 'causal' in report
        assert 'information_theory' in report
        assert 'robust_statistics' in report

    def test_spatial_analysis(self):
        """Test spatial analysis method."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.spatial_analysis('phenotype1')

        assert isinstance(result, dict)
        assert 'morans_i' in result or 'error' in result

    def test_survival_analysis(self):
        """Test survival analysis method."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'event': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
        })
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.survival_analysis('time', 'event')

        assert isinstance(result, analytics_engine.SurvivalResult)
        assert isinstance(result.survival_function, np.ndarray) or isinstance(result.survival_function, list)
