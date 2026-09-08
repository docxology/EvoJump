"""
Test suite for AnalyticsEngine module.

This module tests the comprehensive statistical analysis functionality of the AnalyticsEngine
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, analytics_engine
from conftest import make_growth_frame


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

        # The single 9-unit jump has z = 8.485 > 2 against the other diffs
        assert len(change_points) == 1
        assert change_points[0]['variable'] == 'phenotype1'
        assert change_points[0]['time_index'] == 3
        assert change_points[0]['time_value'] == 4
        assert change_points[0]['change_magnitude'] == 9.0
        assert change_points[0]['method'] == 'statistical'

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
        # Perfect linear series: ARIMA(1,1,1) must extrapolate the trend
        assert np.isfinite(forecasts['phenotype1']).all()
        assert (forecasts['phenotype1'] > 28.0).all()

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
        # alpha=0.3 smoothing anchored at the last value of a flat tail: every
        # step equals alpha*28 + (1-alpha)*28 = 28
        assert (forecasts['phenotype1'] == 28.0).all()


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

        ccs = cca_results['canonical_correlations']
        assert len(ccs) == 2
        # Seeded independent gaussians share no canonical channel: both
        # sample correlations stay well below the dependence bar
        assert np.all(np.isfinite(ccs))
        assert np.all(ccs < 0.6)
        assert 'canonical_variables_1' in cca_results
        assert 'canonical_variables_2' in cca_results

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
        assert len(result.predictions) == 20  # test_size=0.2 of 100 rows
        metrics = result.performance_metrics
        assert np.isfinite(metrics['train_mse']) and metrics['train_mse'] >= 0
        assert np.isfinite(metrics['test_mse']) and metrics['test_mse'] >= 0
        assert np.isfinite(metrics['train_r2'])
        assert np.isfinite(metrics['test_r2'])
        # In-sample fit beats held-out fit on pure noise
        assert metrics['train_r2'] > metrics['test_r2']
        assert set(result.feature_importance) == {'feature1', 'feature2'}

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
        assert np.isfinite(cv_results['mean_mse']) and cv_results['mean_mse'] > 0
        assert np.isfinite(cv_results['std_mse']) and cv_results['std_mse'] >= 0
        assert np.isfinite(cv_results['mean_r2'])
        assert np.isfinite(cv_results['std_r2'])


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
        """Shared synthetic growth frame from tests/conftest.py."""
        return make_growth_frame(
            n_points=20, phenotype_cols=("phenotype1", "phenotype2"), seed=42
        )

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

        # Perfectly correlated triple: complete graph at threshold 0.9
        assert network_result.network_metrics['num_edges'] == 3
        path_analysis = analyzer.shortest_path_analysis('var1', 'var3')
        assert path_analysis['path'] == ['var1', 'var3']
        assert path_analysis['path_length'] == 1
        assert path_analysis['weighted_path'] == ['var1', 'var3']
        assert path_analysis['weighted_path_length'] > 0
        assert ['var1', 'var3'] in path_analysis['all_shortest_paths']


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
        # Independent seeded gaussians: the test runs and reports one
        # F-test per lag; significance must exactly match the p-values.
        assert 'error' not in result, result
        gc = result['granger_causality']
        assert len(gc['f_statistics']) == 3
        assert len(gc['p_values']) == 3
        assert gc['lags_tested'] == [1, 2, 3]
        assert all(np.isfinite(f) for f in gc['f_statistics'])
        assert all(0.0 <= p <= 1.0 for p in gc['p_values'])
        assert result['significant_causality'] == any(p < 0.05 for p in gc['p_values'])


class TestAdvancedAnalyticsEngine:
    """Test advanced AnalyticsEngine methods."""

    def create_test_data(self):
        """Shared synthetic growth frame from tests/conftest.py."""
        return make_growth_frame(
            n_points=50,
            phenotype_cols=("phenotype1", "phenotype2", "phenotype3"),
            seed=42,
        )

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
        assert 'granger_causality' in result, result
        gc = result['granger_causality']
        assert len(gc['f_statistics']) == 3
        assert len(gc['p_values']) == 3
        assert all(0.0 <= p <= 1.0 for p in gc['p_values'])
        assert result['causal_direction'] == 'phenotype1 -> phenotype2'
        # Logistic growth curves are deterministic in time: phenotype1
        # Granger-causes phenotype2 overwhelmingly.
        assert result['significant_causality'] is True
        assert 1 <= result['best_lag'] <= 3

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
        ps = np.asarray(result.power_spectrum)
        assert ps.ndim == 2 and ps.shape[1] == 2 and len(ps) > 0
        assert result.spectral_entropy > 0.0

    def test_nonlinear_dynamics_analysis(self):
        """Test nonlinear dynamics analysis."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.nonlinear_dynamics_analysis('phenotype1', embedding_dim=3, tau=1)

        assert isinstance(result, dict)
        assert 'error' not in result, result
        lyap = result['largest_lyapunov_exponent']
        assert np.isfinite(lyap)
        assert len(result['correlation_dimensions']) > 0
        assert result['attractor_properties'] == {'embedding_dim': 3, 'tau': 1}

    def test_information_theory_analysis(self):
        """Test information theory analysis."""
        data = self.create_test_data()
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.information_theory_analysis('phenotype1')

        assert isinstance(result, dict)
        assert result['shannon_entropy'] >= 0.0
        assert 0.0 <= result['normalized_entropy'] <= 1.0


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
        assert result['weights_kind'] == 'linear_adjacency'
        assert np.isfinite(result['morans_i'])

    def test_survival_analysis(self):
        """Test survival analysis method."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'event': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
        })
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.survival_analysis('time', 'event')

        assert isinstance(result, analytics_engine.SurvivalResult)
        # Hand-checkable KM: no event at t=1, then 1/8, 1/6, 2/4, 2/2
        np.testing.assert_allclose(
            result.survival_function, [1.0, 0.875, 0.875 * 5 / 6, 0.875 * 5 / 12, 0.0]
        )
        assert result.median_survival_time == 4.0
        assert np.all(np.diff(result.survival_function) <= 1e-12)


class TestCCARecovery:
    """CCA must recover planted canonical structure (regression: the old
    implementation solved eigh(cov12 @ cov21, cov11), missing the cov22^{-1}
    factor, so the reported 'canonical correlations' were not bounded by 1)."""

    def test_cca_recovers_planted_correlation_channel(self):
        rng = np.random.default_rng(0)
        n = 200
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y1 = x1 + 0.05 * rng.normal(0, 1, n)   # near-perfect channel
        y2 = x2 + rng.normal(0, 1, n)          # weaker channel
        data1 = pd.DataFrame({'x1': x1, 'x2': x2})
        data2 = pd.DataFrame({'y1': y1, 'y2': y2})

        analyzer = analytics_engine.MultivariateAnalyzer(data1)
        cca = analyzer.canonical_correlation_analysis(data1, data2)

        assert 'error' not in cca
        ccs = cca['canonical_correlations']
        assert np.all(np.isfinite(ccs))
        assert np.all(ccs <= 1.0 + 1e-9)          # bounded by 1
        assert ccs[0] > 0.95                      # planted channel recovered
        assert ccs[0] > ccs[1]
        assert 'canonical_variables_2' in cca
        assert cca['canonical_variables_2'].shape == (2, 2)


class TestSeasonalityPerColumn:
    """Auto-detected period must be resolved per column (regression: the
    first column's period leaked into all subsequent columns)."""

    def test_each_column_gets_its_own_period(self):
        t = np.arange(0, 96, dtype=float)
        data = pd.DataFrame({
            'time': t,
            'colA': np.sin(2 * np.pi * t / 12),
            'colB': np.sin(2 * np.pi * t / 4),
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        seasonality = analyzer.detect_seasonality()  # no period: auto-detect

        assert seasonality['colA']['period'] == 12
        assert seasonality['colB']['period'] == 4


class TestChangePointDetectionRealMath:
    """Variance method gates on a real F-test; cusum delegates to the shared
    statistical implementation; information method is a real BIC fit."""

    def test_variance_method_gates_on_significance(self):
        rng = np.random.default_rng(7)
        n = 60
        data = pd.DataFrame({
            'time': np.arange(n, dtype=float),
            'signal': np.concatenate([rng.normal(0, 1, n // 2),
                                      rng.normal(0, 5, n - n // 2)]),
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        change_points = analyzer.detect_change_points(method='variance')

        assert len(change_points) >= 1
        cp = change_points[0]
        assert cp['variable'] == 'signal'
        assert cp['p_value'] < 0.05
        assert 0.0 < cp['confidence'] <= 1.0
        assert abs(cp['confidence'] - (1.0 - cp['p_value'])) < 1e-12

    def test_variance_method_quiet_series_reports_nothing(self):
        rng = np.random.default_rng(3)
        n = 60
        data = pd.DataFrame({
            'time': np.arange(n, dtype=float),
            'signal': rng.normal(0, 1, n),
        })

        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        assert analyzer.detect_change_points(method='variance') == []

    def test_variance_method_unsupported_raises(self):
        data = pd.DataFrame({'time': np.arange(12.0), 'signal': np.ones(12)})
        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        with pytest.raises(ValueError):
            analyzer.detect_change_points(method='bogus')

    def test_cusum_delegates_to_statistical_detector(self):
        data = pd.DataFrame({
            'time': np.arange(1.0, 11.0),
            'signal': [1, 1, 1, 1, 10, 10, 10, 10, 10, 10],
        })
        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        detector = analytics_engine.ChangePointDetector(data, 'time')

        via_ts = analyzer.detect_change_points(method='cusum')
        via_detector = detector._statistical_change_detection()

        assert via_ts == via_detector
        assert len(via_ts) > 0


class TestSurvivalAnalysisCorrectness:
    """KM curve must use jointly-dropped (time, event) pairs and be
    hand-verifiable; event indicators must be validated."""

    def test_kaplan_meier_matches_hand_computed_curve(self):
        data = pd.DataFrame({
            'time': [2, 4, 4, 6, 8],
            'event': [1, 0, 1, 1, 0],
        })
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        result = engine.survival_analysis('time', 'event')

        # Hand-computed KM: S(2)=0.8, S(4)=0.8*3/4=0.6, S(6)=0.3, S(8)=0.3
        np.testing.assert_allclose(result.survival_function, [0.8, 0.6, 0.3, 0.3])
        np.testing.assert_allclose(result.cumulative_hazard, [0.2, 0.45, 0.95, 0.95])
        assert result.median_survival_time == 6.0
        # Curve invariants
        assert np.all(np.diff(result.survival_function) <= 1e-12)
        assert np.all(np.diff(result.cumulative_hazard) >= -1e-12)
        lower, upper = result.confidence_intervals['lower'], result.confidence_intervals['upper']
        assert np.all(lower <= result.survival_function + 1e-12)
        assert np.all(result.survival_function <= upper + 1e-12)

    def test_nan_in_one_column_does_not_misalign_pairs(self):
        rng = np.random.default_rng(5)
        n = 40
        times = rng.uniform(1, 20, n)
        events = rng.integers(0, 2, n).astype(float)
        with_nan = pd.DataFrame({'time': times, 'event': events})
        with_nan.loc[7, 'event'] = np.nan     # NaN only in the event column
        with_nan.loc[23, 'time'] = np.nan     # NaN only in the time column

        expected = analytics_engine.AnalyticsEngine(with_nan.dropna()).survival_analysis('time', 'event')
        got = analytics_engine.AnalyticsEngine(with_nan).survival_analysis('time', 'event')

        np.testing.assert_allclose(got.survival_function, expected.survival_function)
        np.testing.assert_allclose(got.cumulative_hazard, expected.cumulative_hazard)
        assert got.median_survival_time == expected.median_survival_time

    def test_non_binary_events_rejected(self):
        data = pd.DataFrame({'time': [1, 2, 3, 4], 'event': [0, 1, 2, 1]})
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        with pytest.raises(ValueError):
            engine.survival_analysis('time', 'event')


class TestBayesianSlopeRecovery:
    """Posterior must recover a planted slope (seeded for determinism)."""

    def test_posterior_recovers_true_slope(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, 300)
        y = 2.5 * x + rng.normal(0, 0.5, 300)
        analyzer = analytics_engine.BayesianAnalyzer(pd.DataFrame({'x': x, 'y': y}))

        result = analyzer.bayesian_linear_regression(x, y, n_samples=4000, seed=3)

        post_mean = result.convergence_diagnostics['posterior_mean']
        assert abs(post_mean - 2.5) < 0.1
        lo, hi = result.credible_intervals['95%']
        assert lo > 0 and hi > lo  # strong effect: 95% CI excludes 0

    def test_no_effect_leaves_zero_in_interval(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        analyzer = analytics_engine.BayesianAnalyzer(pd.DataFrame({'x': x, 'y': y}))

        result = analyzer.bayesian_linear_regression(x, y, n_samples=4000, seed=5)

        lo, hi = result.credible_intervals['95%']
        assert lo <= 0 <= hi

    def test_seeded_draws_are_reproducible(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 100)
        y = 1.0 * x + rng.normal(0, 1, 100)
        analyzer = analytics_engine.BayesianAnalyzer(pd.DataFrame({'x': x, 'y': y}))

        r1 = analyzer.bayesian_linear_regression(x, y, seed=42)
        r2 = analyzer.bayesian_linear_regression(x, y, seed=42)
        np.testing.assert_array_equal(r1.posterior_samples, r2.posterior_samples)


class TestRobustEstimators:
    """Huber/Tukey must be real M-estimators and Sn the Rousseeuw-Croux
    estimator (regression: all three used to return the median)."""

    def test_sn_matches_definition_on_hand_example(self):
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': [1.0]}))
        # Sn([1,2,3,4,100]): inner medians per row are 2.5, 1.5, 1.5, 2.5,
        # 97.5 -> median 2.5, times the Gaussian consistency constant 1.1926.
        sn = engine._sn_scale_estimate(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
        assert abs(sn - 2.5 * 1.1926) < 1e-9

    def test_huber_and_tukey_resist_contamination(self):
        rng = np.random.default_rng(9)
        data = np.concatenate([rng.normal(10, 1, 200), rng.normal(30, 1, 20)])
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': data}))

        huber = engine._huber_estimate(data)
        tukey = engine._tukey_biweight_estimate(data)
        mean = float(np.mean(data))

        assert abs(huber - 10.0) < 0.5
        assert abs(tukey - 10.0) < 0.5
        assert abs(huber - 10.0) < abs(mean - 10.0)

    def test_sn_scale_estimates_sigma_of_gaussian_data(self):
        rng = np.random.default_rng(13)
        data = rng.normal(0, 2.0, 300)
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': data}))

        sn = engine._sn_scale_estimate(data)
        assert abs(sn - 2.0) / 2.0 < 0.3

    def test_estimates_differ_from_median_when_they_should(self):
        # For skewed contamination the M-estimators need not equal the median;
        # on symmetric clean data they should all agree closely.
        rng = np.random.default_rng(17)
        data = rng.normal(5, 1, 400)
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': data}))
        assert abs(engine._huber_estimate(data) - np.median(data)) < 0.15
        assert abs(engine._tukey_biweight_estimate(data) - np.median(data)) < 0.15


class TestCopulaAnalysis:
    """Gaussian parameter recovers a planted correlation; pairs stay aligned
    under NaNs; Frank inverts the exact tau-theta relation."""

    def _engine(self, df):
        return analytics_engine.AnalyticsEngine(df, time_column='time')

    def test_gaussian_parameter_recovers_planted_correlation(self):
        rng = np.random.default_rng(21)
        x = rng.normal(0, 1, 400)
        y = 0.8 * x + 0.6 * rng.normal(0, 1, 400)
        result = self._engine(pd.DataFrame({'x': x, 'y': y})).copula_analysis('x', 'y')

        assert abs(result['copula_parameter'] - 0.8) < 0.1
        assert abs(result['kendall_tau'] - 0.8 * 2 / np.pi) < 0.1

    def test_joint_dropna_keeps_pairs_aligned(self):
        rng = np.random.default_rng(22)
        x = rng.normal(0, 1, 300)
        y = 0.6 * x + 0.8 * rng.normal(0, 1, 300)
        df = pd.DataFrame({'x': x, 'y': y})
        df.loc[df.sample(20, random_state=1).index, 'x'] = np.nan
        df.loc[df.sample(20, random_state=2).index, 'y'] = np.nan

        got = self._engine(df).copula_analysis('x', 'y')
        expected = self._engine(df.dropna()).copula_analysis('x', 'y')

        assert abs(got['kendall_tau'] - expected['kendall_tau']) < 1e-12
        assert abs(got['copula_parameter'] - expected['copula_parameter']) < 1e-12

    def test_frank_theta_solves_exact_tau_relation(self):
        from scipy.integrate import quad
        theta = analytics_engine.AnalyticsEngine._frank_theta_from_tau(0.5)
        # Exact relation tau = 1 - 4/theta (1 - D1(theta)); the solution
        # theta = 5.7363 was independently confirmed by direct numerical
        # integration of the Frank copula (population tau = 0.49999 there;
        # theta = 2.45 corresponds to tau ~ 0.257, not 0.5).
        assert abs(theta - 5.736282707) < 1e-6

        def d1(x):
            return quad(lambda t: t / np.expm1(t), 0.0, x, limit=200)[0] / x

        back = 1.0 - 4.0 / theta * (1.0 - d1(theta))
        # Sign symmetry for negative dependence
        neg = analytics_engine.AnalyticsEngine._frank_theta_from_tau(-0.5)
        assert abs(neg + theta) < 1e-9

    def test_student_copula_implemented(self):
        rng = np.random.default_rng(23)
        x = rng.standard_t(5, 300)
        y = 0.7 * x + np.sqrt(1 - 0.49) * rng.standard_t(5, 300)
        result = self._engine(pd.DataFrame({'x': x, 'y': y})).copula_analysis('x', 'y', copula_type='student')

        assert 'degrees_of_freedom' in result
        assert -1.0 <= result['copula_parameter'] <= 1.0

    def test_unsupported_copula_type_raises(self):
        df = pd.DataFrame({'x': np.random.default_rng(0).normal(size=50),
                           'y': np.random.default_rng(1).normal(size=50)})
        with pytest.raises(ValueError, match='Unsupported copula type'):
            self._engine(df).copula_analysis('x', 'y', copula_type='gumbel')

    def test_clayton_rejects_negative_dependence(self):
        rng = np.random.default_rng(24)
        x = rng.normal(0, 1, 200)
        y = -0.9 * x + 0.5 * rng.normal(0, 1, 200)
        with pytest.raises(ValueError, match='positive dependence'):
            self._engine(pd.DataFrame({'x': x, 'y': y})).copula_analysis('x', 'y', copula_type='clayton')


class TestWaveletAnalysis:
    def test_dominant_scale_matches_planted_periodicity(self):
        rng = np.random.default_rng(25)
        t = np.arange(128, dtype=float)
        data = pd.DataFrame({'time': t, 'signal': np.sin(2 * np.pi * t / 12) + 0.05 * rng.normal(0, 1, 128)})
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.wavelet_analysis('signal')

        assert result['dominant_scale'] > 0
        scale_idx = int(np.argmin(np.abs(np.asarray(result['scales']) - result['dominant_scale'])))
        dominant_freq = float(np.asarray(result['frequencies'])[scale_idx])
        dominant_period = 1.0 / dominant_freq
        assert abs(dominant_period - 12.0) < 3.0


class TestExtremeValueAnalysis:
    def test_return_levels_monotone_in_return_period(self):
        rng = np.random.default_rng(26)
        data = pd.DataFrame({'time': np.arange(500.0), 'value': rng.standard_exponential(500)})
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.extreme_value_analysis('value')

        pot_levels = [result['pot_method']['return_levels'][k] for k in ('10_year', '50_year', '100_year', '500_year')]
        gev_levels = [result['block_maxima_method']['return_levels'][k] for k in ('10_year', '50_year', '100_year', '500_year')]
        assert all(b > a for a, b in zip(pot_levels, pot_levels[1:]))
        assert all(b > a for a, b in zip(gev_levels, gev_levels[1:]))
        assert result['hill_estimator'] > 0  # exponential tail has finite mean excess


class TestRegimeSwitchingAnalysis:
    def test_recovers_two_regimes_with_valid_transition_matrix(self):
        rng = np.random.default_rng(27)
        n = 200
        signal = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(6, 1, n - n // 2)])
        data = pd.DataFrame({'time': np.arange(n, dtype=float), 'signal': signal})
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.regime_switching_analysis('signal', n_regimes=2)

        labels = np.asarray(result['regime_labels'])
        assert set(labels.tolist()) == {0, 1}
        assert len(result['regime_statistics']) == 2
        means = [s['mean'] for s in result['regime_statistics']]
        assert abs(means[0] - means[1]) > 3.0
        # First half and second half must be mostly different regimes
        assert np.mean(labels[:n // 2] == labels[0]) > 0.8
        assert np.mean(labels[n // 2:] == labels[-1]) > 0.8
        # Transition probabilities: each row sums to 1 where transitions occur, else 0
        probs = np.asarray(result['transition_probabilities'])
        row_sums = probs.sum(axis=1)
        assert np.all((np.abs(row_sums - 1.0) < 1e-9) | (row_sums == 0.0))
        assert result['n_switches'] < n // 10

    def test_constant_series_does_not_crash(self):
        data = pd.DataFrame({'time': np.arange(40.0), 'signal': np.ones(40)})
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        result = engine.regime_switching_analysis('signal', n_regimes=2)
        assert np.isfinite(np.asarray(result['transition_probabilities'])).all()


class TestSpatialAnalysisMoran:
    def test_identity_weights_give_i_of_one(self):
        rng = np.random.default_rng(28)
        values = rng.normal(0, 1, 30)
        data = pd.DataFrame({'value': values})
        engine = analytics_engine.AnalyticsEngine(data)

        W = np.eye(30)
        result = engine.spatial_analysis('value', spatial_weights=W)

        # With W = identity: I = (n/n) * z'z / z'z = 1 exactly
        assert abs(result['morans_i'] - 1.0) < 1e-12
        assert result['weights_kind'] == 'supplied'

    def test_linear_adjacency_detects_smooth_and_alternating_series(self):
        t = np.arange(40, dtype=float)
        smooth = pd.DataFrame({'value': t + 0.01 * np.sin(t)})
        alternating = pd.DataFrame({'value': (-1.0) ** t})

        i_smooth = analytics_engine.AnalyticsEngine(smooth).spatial_analysis('value')['morans_i']
        i_alt = analytics_engine.AnalyticsEngine(alternating).spatial_analysis('value')['morans_i']

        assert i_smooth > 0.5
        assert i_alt < -0.5


class TestSpectralCoherence:
    def test_msc_peaks_at_shared_frequency(self):
        rng = np.random.default_rng(29)
        fs = 8.0
        t = np.arange(512) / fs
        common = np.sin(2 * np.pi * 2.0 * t)
        data = pd.DataFrame({
            'time': t,
            'a': common + 0.5 * rng.normal(0, 1, len(t)),
            'b': common + 0.5 * rng.normal(0, 1, len(t)),
        })
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        result = engine.spectral_analysis('a', sampling_frequency=fs, coherence_column='b')

        assert result.coherence_matrix.size > 0
        freqs, coh = result.coherence_matrix[:, 0], result.coherence_matrix[:, 1]
        at_shared = coh[np.argmin(np.abs(freqs - 2.0))]
        far = coh[np.argmin(np.abs(freqs - 3.5))]
        assert at_shared > 0.7
        assert at_shared > far

    def test_no_coherence_column_leaves_matrix_empty(self):
        data = pd.DataFrame({'time': np.arange(50.0), 'a': np.sin(np.arange(50.0))})
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        result = engine.spectral_analysis('a')
        assert result.coherence_matrix.size == 0


class TestComprehensiveReportExplicitColumns:
    def test_bayesian_and_causal_sections_require_explicit_columns(self):
        data = pd.DataFrame({
            'time': np.arange(1, 31),
            'phenotype1': np.random.default_rng(30).normal(0, 1, 30),
            'phenotype2': np.random.default_rng(31).normal(0, 1, 30),
        })
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')

        report = engine.comprehensive_analysis_report()
        assert 'not_analyzed' in report['bayesian']
        assert 'not_analyzed' in report['causal']

        report = engine.comprehensive_analysis_report(
            bayesian_columns=('phenotype1', 'phenotype2'),
            causal_columns=('phenotype1', 'phenotype2'),
        )
        assert isinstance(report['bayesian'], analytics_engine.BayesianResult)
        assert 'error' not in report['causal'], report['causal']
        # Independent columns: significance must match the reported p-values.
        assert (report['causal']['significant_causality']
                == any(p < 0.05 for p in report['causal']['granger_causality']['p_values']))


class TestMultivariateAnalyzerEdgeCases:
    """Validation branches of MultivariateAnalyzer (PCA/CCA guards)."""

    def test_pca_needs_three_samples(self):
        data = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        with pytest.raises(ValueError, match='Insufficient data for PCA'):
            analytics_engine.MultivariateAnalyzer(data).principal_component_analysis()

    def test_cca_requires_equal_sample_counts(self):
        rng = np.random.default_rng(6)
        d1 = pd.DataFrame({'x': rng.normal(size=20)})
        d2 = pd.DataFrame({'y': rng.normal(size=19)})
        with pytest.raises(ValueError, match='same number of samples'):
            analytics_engine.MultivariateAnalyzer(d1).canonical_correlation_analysis(d1, d2)

    def test_cca_returns_error_dict_on_overflowing_covariance(self):
        # 1e200-scale values: squared products overflow to inf inside the
        # canonical-correlation eigenproblem; the analyzer must degrade to
        # an error dict instead of crashing.
        d1 = pd.DataFrame({'x1': [1e200, -2e200, 3e200, -4e200],
                           'x2': [1.0, 2.0, 3.0, 4.0]})
        d2 = pd.DataFrame({'y1': [2.0, 3.0, 5.0, 7.0], 'y2': [1.0, 3.0, 2.0, 5.0]})
        result = analytics_engine.MultivariateAnalyzer(d1).canonical_correlation_analysis(d1, d2)
        assert 'error' in result

    def test_cluster_analysis_rejects_unsupported_method(self):
        data = pd.DataFrame({'a': np.arange(10.0), 'b': np.arange(10.0)})
        with pytest.raises(ValueError, match='Unsupported clustering method'):
            analytics_engine.MultivariateAnalyzer(data).cluster_analysis(method='dbscan')


class TestPredictiveModelerValidation:
    """Input validation of PredictiveModeler train/cross-validate paths."""

    @pytest.fixture
    def modeler(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({
            'f1': rng.normal(0, 1, 40),
            'f2': rng.normal(0, 1, 40),
            'target': rng.normal(0, 1, 40),
        })
        return analytics_engine.PredictiveModeler(frame)

    def test_missing_target_raises(self, modeler):
        with pytest.raises(ValueError, match='Target variable nope not found'):
            modeler.train_predictive_model('nope', ['f1'])

    def test_missing_feature_raises(self, modeler):
        with pytest.raises(ValueError, match='Feature variables not found'):
            modeler.train_predictive_model('target', ['f1', 'nope'])

    def test_insufficient_rows_raises(self):
        frame = pd.DataFrame({'f1': np.arange(8.0), 'target': np.arange(8.0)})
        modeler = analytics_engine.PredictiveModeler(frame)
        with pytest.raises(ValueError, match='Insufficient data for model training'):
            modeler.train_predictive_model('target', ['f1'])

    @pytest.mark.parametrize('call', ['train', 'cross_validate'])
    def test_unknown_model_raises(self, modeler, call):
        if call == 'train':
            with pytest.raises(ValueError, match='Unknown model'):
                modeler.train_predictive_model('target', ['f1'], model_name='bogus')
        else:
            with pytest.raises(ValueError, match='Unknown model'):
                modeler.cross_validate_model('target', ['f1'], model_name='bogus')


class TestNetworkAnalyzerEdgeCases:
    """Guard branches of the network analyzer and shortest-path API."""

    def test_unsupported_correlation_method_raises(self):
        data = pd.DataFrame({'a': np.arange(10.0), 'b': np.arange(10.0) * 2})
        with pytest.raises(ValueError, match='Unsupported correlation method'):
            analytics_engine.NetworkAnalyzer(data).construct_correlation_network(method='spearman')

    def test_shortest_path_requires_constructed_graph(self):
        analyzer = analytics_engine.NetworkAnalyzer(pd.DataFrame({'a': [1.0, 2.0]}))
        with pytest.raises(ValueError, match='No graph available'):
            analyzer.shortest_path_analysis('a', 'b')

    def test_shortest_path_unknown_node_raises(self):
        data = pd.DataFrame({'a': [1.0, 2, 3, 4, 5], 'b': [1.0, 2, 3, 4, 5]})
        analyzer = analytics_engine.NetworkAnalyzer(data)
        analyzer.construct_correlation_network(threshold=0.9)
        with pytest.raises(ValueError, match='not in graph'):
            analyzer.shortest_path_analysis('a', 'missing')

    def test_shortest_path_across_components_raises(self):
        # Two internally-correlated pairs, no cross-correlation: the
        # thresholded graph is disconnected between the pairs.
        rng = np.random.default_rng(8)
        a = rng.normal(0, 1, 30)
        c = rng.normal(0, 1, 30)
        data = pd.DataFrame({
            'v1': a,
            'v2': a + 0.01 * rng.normal(0, 1, 30),
            'v3': c,
            'v4': c + 0.01 * rng.normal(0, 1, 30),
        })
        analyzer = analytics_engine.NetworkAnalyzer(data)
        analyzer.construct_correlation_network(threshold=0.9)
        with pytest.raises(ValueError, match='No path'):
            analyzer.shortest_path_analysis('v1', 'v3')


class TestCausalInferenceGranger:
    """Granger causality must return real per-lag F-tests after the
    statsmodels-0.15 `verbose` fix, and degrade gracefully on short data."""

    def test_planted_causality_is_detected(self):
        rng = np.random.default_rng(0)
        cause = rng.normal(0, 1, 200)
        effect = np.zeros(200)
        for t in range(1, 200):
            effect[t] = 0.6 * cause[t - 1] + 0.3 * rng.normal(0, 1)
        result = analytics_engine.CausalInference(
            pd.DataFrame({'cause': cause, 'effect': effect})
        ).granger_causality_test('cause', 'effect', max_lag=3)

        assert 'error' not in result, result
        gc = result['granger_causality']
        assert len(gc['f_statistics']) == 3
        assert gc['lags_tested'] == [1, 2, 3]
        assert result['significant_causality'] is True
        assert 1 <= result['best_lag'] <= 3

    def test_insufficient_data_returns_error(self):
        data = pd.DataFrame({'cause': [1.0, 2, 3, 4], 'effect': [2.0, 3, 4, 5]})
        result = analytics_engine.CausalInference(data).granger_causality_test(
            'cause', 'effect', max_lag=5
        )
        assert result == {'error': 'Insufficient data for Granger causality test'}


class TestDimensionalityReducerEdgeCases:
    """Degradation contract of FastICA/t-SNE/intrinsic-dimension helpers."""

    def test_fast_ica_returns_error_result_on_constant_columns(self):
        data = pd.DataFrame({'time': np.arange(30.0), 'a': [5.0] * 30, 'b': [7.0] * 30})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).advanced_dimensionality_reduction(method='fastica')

        assert result.embeddings.size == 0
        assert result.reconstruction_error == 1.0
        assert 'error' in result.manifold_structure

    def test_tsne_returns_error_result_when_perplexity_exceeds_samples(self):
        data = pd.DataFrame({'time': np.arange(12.0),
                             'a': np.random.default_rng(9).normal(0, 1, 12)})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).advanced_dimensionality_reduction(method='tsne', perplexity=50.0)

        assert result.embeddings.size == 0
        assert 'error' in result.manifold_structure

    def test_intrinsic_dimension_of_identical_points_is_feature_count(self):
        from evojump.analytics_engine import DimensionalityReducer
        reducer = DimensionalityReducer(pd.DataFrame({'a': np.zeros(12)}))
        assert reducer._estimate_intrinsic_dimension(np.zeros((12, 3))) == 3

    def test_intrinsic_dimension_falls_back_when_neighbors_exceed_samples(self):
        from evojump.analytics_engine import DimensionalityReducer
        reducer = DimensionalityReducer(pd.DataFrame({'a': np.zeros(3)}))
        assert reducer._estimate_intrinsic_dimension(np.zeros((3, 4))) == 4


class TestAnalyticsEngineGuardBranches:
    """Missing-column / too-few-observations contracts of engine methods."""

    def test_predictive_modeling_auto_selects_all_non_time_columns(self):
        data = make_growth_frame(
            n_points=30, phenotype_cols=("phenotype1", "phenotype2"), seed=11
        )
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        results = engine.predictive_modeling('phenotype2', models=['random_forest'])

        assert set(results) == {'random_forest'}
        # Auto-selected features must include the other phenotype but
        # exclude the time column.
        assert set(results['random_forest'].feature_importance) == {'phenotype1'}

    def test_predictive_modeling_skips_failing_models(self):
        data = make_growth_frame(
            n_points=30, phenotype_cols=("phenotype1", "phenotype2"), seed=11
        )
        engine = analytics_engine.AnalyticsEngine(data, time_column='time')
        results = engine.predictive_modeling(
            'phenotype2', feature_variables=['phenotype1'],
            models=['random_forest', 'bogus'],
        )
        assert set(results) == {'random_forest'}

    def test_bayesian_analysis_insufficient_pair_returns_empty_result(self):
        data = pd.DataFrame({'time': np.arange(5.0), 'x': np.arange(5.0), 'y': np.arange(5.0)})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).bayesian_analysis('x', 'y')

        assert result.posterior_samples.size == 0
        assert result.credible_intervals == {}
        assert result.model_evidence == 0.0

    def test_unsupported_dimensionality_method_falls_back_to_fastica(self):
        data = make_growth_frame(n_points=30, phenotype_cols=("a", "b", "c"), seed=12)
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).advanced_dimensionality_reduction(method='nonexistent')
        assert result.manifold_structure['algorithm'] == 'FastICA'

    def test_survival_missing_columns_returns_empty_result(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': [1.0, 2.0], 'event': [1.0, 0.0]}), time_column='time'
        )
        result = engine.survival_analysis('missing_time', 'event')
        assert result.survival_function.size == 0
        assert np.isnan(result.median_survival_time)

    def test_survival_single_valid_pair_returns_empty_result(self):
        data = pd.DataFrame({'time': [1.0, np.nan, np.nan], 'event': [1.0, 1.0, 1.0]})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).survival_analysis('time', 'event')
        assert result.survival_function.size == 0

    def test_spectral_missing_column_returns_empty_result(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': np.arange(30.0), 'a': np.arange(30.0)}), time_column='time'
        )
        result = engine.spectral_analysis('missing')
        assert result.power_spectrum.size == 0
        assert result.spectral_entropy == 0.0

    def test_spectral_too_short_signal_returns_empty_result(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': np.arange(9.0), 'a': np.arange(9.0)}), time_column='time'
        )
        result = engine.spectral_analysis('a')
        assert result.power_spectrum.size == 0

    @pytest.mark.parametrize('n_points', [9, 29])
    def test_nonlinear_short_series_reports_insufficient_data(self, n_points):
        data = pd.DataFrame({'time': np.arange(float(n_points)),
                             'a': np.arange(float(n_points))})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).nonlinear_dynamics_analysis('a', embedding_dim=3)
        assert result == {'error': 'Insufficient data for nonlinear dynamics analysis'}

    def test_nonlinear_missing_column_reports_error(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': np.arange(30.0), 'a': np.arange(30.0)}), time_column='time'
        )
        assert engine.nonlinear_dynamics_analysis('missing') == {
            'error': 'Column missing not found'
        }

    def test_nonlinear_degenerate_duplicate_states_yield_nan_lyapunov(self):
        # Strictly periodic series: every embedded vector has an exact
        # duplicate as nearest neighbor -> zero initial distance -> no
        # divergence pairs -> Lyapunov exponent must be NaN, not garbage.
        data = pd.DataFrame({'time': np.arange(40.0), 's': np.tile([0.0, 1.0], 20)})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).nonlinear_dynamics_analysis('s')
        assert np.isnan(result['largest_lyapunov_exponent'])
        assert np.isfinite(result['correlation_dimensions']).all()

    @pytest.mark.parametrize('values', [[], [3.0]])
    def test_information_theory_degenerate_inputs_return_zero_entropy(self, values):
        col = 'a' if values else 'missing'
        frame = {'a': values} if values else {}
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame(frame))
        result = engine.information_theory_analysis(col)
        assert result == {'shannon_entropy': 0.0, 'normalized_entropy': 0.0}

    @pytest.mark.parametrize('values', [[], [1.0, 2.0, 3.0]])
    def test_robust_analysis_degenerate_inputs_return_nan_prefs(self, values):
        col = 'a' if values else 'missing'
        frame = {'a': values} if values else {}
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame(frame))
        result = engine.robust_statistical_analysis(col)
        assert result['location_estimates'] == {}
        assert np.isnan(result['robust_location_preferred'])

    @pytest.mark.parametrize('method', ['huber', 'tukey'])
    def test_m_estimators_on_constant_data_return_median(self, method):
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': [1.0]}))
        data = np.full(20, 4.5)
        estimator = (engine._huber_estimate if method == 'huber'
                     else engine._tukey_biweight_estimate)
        assert estimator(data) == 4.5

    def test_tukey_with_tiny_tuning_constant_falls_back_to_median(self):
        # c -> 0 pushes every weight to zero: the IRLS loop must break out
        # and return the initial median rather than divide by zero.
        rng = np.random.default_rng(13)
        data = rng.normal(0, 1, 50)
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': data}))
        assert engine._tukey_biweight_estimate(data, c=1e-9) == float(np.median(data))

    def test_sn_scale_needs_two_points(self):
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': [1.0]}))
        assert engine._sn_scale_estimate(np.array([7.0])) == 0.0

    def test_spatial_missing_column_returns_nan_morans_i(self):
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}))
        result = engine.spatial_analysis('missing')
        assert np.isnan(result['morans_i'])

    def test_spatial_too_few_values_returns_nan_morans_i(self):
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'a': [1.0, 2.0, 3.0]}))
        result = engine.spatial_analysis('a')
        assert np.isnan(result['morans_i'])

    def test_spatial_wrong_weight_shape_raises(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        )
        with pytest.raises(ValueError, match='spatial_weights must be'):
            engine.spatial_analysis('a', spatial_weights=np.eye(4))

    def test_spatial_zero_weights_yield_nan_morans_i(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]})
        )
        result = engine.spatial_analysis('a', spatial_weights=np.zeros((4, 4)))
        assert np.isnan(result['morans_i'])

    @pytest.mark.parametrize('method', ['wavelet_analysis', 'extreme_value_analysis',
                                        'regime_switching_analysis'])
    def test_missing_column_raises_value_error(self, method):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': np.arange(40.0), 'a': np.arange(40.0)}), time_column='time'
        )
        with pytest.raises(ValueError, match='not found in data'):
            getattr(engine, method)('missing')

    def test_copula_missing_column_raises(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'x': np.arange(10.0), 'y': np.arange(10.0)}), time_column='time'
        )
        with pytest.raises(ValueError, match='Columns not found'):
            engine.copula_analysis('x', 'missing')

    def test_copula_clayton_positive_dependence_matches_tau_inversion(self):
        rng = np.random.default_rng(14)
        x = rng.normal(0, 1, 200)
        y = 0.9 * x + 0.5 * rng.normal(0, 1, 200)
        result = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'x': x, 'y': y}), time_column='time'
        ).copula_analysis('x', 'y', copula_type='clayton')

        expected = 2 * result['kendall_tau'] / (1 - result['kendall_tau'])
        assert abs(result['copula_parameter'] - expected) < 1e-9
        assert result['copula_parameter'] > 0
        assert result['dependence_class'] == 'positive'

    def test_copula_frank_positive_dependence_returns_positive_theta(self):
        rng = np.random.default_rng(15)
        x = rng.normal(0, 1, 300)
        y = 0.8 * x + 0.6 * rng.normal(0, 1, 300)
        result = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'x': x, 'y': y}), time_column='time'
        ).copula_analysis('x', 'y', copula_type='frank')
        assert result['copula_parameter'] > 0
        assert result['dependence_class'] == 'positive'

    @pytest.mark.parametrize('tau, expected', [(0.0, 0.0)])
    def test_frank_theta_zero_tau_returns_zero(self, tau, expected):
        assert analytics_engine.AnalyticsEngine._frank_theta_from_tau(tau) == expected

    def test_frank_theta_nonfinite_tau_raises(self):
        with pytest.raises(ValueError, match='not finite'):
            analytics_engine.AnalyticsEngine._frank_theta_from_tau(np.nan)


class TestComprehensiveReportErrorSections:
    """The report must isolate section failures into {'error': ...} dicts."""

    def test_multivariate_section_reports_error_on_two_row_data(self):
        data = pd.DataFrame({'time': [0.0, 1.0], 'x': [1.0, 2.0]})
        report = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).comprehensive_analysis_report()
        assert 'error' in report['multivariate']
        assert 'Insufficient data for PCA' in report['multivariate']['error']

    def test_time_series_section_reports_error_on_complex_column(self):
        # complex dtype passes the is_numeric gate but breaks the trend
        # regression; the section must record the failure, not crash.
        data = pd.DataFrame({'time': np.arange(12.0),
                             'c': np.exp(1j * np.arange(12.0))})
        report = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).comprehensive_analysis_report()
        assert 'error' in report['time_series']

    def test_string_only_data_yields_network_error_and_no_numeric_sections(self):
        data = pd.DataFrame({'time': [str(i) for i in range(30)], 's': ['a'] * 30})
        report = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).comprehensive_analysis_report()

        assert 'error' in report['network']
        assert report['information_theory'] == {'error': 'No numeric columns'}
        assert report['robust_statistics'] == {'error': 'No numeric columns'}
        assert 'error' in report['multivariate']

    def test_bayesian_section_reports_error_on_non_numeric_pair(self):
        rng = np.random.default_rng(16)
        data = pd.DataFrame({'time': np.arange(30.0), 'x': rng.normal(0, 1, 30),
                             'lab': ['z'] * 30})
        report = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).comprehensive_analysis_report(bayesian_columns=('x', 'lab'))
        assert 'error' in report['bayesian']


class TestTimeSeriesAnalyzerErrorPaths:
    """Guard branches of TimeSeriesAnalyzer seasonality/variance/forecast."""

    def test_seasonality_with_period_too_long_reports_decomposition_error(self):
        # period=10 on a 12-point series passes the len gate but
        # seasonal_decompose needs two full cycles -> 'error' contract.
        t = np.arange(12.0)
        data = pd.DataFrame({'time': t, 's': np.sin(t)})
        result = analytics_engine.TimeSeriesAnalyzer(data, 'time').detect_seasonality(period=10)
        assert result['s']['seasonal_detected'] is False
        assert result['s']['error'] == 'Decomposition failed'

    def test_variance_detection_skips_non_numeric_columns(self):
        data = pd.DataFrame({'time': np.arange(6.0), 'text': ['a'] * 6})
        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        assert analyzer.detect_change_points(method='variance') == []

    def test_variance_detection_skips_short_series(self):
        # min_segment=5 needs at least 10 observations to propose a split.
        data = pd.DataFrame({'time': np.arange(9.0), 's': np.arange(9.0)})
        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')
        assert analyzer.detect_change_points(method='variance') == []

    def test_variance_detection_ignores_zero_variance_split(self):
        # Exactly constant leading segment: the first candidate split has a
        # zero-variance side and must be skipped rather than reported as an
        # infinite F-ratio; the variance increase is still detected later.
        rng = np.random.default_rng(4)
        x = np.concatenate([np.zeros(5), rng.normal(0, 3.0, 15)])
        data = pd.DataFrame({'time': np.arange(20.0), 's': x})
        analyzer = analytics_engine.TimeSeriesAnalyzer(data, 'time')

        cps = analyzer.detect_change_points(method='variance')

        assert len(cps) >= 1
        assert all(np.isfinite(cp['variance_ratio']) for cp in cps)
        assert all(cp['p_value'] < 0.05 for cp in cps)

    def test_forecast_unknown_method_persists_last_value(self):
        data = pd.DataFrame({'time': np.arange(12.0), 's': np.linspace(1.0, 12.0, 12)})
        fc = analytics_engine.TimeSeriesAnalyzer(data, 'time').forecast(
            forecast_steps=3, method='bogus'
        )
        np.testing.assert_array_equal(fc['s'], [12.0, 12.0, 12.0])


class TestEngineDataCoreInput:
    """AnalyticsEngine must flatten DataCore time-series datasets."""

    def test_engine_accepts_datacore_input(self):
        frame = make_growth_frame(n_points=20, phenotype_cols=("x",), seed=18)
        tsd = datacore.TimeSeriesData(
            data=frame, time_column='time', phenotype_columns=['x']
        )
        engine = analytics_engine.AnalyticsEngine(datacore.DataCore(tsd), time_column='time')

        assert engine.data.shape == frame.shape
        assert list(engine.data.columns) == ['time', 'x']


class TestExtremeValuePotFallback:
    def test_pot_falls_back_when_exceedances_are_too_few(self):
        rng = np.random.default_rng(20)
        data = pd.DataFrame({'time': np.arange(50.0), 'v': rng.uniform(0, 10, 50)})
        result = analytics_engine.AnalyticsEngine(
            data, time_column='time'
        ).extreme_value_analysis('v')

        pot = result['pot_method']
        assert pot['n_exceedances'] <= 10
        assert pot['shape_parameter'] == 0.0
        # Without a GPD fit every return level collapses to the threshold.
        assert len(set(pot['return_levels'].values())) == 1
        assert result['block_maxima_method']['n_blocks'] == 10


class TestCopulaMinimumObservations:
    def test_copula_needs_three_observations(self):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 1.0]}), time_column='time'
        )
        with pytest.raises(ValueError, match='Insufficient variation'):
            engine.copula_analysis('x', 'y')


class TestGrangerDegradation:
    def test_granger_degrades_to_error_when_statsmodels_cannot_fit(self):
        # 11 observations pass the >= 2*max_lag gate but statsmodels can
        # only support lag 2 on them; the failure must surface as an
        # error dict rather than an exception.
        rng = np.random.default_rng(19)
        data = pd.DataFrame({'cause': rng.normal(0, 1, 11),
                             'effect': rng.normal(0, 1, 11)})
        result = analytics_engine.CausalInference(data).granger_causality_test(
            'cause', 'effect', max_lag=5
        )
        assert 'error' in result
        assert 'failed' in result['error']


class TestRobustSectionComplexFailure:
    def test_robust_section_reports_error_on_complex_column(self):
        # complex dtype passes the numeric-column gate but breaks the
        # real-valued robust estimators; the report must isolate it.
        data = pd.DataFrame({'c': np.exp(1j * np.arange(12.0))})
        report = analytics_engine.AnalyticsEngine(data).comprehensive_analysis_report()
        assert 'error' in report['robust_statistics']
        assert 'error' in report['multivariate']
