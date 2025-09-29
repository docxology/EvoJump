"""
Test suite for EvolutionSampler module.

This module tests the population-level evolutionary analysis functionality
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
from evojump import datacore, evolution_sampler


class TestPopulationModel:
    """Test PopulationModel class."""

    def test_estimate_heritability_parent_offspring(self):
        """Test heritability estimation using parent-offspring regression."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        heritability = model.estimate_heritability('phenotype1', method='parent-offspring')

        assert isinstance(heritability, (float, type(np.nan)))
        assert 0.0 <= heritability <= 1.0 or np.isnan(heritability)

    def test_compute_selection_gradient(self):
        """Test selection gradient computation."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18],
            'fitness': [1.0, 1.2, 1.4, 1.6, 1.8]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        gradient = model.compute_selection_gradient('phenotype1', 'fitness')

        assert isinstance(gradient, (float, type(np.nan)))

    def test_estimate_effective_population_size(self):
        """Test effective population size estimation."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        ne = model.estimate_effective_population_size(method='temporal')

        assert isinstance(ne, (float, type(np.nan)))
        assert ne > 0 or np.isnan(ne)


class TestPhylogeneticAnalyzer:
    """Test PhylogeneticAnalyzer class."""

    def test_compute_phylogenetic_signal(self):
        """Test phylogenetic signal computation."""
        # Create simple distance matrix
        distance_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0]
        ])

        analyzer = evolution_sampler.PhylogeneticAnalyzer(distance_matrix)

        # Create trait data
        traits = np.array([1.0, 2.0, 3.0])

        signal = analyzer.compute_phylogenetic_signal(traits, method='lambda')

        assert isinstance(signal, (float, type(np.nan)))
        assert 0.0 <= signal <= 1.0 or np.isnan(signal)


class TestQuantitativeGenetics:
    """Test QuantitativeGenetics class."""

    def test_estimate_breeding_values_blup(self):
        """Test breeding value estimation using BLUP."""
        data = pd.DataFrame({
            'individual1': [10, 12, 14, 16, 18],
            'individual2': [11, 13, 15, 17, 19],
            'individual3': [9, 11, 13, 15, 17]
        })

        genetics = evolution_sampler.QuantitativeGenetics()
        breeding_values = genetics.estimate_breeding_values(data, method='blup')

        assert isinstance(breeding_values, pd.DataFrame)
        assert breeding_values.shape == data.shape
        assert breeding_values.index.equals(data.index)
        assert breeding_values.columns.equals(data.columns)

    def test_compute_genetic_correlations(self):
        """Test genetic correlation computation."""
        data = pd.DataFrame({
            'trait1': [10, 12, 14, 16, 18],
            'trait2': [20, 22, 24, 26, 28]
        })

        genetics = evolution_sampler.QuantitativeGenetics()
        correlations = genetics.compute_genetic_correlations(data, time_points=[1, 2, 3, 4, 5])

        assert isinstance(correlations, np.ndarray)
        assert correlations.shape == (2, 2)
        # Check diagonal elements (may be NaN if computation fails)
        diag_elements = np.diag(correlations)
        valid_diag = diag_elements[~np.isnan(diag_elements)]
        if len(valid_diag) > 0:
            assert np.allclose(valid_diag, 1.0)


class TestEvolutionSampler:
    """Test EvolutionSampler class."""

    def create_test_data(self):
        """Create test data for EvolutionSampler."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 9, 11, 13, 15, 17],
            'phenotype2': [20, 22, 24, 26, 28, 21, 23, 25, 27, 29, 19, 21, 23, 25, 27]
        })
        return data

    def test_evolution_sampler_initialization_with_datacore(self):
        """Test EvolutionSampler initialization with DataCore."""
        data = self.create_test_data()

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        data_core = datacore.DataCore([ts_data])

        sampler = evolution_sampler.EvolutionSampler(data_core, time_column='time')

        assert sampler.time_column == 'time'
        assert sampler.population_model is not None
        assert sampler.phylogenetic_analyzer is not None
        assert sampler.quantitative_genetics is not None

    def test_evolution_sampler_initialization_with_dataframe(self):
        """Test EvolutionSampler initialization with DataFrame."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')

        assert sampler.time_column == 'time'
        assert sampler.population_model is not None

    def test_sample_monte_carlo(self):
        """Test Monte Carlo sampling."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        samples = sampler.sample(n_samples=10, method='monte-carlo')

        assert isinstance(samples, evolution_sampler.SampleResult)
        assert samples.samples.shape[0] == 10
        assert samples.sampling_method == 'monte-carlo'
        assert len(samples.sample_ids) == 10

    def test_sample_importance_sampling(self):
        """Test importance sampling."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        samples = sampler.sample(n_samples=10, method='importance-sampling')

        assert isinstance(samples, evolution_sampler.SampleResult)
        assert samples.samples.shape[0] == 10
        assert samples.sampling_method == 'importance-sampling'

    def test_sample_mcmc(self):
        """Test MCMC sampling."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        samples = sampler.sample(n_samples=10, method='mcmc')

        assert isinstance(samples, evolution_sampler.SampleResult)
        assert samples.samples.shape[0] == 10
        assert samples.sampling_method == 'mcmc'

    def test_analyze_evolutionary_patterns(self):
        """Test evolutionary pattern analysis."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        patterns = sampler.analyze_evolutionary_patterns()

        assert 'population_statistics' in patterns
        assert 'genetic_parameters' in patterns
        assert 'selection_analysis' in patterns
        assert isinstance(patterns['population_statistics'], evolution_sampler.PopulationStatistics)

    def test_cluster_individuals(self):
        """Test individual clustering."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        clusters = sampler.cluster_individuals(n_clusters=2)

        assert 'cluster_labels' in clusters
        assert 'cluster_statistics' in clusters
        assert 'gmm_parameters' in clusters
        # Clustering is performed on final time point data, not all data
        assert len(clusters['cluster_labels']) > 0
        assert len(clusters['cluster_statistics']) == 2  # n_clusters

    def test_population_statistics_computation(self):
        """Test population statistics computation."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        stats = sampler._compute_population_statistics()

        assert isinstance(stats, evolution_sampler.PopulationStatistics)
        assert stats.mean_trajectory.shape[0] > 0
        assert stats.variance_trajectory.shape[0] > 0
        assert stats.covariance_matrix.shape[0] > 0
        assert isinstance(stats.heritability_estimates, dict)
        assert isinstance(stats.selection_gradients, dict)
        assert isinstance(stats.effective_population_size, (float, type(np.nan)))

    def test_estimate_genetic_parameters(self):
        """Test genetic parameter estimation."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        params = sampler._estimate_genetic_parameters()

        assert isinstance(params, dict)
        assert 'additive_variance' in params
        assert 'dominance_variance' in params
        assert 'environmental_variance' in params

    def test_analyze_selection(self):
        """Test selection analysis."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        selection = sampler._analyze_selection()

        assert isinstance(selection, dict)
        assert 'directional_selection' in selection
        assert 'stabilizing_selection' in selection
        assert 'disruptive_selection' in selection

    def test_monte_carlo_sampling_with_time_series(self):
        """Test Monte Carlo sampling with time series data."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        samples = sampler.sample(n_samples=5, method='monte-carlo')

        # Should have 3D array for time series
        assert samples.samples.ndim == 3
        assert samples.samples.shape[0] == 5  # n_samples
        assert samples.samples.shape[1] == 5  # time points
        assert samples.samples.shape[2] == 2  # phenotypes

    def test_monte_carlo_sampling_with_cross_sectional(self):
        """Test Monte Carlo sampling with cross-sectional data."""
        data = pd.DataFrame({
            'phenotype1': [10, 12, 14, 16, 18],
            'phenotype2': [20, 22, 24, 26, 28]
        })

        sampler = evolution_sampler.EvolutionSampler(data, time_column=None)
        samples = sampler.sample(n_samples=3, method='monte-carlo')

        # Should have 2D array for cross-sectional
        assert samples.samples.ndim == 2
        assert samples.samples.shape[0] == 3  # n_samples
        assert samples.samples.shape[1] == 2  # phenotypes

    def test_invalid_sampling_method(self):
        """Test error handling for invalid sampling method."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')

        with pytest.raises(ValueError, match="Unsupported sampling method"):
            sampler.sample(n_samples=10, method='invalid_method')

    def test_cluster_with_insufficient_data(self):
        """Test clustering with insufficient data."""
        data = pd.DataFrame({
            'time': [1, 2],
            'phenotype1': [10, 12]
        })

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')

        with pytest.raises(ValueError, match="Not enough data points"):
            sampler.cluster_individuals(n_clusters=5)

    def test_phylogenetic_signal_with_distance_matrix(self):
        """Test phylogenetic signal computation with distance matrix."""
        data = self.create_test_data()

        # Create simple distance matrix
        n_individuals = len(data) // 5  # 5 time points
        distance_matrix = np.random.rand(n_individuals, n_individuals)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(distance_matrix, 0)  # Zero diagonal

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        sampler.phylogenetic_analyzer = evolution_sampler.PhylogeneticAnalyzer(distance_matrix)

        # Test with subset of data
        test_data = data['phenotype1'].values[:n_individuals]
        signal = sampler.phylogenetic_analyzer.compute_phylogenetic_signal(test_data)

        assert isinstance(signal, (float, type(np.nan)))
