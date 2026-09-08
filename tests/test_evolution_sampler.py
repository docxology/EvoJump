"""
Test suite for EvolutionSampler module.

This module tests the population-level evolutionary analysis functionality
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, evolution_sampler
from scipy import stats as scipy_stats


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

        # Without an explicit 'parent'/'offspring' pedigree there is no
        # estimate: NaN plus a warning, never a fabricated value.
        assert np.isnan(heritability)

    def test_estimate_heritability_known_regression(self):
        """Parent-offspring regression on a synthetic pedigree: h2 = 2 * slope."""
        data = pd.DataFrame({
            'parent': [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            'offspring': [5.0, 5.6, 6.2, 6.8, 7.4, 8.0],  # 0.3 * parent + 2
        })

        model = evolution_sampler.PopulationModel(data)
        heritability = model.estimate_heritability('phenotype', method='parent-offspring')

        assert heritability == pytest.approx(0.6)

    def test_estimate_heritability_caps_at_one(self):
        """A regression slope implying h2 > 1 is capped at 1.0."""
        data = pd.DataFrame({
            'parent': [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            'offspring': [6.0, 7.2, 8.4, 9.6, 10.8, 12.0],  # 0.6 * parent
        })

        model = evolution_sampler.PopulationModel(data)
        heritability = model.estimate_heritability('phenotype', method='parent-offspring')

        assert heritability == 1.0

    def test_estimate_heritability_requires_four_pairs(self):
        """Fewer than four paired observations cannot support the regression."""
        data = pd.DataFrame({
            'parent': [10.0, 12.0, 14.0],
            'offspring': [5.0, 5.6, 6.2],
        })

        model = evolution_sampler.PopulationModel(data)
        heritability = model.estimate_heritability('phenotype', method='parent-offspring')

        assert np.isnan(heritability)

    def test_compute_selection_gradient(self):
        """Test selection gradient computation."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18],
            'fitness': [1.0, 1.2, 1.4, 1.6, 1.8]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        gradient = model.compute_selection_gradient('phenotype1', 'fitness')

        # Fitness perfectly linear in the phenotype: the standardized
        # regression slope is exactly 1.0.
        assert gradient == pytest.approx(1.0)

    def test_compute_selection_gradient_matches_regression(self):
        """The gradient equals the OLS slope of standardized fitness on phenotype."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18],
            'fitness': [1.0, 1.3, 1.2, 1.7, 1.5]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        gradient = model.compute_selection_gradient('phenotype1', 'fitness')

        pheno = data['phenotype1'].to_numpy(dtype=float)
        fitness = data['fitness'].to_numpy(dtype=float)
        expected = scipy_stats.linregress(
            (pheno - pheno.mean()) / pheno.std(),
            (fitness - fitness.mean()) / fitness.std(),
        ).slope
        assert gradient == pytest.approx(expected, rel=1e-12)

    def test_compute_selection_gradient_degenerate_inputs(self):
        """Zero-variance phenotype or missing column yields NaN, not a crash."""
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [5.0, 5.0, 5.0],
            'fitness': [1.0, 2.0, 3.0]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        assert np.isnan(model.compute_selection_gradient('phenotype1', 'fitness'))
        assert np.isnan(model.compute_selection_gradient('missing', 'fitness'))

    def test_estimate_effective_population_size(self):
        """Test effective population size estimation."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        ne = model.estimate_effective_population_size(method='temporal')

        # No 'freq_' allele-frequency columns: Ne is not identifiable from
        # phenotypes alone and is reported as NaN.
        assert np.isnan(ne)

    def test_estimate_effective_population_size_temporal_known_value(self):
        """Waples (1989) plan-II temporal Ne: two loci, p 0.5 -> 0.4.

        F = (0.4 - 0.5)^2 / (0.45 * 0.55) = 0.040404... per locus,
        Ne = 1 / (2F) = 12.375.
        """
        data = pd.DataFrame({
            'time': [0, 0, 1, 1],
            'freq_L1': [0.5, 0.5, 0.4, 0.4],
            'freq_L2': [0.5, 0.5, 0.4, 0.4],
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        ne = model.estimate_effective_population_size(method='temporal')

        assert ne == pytest.approx(12.375, rel=1e-9)

    def test_estimate_effective_population_size_fixed_allele(self):
        """Unchanged allele frequencies give F = 0 and Ne = infinity."""
        data = pd.DataFrame({
            'time': [0, 0, 1, 1],
            'freq_L1': [0.5, 0.5, 0.5, 0.5],
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        ne = model.estimate_effective_population_size(method='temporal')

        assert np.isinf(ne)

    def test_estimate_effective_population_size_single_time_point(self):
        """One time point carries no temporal information: NaN."""
        data = pd.DataFrame({
            'time': [0, 0],
            'freq_L1': [0.5, 0.5],
        })

        model = evolution_sampler.PopulationModel(data, 'time')
        ne = model.estimate_effective_population_size(method='temporal')

        assert np.isnan(ne)


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
        assert 0.0 <= signal <= 1.0


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
        params = {'temperature': 1.0}
        samples = sampler.sample(n_samples=10, method='importance-sampling', parameters=params)

        assert isinstance(samples, evolution_sampler.SampleResult)
        assert samples.samples.shape[0] == 10
        assert samples.sampling_method == 'importance-sampling'
        # Effective sample size recorded on the result's copy of parameters.
        assert 0.0 < samples.parameters['ess'] <= 10
        # The caller's dict is neither mutated nor aliased.
        assert 'ess' not in params
        assert samples.parameters is not params

    def test_sample_mcmc(self):
        """Test MCMC sampling."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        params = {'step_size': 0.5}
        samples = sampler.sample(n_samples=10, method='mcmc', parameters=params)

        assert isinstance(samples, evolution_sampler.SampleResult)
        assert samples.samples.shape[0] == 10
        assert samples.sampling_method == 'mcmc'
        # Post-burn-in acceptance rate recorded on the result's copy.
        assert 0.0 <= samples.parameters['acceptance_rate'] <= 1.0
        assert 'acceptance_rate' not in params
        assert samples.parameters is not params

    def test_seed_reproducibility(self):
        """Identical seeds reproduce samples and diagnostics; different seeds differ."""
        data = self.create_test_data()

        s1 = evolution_sampler.EvolutionSampler(data, time_column='time')
        s2 = evolution_sampler.EvolutionSampler(data, time_column='time')
        s3 = evolution_sampler.EvolutionSampler(data, time_column='time')
        s1.seed(42)
        s2.seed(42)
        s3.seed(7)

        r1 = s1.sample(n_samples=10, method='monte-carlo')
        r2 = s2.sample(n_samples=10, method='monte-carlo')
        r3 = s3.sample(n_samples=10, method='monte-carlo')
        assert np.array_equal(r1.samples, r2.samples)
        assert not np.array_equal(r1.samples, r3.samples)

        # Same generators, advanced by identical call sequences: MCMC and
        # importance sampling reproduce samples and diagnostics exactly.
        m1 = s1.sample(n_samples=10, method='mcmc')
        m2 = s2.sample(n_samples=10, method='mcmc')
        assert np.array_equal(m1.samples, m2.samples)
        assert m1.parameters['acceptance_rate'] == m2.parameters['acceptance_rate']

        i1 = s1.sample(n_samples=10, method='importance-sampling')
        i2 = s2.sample(n_samples=10, method='importance-sampling')
        assert np.array_equal(i1.samples, i2.samples)
        assert i1.parameters['ess'] == i2.parameters['ess']

    def test_analyze_evolutionary_patterns(self):
        """Test evolutionary pattern analysis."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        patterns = sampler.analyze_evolutionary_patterns()

        assert 'population_statistics' in patterns
        assert 'genetic_parameters' in patterns
        assert 'selection_analysis' in patterns
        assert isinstance(patterns['population_statistics'], evolution_sampler.PopulationStatistics)

    def test_phylogenetic_signal_requires_matching_distance_matrix(self):
        """Moran's I is skipped without a matrix and computed with one."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 1, 2, 3],
            'phenotype1': [10.0, 12.0, 14.0, 11.0, 13.0, 15.0],
        })

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        patterns = sampler.analyze_evolutionary_patterns()
        # No distance matrix supplied: the statistic is undefined (rows of a
        # time-series table are not taxa), so the entry stays empty rather
        # than holding meaningless NaNs.
        assert patterns['phylogenetic_signal'] == {}

        # A matching distance matrix (rows ordered like the trait values)
        # yields finite Moran's I.
        values = data['phenotype1'].to_numpy(dtype=float)
        distance = np.abs(values[:, None] - values[None, :])
        sampler.phylogenetic_analyzer = evolution_sampler.PhylogeneticAnalyzer(distance)
        patterns = sampler.analyze_evolutionary_patterns()
        assert np.isfinite(patterns['phylogenetic_signal']['phenotype1'])

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
        # No fitness measure available in a phenotype-only time series:
        # gradients are NaN, never a degenerate phenotype-vs-itself 1.0.
        assert all(np.isnan(v) for v in stats.selection_gradients.values())
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
        # Without a pedigree the additive/environmental split is not
        # identifiable: NaN plus an explicit 'available' marker, never
        # fabricated zeros.
        assert params['available'] is False
        assert np.isnan(params['narrow_sense_heritability'])
        assert np.isnan(params['additive_variance'])

    def test_estimate_genetic_parameters_with_pedigree(self):
        """With pedigree and replication: V_A = h2 * V_P, V_E = (1 - h2) * V_P."""
        data = pd.DataFrame({
            'time': [1, 1, 1, 2, 2, 2],
            'phenotype1': [10.0, 12.0, 14.0, 11.0, 13.0, 15.0],
            'parent': [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            'offspring': [5.0, 5.6, 6.2, 6.8, 7.4, 8.0],  # slope 0.3 -> h2 0.6
        })

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        params = sampler._estimate_genetic_parameters()

        assert params['available'] is True
        assert params['narrow_sense_heritability'] == pytest.approx(0.6)
        # Within-time-point phenotypic variance is 4.0 at both time points.
        assert params['additive_variance'] == pytest.approx(0.6 * 4.0)
        assert params['environmental_variance'] == pytest.approx(0.4 * 4.0)
        # Not identifiable from phenotypic time series alone.
        assert np.isnan(params['dominance_variance'])
        assert np.isnan(params['epistatic_variance'])
        assert np.isnan(params['broad_sense_heritability'])

    def test_analyze_selection(self):
        """Test selection analysis."""
        data = self.create_test_data()

        sampler = evolution_sampler.EvolutionSampler(data, time_column='time')
        selection = sampler._analyze_selection()

        assert isinstance(selection, dict)
        assert 'directional_selection' in selection
        assert 'stabilizing_selection' in selection
        assert 'disruptive_selection' in selection
        # Means shift 10 -> 18 (phenotype1) and 20 -> 28 (phenotype2); the
        # SD at the first time point is 1.0 for both, so the mean
        # standardized directional differential is 8.0.
        assert selection['selection_differential']['phenotype1'] == pytest.approx(8.0)
        assert selection['directional_selection'] == pytest.approx(8.0)
        # Variance is unchanged (1.0 at both ends): neither stabilizing nor
        # disruptive selection is measurable, reported as NaN not 0.
        assert np.isnan(selection['stabilizing_selection'])
        assert np.isnan(selection['disruptive_selection'])
        # No pedigree: the Lande response is not estimable, reported as NaN.
        assert np.isnan(selection['selection_response']['phenotype1'])

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
        """Lambda estimation separates phylogenetically ordered from shuffled traits."""
        # Four taxa equally spaced on a path phylogeny: a fixed, Euclidean
        # distance matrix (deterministic, unlike the previous unseeded
        # np.random.rand matrix).
        distance_matrix = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)

        analyzer = evolution_sampler.PhylogeneticAnalyzer(distance_matrix)

        # Traits increasing exactly along the phylogeny: lambda near 1.
        ordered = analyzer.compute_phylogenetic_signal(np.array([0.0, 1.0, 2.0, 3.0]))
        assert 0.9 <= ordered <= 1.0

        # Traits orthogonal to the phylogeny's leading eigenvector (shuffled
        # relative to the topology): lambda near 0.
        shuffled = analyzer.compute_phylogenetic_signal(np.array([1.0, -1.0, -1.0, 1.0]))
        assert 0.0 <= shuffled <= 0.1

    def test_phylogenetic_signal_requires_euclidean_distance_matrix(self):
        """A non-Euclidean matrix triggers a warning and still returns a bounded lambda."""
        distance_matrix = np.array([
            [0.0, 10.0, 1.0],
            [10.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])

        analyzer = evolution_sampler.PhylogeneticAnalyzer(distance_matrix)
        with pytest.warns(UserWarning, match="Euclidean"):
            signal = analyzer.compute_phylogenetic_signal(np.array([1.0, 2.0, 3.0]))

        assert 0.0 <= signal <= 1.0
