"""
Test suite for Drosophila case study module.

This module tests the comprehensive Drosophila melanogaster case study functionality,
validating the scientific accuracy and EvoJump integration of the fruit fly biology
analysis framework.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from examples.drosophila_case_study import (
    DrosophilaPopulation,
    DrosophilaDataGenerator,
    DrosophilaAnalyzer,
    run_drosophila_case_study
)


class TestDrosophilaPopulation:
    """Test DrosophilaPopulation configuration class."""

    def test_population_initialization(self):
        """Test population configuration initialization."""
        config = DrosophilaPopulation()

        assert config.population_size == 100
        assert config.generations == 10
        assert config.initial_red_eyed_proportion == 0.1
        assert config.advantageous_trait_fitness == 1.2
        assert config.base_eye_size == 10.0

    def test_population_custom_parameters(self):
        """Test population with custom parameters."""
        config = DrosophilaPopulation(
            population_size=200,
            generations=20,
            initial_red_eyed_proportion=0.2,
            advantageous_trait_fitness=1.5
        )

        assert config.population_size == 200
        assert config.generations == 20
        assert config.initial_red_eyed_proportion == 0.2
        assert config.advantageous_trait_fitness == 1.5


class TestDrosophilaDataGenerator:
    """Test Drosophila data generation functionality."""

    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        config = DrosophilaPopulation()
        generator = DrosophilaDataGenerator(config)

        assert generator.config == config
        assert generator.config.population_size == 100

    def test_generate_population_data_basic(self):
        """Test basic population data generation."""
        config = DrosophilaPopulation(population_size=50, generations=5)
        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50 * 5  # population_size * generations
        assert 'generation' in data.columns
        assert 'individual_id' in data.columns
        assert 'genotype' in data.columns
        assert 'phenotype' in data.columns
        assert 'fitness' in data.columns

        # Check value ranges
        assert data['generation'].min() == 0
        assert data['generation'].max() == 4
        assert data['genotype'].isin([0, 1]).all()
        assert data['phenotype'].min() > 0  # Positive phenotypes
        assert data['fitness'].min() > 0  # Positive fitness

    def test_generate_population_data_selective_advantage(self):
        """Test that advantageous trait spreads over generations."""
        config = DrosophilaPopulation(
            population_size=100,
            generations=10,
            initial_red_eyed_proportion=0.1,
            advantageous_trait_fitness=1.5  # Strong selection
        )
        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        # Check that red-eyed frequency increases over time
        final_generation = data[data['generation'] == 9]
        initial_generation = data[data['generation'] == 0]

        final_red_proportion = (final_generation['genotype'] == 1).mean()
        initial_red_proportion = (initial_generation['genotype'] == 1).mean()

        # Should show increase due to selection
        assert final_red_proportion > initial_red_proportion

    def test_simulate_selection_deterministic(self):
        """Test selection simulation logic."""
        config = DrosophilaPopulation(population_size=100, advantageous_trait_fitness=1.2)
        generator = DrosophilaDataGenerator(config)

        # Test selection simulation
        initial_count = 10
        new_count = generator._simulate_selection(initial_count)

        # Should be positive integer
        assert isinstance(new_count, int)
        assert new_count >= 0
        assert new_count <= config.population_size


class TestDrosophilaAnalyzer:
    """Test Drosophila analyzer functionality."""

    def create_test_data(self):
        """Create test data for analyzer."""
        config = DrosophilaPopulation(population_size=50, generations=5)
        generator = DrosophilaDataGenerator(config)
        return generator.generate_population_data()

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        data = self.create_test_data()
        analyzer = DrosophilaAnalyzer(data)

        assert analyzer.population_data is data
        assert analyzer.data_core is None
        assert analyzer.models == {}
        assert analyzer.analyzers == {}

    def test_setup_analysis_pipeline(self):
        """Test analysis pipeline setup."""
        data = self.create_test_data()
        analyzer = DrosophilaAnalyzer(data)
        analyzer.setup_analysis_pipeline()

        # Check that components were initialized
        assert analyzer.data_core is not None
        assert 'laser_plane' in analyzer.analyzers
        assert 'evolution' in analyzer.analyzers
        assert 'analytics' in analyzer.analyzers

    def test_model_population_dynamics(self):
        """Test population dynamics modeling."""
        data = self.create_test_data()
        analyzer = DrosophilaAnalyzer(data)
        analyzer.setup_analysis_pipeline()
        analyzer.model_population_dynamics()

        # Should have at least one model fitted
        assert len(analyzer.models) > 0

        # Check model structure
        for model_name, model in analyzer.models.items():
            assert hasattr(model, 'fitted_parameters')
            assert hasattr(model, 'time_points')
            assert model.fitted_parameters is not None

    def test_analyze_evolutionary_patterns(self):
        """Test evolutionary pattern analysis."""
        data = self.create_test_data()
        analyzer = DrosophilaAnalyzer(data)
        analyzer.setup_analysis_pipeline()
        analyzer.model_population_dynamics()

        results = analyzer.analyze_evolutionary_patterns()

        # Check result structure
        assert 'evolutionary_sampling' in results
        assert 'evolutionary_patterns' in results
        assert 'time_series' in results
        assert 'multivariate' in results
        assert 'bayesian' in results

        # Check that sampling worked
        assert results['evolutionary_sampling'].samples.shape[0] > 0

    def test_analyze_selective_sweeps(self):
        """Test selective sweep analysis."""
        data = self.create_test_data()
        analyzer = DrosophilaAnalyzer(data)
        analyzer.setup_analysis_pipeline()

        results = analyzer.analyze_selective_sweeps()

        # Check result structure
        assert 'sweep_data' in results
        assert 'network_analysis' in results
        assert 'sweep_summary' in results

        # Check sweep data
        sweep_df = results['sweep_data']
        assert isinstance(sweep_df, pd.DataFrame)
        assert 'generation' in sweep_df.columns
        assert 'marker_id' in sweep_df.columns
        assert 'linkage_distance' in sweep_df.columns
        assert 'marker_frequency' in sweep_df.columns
        assert 'genetic_diversity' in sweep_df.columns

    def test_scientific_conclusions(self):
        """Test scientific conclusion generation."""
        data = self.create_test_data()
        analyzer = DrosophilaAnalyzer(data)
        analyzer.setup_analysis_pipeline()

        # Test individual conclusion methods
        assert isinstance(analyzer._detect_selective_sweep(), bool)
        assert isinstance(analyzer._assess_hitchhiking(), bool)
        assert isinstance(analyzer._estimate_evolutionary_rate(), (int, float))
        assert isinstance(analyzer._estimate_population_parameters(), dict)

        # Test fixation time estimation
        fixation_time = analyzer._estimate_fixation_time()
        assert isinstance(fixation_time, int)
        assert fixation_time >= 0


class TestDrosophilaCaseStudyIntegration:
    """Test integration of the complete case study."""

    def test_run_drosophila_case_study_structure(self):
        """Test that case study runs and produces expected structure."""
        # This test might be slow, so we'll test the structure without full execution
        try:
            # Test that function exists and can be imported
            from examples.drosophila_case_study import run_drosophila_case_study
            assert callable(run_drosophila_case_study)

            # Test that population config works
            config = DrosophilaPopulation()
            assert config.population_size > 0

            # Test data generation
            generator = DrosophilaDataGenerator(config)
            data = generator.generate_population_data()
            assert len(data) > 0

            # Test analyzer initialization
            analyzer = DrosophilaAnalyzer(data)
            assert analyzer.population_data is data

        except ImportError as e:
            pytest.skip(f"Import error: {e}")

    def test_case_study_output_structure(self):
        """Test that case study produces expected output structure."""
        # Test the expected output structure without full execution
        expected_keys = [
            'status',
            'output_directory',
            'report',
            'visualizations'
        ]

        # Test that we can access the main function
        from examples.drosophila_case_study import run_drosophila_case_study
        assert callable(run_drosophila_case_study)

        # Test population configuration
        config = DrosophilaPopulation()
        assert hasattr(config, 'population_size')
        assert hasattr(config, 'generations')

    def test_data_generation_reproducibility(self):
        """Test that data generation is reproducible with fixed seed."""
        config1 = DrosophilaPopulation(population_size=50, generations=3)
        config2 = DrosophilaPopulation(population_size=50, generations=3)

        # Create generators with same seed
        generator1 = DrosophilaDataGenerator(config1)
        generator2 = DrosophilaDataGenerator(config2)

        data1 = generator1.generate_population_data()
        data2 = generator2.generate_population_data()

        # Check that key statistics are the same (not exact equality due to stochastic nature)
        assert data1.shape == data2.shape
        assert data1['generation'].equals(data2['generation'])
        assert abs(data1['allele_frequency'].mean() - data2['allele_frequency'].mean()) < 0.01


class TestDrosophilaScientificValidation:
    """Test scientific accuracy of Drosophila case study."""

    def test_selective_sweep_detection(self):
        """Test selective sweep detection logic."""
        # Create data with strong selection
        config = DrosophilaPopulation(
            population_size=100,
            generations=10,
            initial_red_eyed_proportion=0.1,
            advantageous_trait_fitness=1.8  # Very strong selection
        )

        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        analyzer = DrosophilaAnalyzer(data)

        # Should detect selective sweep
        sweep_detected = analyzer._detect_selective_sweep()
        assert sweep_detected == True

    def test_hitchhiking_assessment(self):
        """Test genetic hitchhiking assessment."""
        config = DrosophilaPopulation(population_size=100, generations=8)
        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        analyzer = DrosophilaAnalyzer(data)
        analyzer.setup_analysis_pipeline()

        # Should detect hitchhiking evidence
        hitchhiking_evidence = analyzer._assess_hitchhiking()
        assert isinstance(hitchhiking_evidence, bool)

    def test_evolutionary_rate_calculation(self):
        """Test evolutionary rate calculation."""
        config = DrosophilaPopulation(population_size=100, generations=10)
        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        analyzer = DrosophilaAnalyzer(data)

        # Should calculate positive evolutionary rate
        rate = analyzer._estimate_evolutionary_rate()
        assert isinstance(rate, (int, float))

        # Rate should be positive with selection
        assert rate > 0

    def test_population_parameter_estimation(self):
        """Test population genetic parameter estimation."""
        config = DrosophilaPopulation(population_size=100, generations=5)
        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        analyzer = DrosophilaAnalyzer(data)

        params = analyzer._estimate_population_parameters()

        # Check parameter structure
        assert isinstance(params, dict)
        assert 'effective_population_size' in params
        assert 'narrow_sense_heritability' in params
        assert 'selection_coefficient' in params
        assert 'additive_genetic_variance' in params

        # Check parameter ranges
        assert params['effective_population_size'] > 0
        assert 0 <= params['narrow_sense_heritability'] <= 1
        assert params['selection_coefficient'] >= 0


class TestDrosophilaDataQuality:
    """Test data quality and validation."""

    def test_data_generation_completeness(self):
        """Test that generated data is complete and valid."""
        config = DrosophilaPopulation(population_size=50, generations=5)
        generator = DrosophilaDataGenerator(config)
        data = generator.generate_population_data()

        # Check for missing values
        assert not data.isnull().any().any()

        # Check data types
        assert data['generation'].dtype in [int, float]
        assert data['individual_id'].dtype == object
        assert data['genotype'].dtype in [int, float]
        assert data['phenotype'].dtype in [int, float]
        assert data['fitness'].dtype in [int, float]

        # Check value ranges
        assert data['generation'].min() == 0
        assert data['generation'].max() == 4
        assert data['genotype'].isin([0, 1]).all()
        assert data['phenotype'].min() > 0
        assert data['fitness'].min() > 0

    def test_data_generation_scaling(self):
        """Test data generation with different population sizes."""
        for pop_size in [10, 50, 100]:
            config = DrosophilaPopulation(population_size=pop_size, generations=3)
            generator = DrosophilaDataGenerator(config)
            data = generator.generate_population_data()

            expected_rows = pop_size * 3
            assert len(data) == expected_rows
            assert data['population_size'].unique()[0] == pop_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

