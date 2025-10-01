# EvoJump Testing Framework Documentation

## Overview

This document outlines the comprehensive testing framework for the EvoJump project, ensuring high-quality, reliable, and maintainable code through rigorous test-driven development (TDD) practices.

## Testing Philosophy

EvoJump follows a strict test-driven development approach with the following core principles:

- **Real Data Testing**: All tests use real biological and synthetic data, never mocks
- **Comprehensive Coverage**: 95%+ code coverage requirement across all modules
- **Integration Testing**: Tests validate interactions between all major components
- **Edge Case Validation**: Extensive testing of error conditions and boundary cases
- **Performance Validation**: Tests ensure computational efficiency for large datasets

## Test Structure

### Core Test Suites

#### 1. `test_datacore.py` - Data Management Testing
- **Purpose**: Validates data ingestion, preprocessing, and quality control
- **Coverage**: TimeSeriesData, DataCore, MetadataManager classes
- **Key Tests**:
  - Data validation and quality metrics
  - Missing data interpolation
  - Outlier detection and removal
  - Normalization methods (z-score, min-max, robust)
  - Cross-dataset consistency validation

#### 2. `test_jumprope.py` - Jump-Diffusion Modeling
- **Purpose**: Tests stochastic process implementations and model fitting
- **Coverage**: ModelParameters, JumpRope, stochastic process classes
- **Key Tests**:
  - Parameter estimation and validation
  - Trajectory generation and simulation
  - Model fitting across different stochastic processes
  - Cross-section computation
  - Model persistence and serialization

#### 3. `test_laserplane.py` - Cross-Sectional Analysis
- **Purpose**: Validates distribution analysis and statistical comparisons
- **Coverage**: DistributionFitter, DistributionComparer, MomentAnalyzer, LaserPlaneAnalyzer
- **Key Tests**:
  - Distribution fitting (normal, lognormal, gamma, etc.)
  - Statistical comparison methods (KS test, Mann-Whitney)
  - Moment computation and confidence intervals
  - Bootstrap analysis
  - Goodness-of-fit assessment

#### 4. `test_trajectory_visualizer.py` - Visualization Testing
- **Purpose**: Ensures accurate and robust visualization functionality
- **Coverage**: TrajectoryVisualizer, AnimationController, PlotConfig
- **Key Tests**:
  - Static and animated plotting
  - Cross-section visualizations
  - Landscape analysis plots
  - Model comparison visualizations
  - Animation frame generation

#### 5. `test_analytics_engine.py` - Statistical Analysis
- **Purpose**: Comprehensive validation of statistical analysis methods
- **Coverage**: AnalyticsEngine, BayesianAnalyzer, NetworkAnalyzer, CausalInference
- **Key Tests**:
  - Time series analysis (trends, seasonality, change points)
  - Multivariate analysis (PCA, CCA, clustering)
  - Predictive modeling (random forest, cross-validation)
  - Bayesian analysis and model comparison
  - Network analysis and causal inference

#### 6. `test_evolution_sampler.py` - Evolutionary Analysis
- **Purpose**: Tests population-level evolutionary modeling
- **Coverage**: PopulationModel, PhylogeneticAnalyzer, QuantitativeGenetics, EvolutionSampler
- **Key Tests**:
  - Heritability estimation
  - Selection gradient computation
  - Effective population size estimation
  - Monte Carlo and MCMC sampling
  - Phylogenetic signal analysis

#### 7. `test_advanced_features.py` - Advanced Stochastic Models
- **Purpose**: Validates advanced stochastic process implementations
- **Coverage**: Fractional Brownian Motion, CIR process, Levy processes
- **Key Tests**:
  - Process initialization and parameter validation
  - Trajectory simulation and statistical properties
  - Parameter estimation methods
  - Integration with JumpRope framework
  - Advanced visualization methods

#### 8. `test_cli.py` - Command-Line Interface
- **Purpose**: Tests command-line interface functionality
- **Coverage**: Argument parsing, subcommand execution, data validation
- **Key Tests**:
  - CLI argument parsing and validation
  - Subcommand functionality (analyze, fit, visualize, sample)
  - Input file validation and error handling
  - Logging and output formatting
  - Integration with core modules

## Test Execution

### Running Tests

The project uses both pytest and the comprehensive `run_all_tests.py` script for complete validation:

#### Quick Testing (run_all_tests.py)
```bash
# Run all tests quickly (no coverage, fast feedback)
python run_all_tests.py --quick

# Run tests with comprehensive coverage report
python run_all_tests.py --coverage --verbose

# Run all validation checks (tests + linting + documentation)
python run_all_tests.py --all

# Run performance benchmarks
python run_all_tests.py --benchmark --parallel

# Run code quality checks only
python run_all_tests.py --lint

# Check documentation completeness
python run_all_tests.py --docs
```

#### Detailed Testing (pytest)
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evojump --cov-report=html

# Run specific test file
pytest tests/test_datacore.py

# Run with verbose output
pytest -v

# Run specific test class or method
pytest tests/test_jumprope.py::TestOrnsteinUhlenbeckJump::test_simulate_trajectory

# Run tests in parallel
pytest -n auto

# Run only integration tests
pytest -k "integration or fit or analyze or compare"

# Run only unit tests
pytest -k "not integration and not fit and not analyze and not compare"
```

### Test Configuration

All test configuration is managed through `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--cov=evojump",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=95",
]
```

## Coverage Requirements

- **Minimum Coverage**: 95% across all modules
- **Branch Coverage**: Validated for critical paths
- **Integration Coverage**: All module interactions tested
- **Performance Coverage**: Large dataset handling validated

## Test Data Strategy

### Synthetic Data Generation
Tests use carefully designed synthetic datasets that:
- Mimic real biological data characteristics
- Cover edge cases and boundary conditions
- Validate numerical accuracy
- Ensure reproducible results

### Real Data Integration
Where applicable, tests incorporate real biological datasets to:
- Validate method performance on actual data
- Ensure biological relevance
- Test scalability with realistic data sizes

## Quality Assurance

### Code Quality Metrics
- **Test Maintainability**: Tests are designed to be easily updated as code evolves
- **Documentation Quality**: All tests include comprehensive docstrings
- **Performance Validation**: Tests ensure computational efficiency
- **Memory Management**: Tests validate memory usage patterns

### Continuous Integration
Tests are designed to run efficiently in CI/CD environments:
- Parallel test execution support
- Minimal external dependencies
- Deterministic results
- Fast feedback loops

## Best Practices

### Test Design
1. **Single Responsibility**: Each test validates one specific behavior
2. **Clear Assertions**: Tests use descriptive assertion messages
3. **Setup/Teardown**: Proper test isolation and cleanup
4. **Error Handling**: Validation of error conditions and edge cases

### Test Data Management
1. **Realistic Data**: Use biologically meaningful synthetic data
2. **Edge Cases**: Include boundary conditions and error scenarios
3. **Scalability Testing**: Validate performance with varying data sizes
4. **Reproducibility**: Ensure test results are deterministic

### Documentation
1. **Module Documentation**: Clear description of test purpose and scope
2. **Class Documentation**: Explanation of test class responsibilities
3. **Method Documentation**: Detailed description of test scenarios
4. **Assertion Clarity**: Self-documenting test assertions

## Test Results and Reporting

### Coverage Reports
- **HTML Reports**: Interactive coverage visualization in `htmlcov/`
- **XML Reports**: CI/CD integration in `coverage.xml`
- **Terminal Reports**: Real-time coverage feedback

### Test Metrics
- **Pass/Fail Rates**: Tracked across development cycles
- **Performance Benchmarks**: Execution time monitoring
- **Memory Usage**: Resource utilization tracking
- **Flaky Test Detection**: Identification of non-deterministic tests

## Troubleshooting

### Common Issues
1. **Test Data Dependencies**: Ensure synthetic data generation is stable
2. **Numerical Precision**: Handle floating-point comparisons appropriately
3. **Random Seed Management**: Ensure reproducible stochastic tests
4. **Memory Management**: Monitor large dataset test memory usage

### Debug Strategies
1. **Verbose Mode**: Use `-v` flag for detailed test output
2. **Specific Test Focus**: Run individual tests for targeted debugging
3. **Coverage Analysis**: Use coverage reports to identify untested code paths
4. **Performance Profiling**: Monitor test execution times

## Future Enhancements

### Planned Improvements
1. **Property-Based Testing**: Integration of hypothesis for generative testing
2. **Performance Benchmarks**: Automated performance regression testing
3. **Integration Tests**: Enhanced cross-module integration validation
4. **Load Testing**: Large-scale dataset performance validation

### Test Infrastructure
1. **Test Parallelization**: Improved parallel test execution
2. **Test Discovery**: Enhanced test organization and discovery
3. **CI/CD Integration**: Streamlined continuous integration workflows
4. **Documentation Updates**: Automated test documentation generation

---

This testing framework ensures EvoJump maintains the highest standards of scientific software quality, reliability, and maintainability.
