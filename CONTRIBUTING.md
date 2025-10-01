# Contributing to EvoJump

Thank you for your interest in contributing to EvoJump! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/yourusername/evojump.git
cd evojump
```

2. **Create a virtual environment and install dependencies**:
```bash
# Using uv (recommended)
uv sync --group dev

# Or using traditional pip
python -m venv evojump_env
source evojump_env/bin/activate  # On Windows: evojump_env\Scripts\activate
pip install -e ".[dev]"
```

3. **Verify your setup**:
```bash
# Run all tests
python run_all_tests.py --quick

# Run with coverage
python run_all_tests.py --coverage
```

## Development Workflow

### Test-Driven Development (TDD)

EvoJump follows strict test-driven development practices:

1. **Write tests first**: Before implementing new features, write comprehensive tests
2. **Use real data**: All tests must use real biological or synthetic data (no mocks)
3. **Maintain coverage**: Ensure 95%+ code coverage for all new code
4. **Run tests frequently**: Use `python run_all_tests.py --quick` for rapid feedback

### Code Quality Standards

#### Python Style Guide
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (no tabs)
- Limit line length to 88 characters (Black default)
- Use meaningful variable and function names
- Write comprehensive docstrings for all public APIs

#### Type Annotations
- Use type hints for all function parameters and return values
- Run MyPy to verify type correctness:
```bash
mypy src/ tests/
```

#### Code Formatting
We use Black for consistent code formatting:
```bash
# Check formatting
black --check --diff src/ tests/

# Apply formatting
black src/ tests/
```

#### Linting
We use Flake8 for style checking:
```bash
flake8 src/ tests/
```

### Documentation Standards

#### Docstring Format
Use Google-style docstrings with comprehensive documentation:

```python
def analyze_trajectory(
    data: np.ndarray,
    time_points: np.ndarray,
    model_type: str = "jump-diffusion"
) -> Dict[str, Any]:
    """
    Analyze developmental trajectory using stochastic process models.

    This function fits a specified stochastic process model to longitudinal
    developmental data and returns comprehensive statistical analysis.

    Args:
        data: Array of phenotypic measurements, shape (n_timepoints, n_features)
        time_points: Array of observation times, shape (n_timepoints,)
        model_type: Type of stochastic process model to fit. Options include:
            - "jump-diffusion": Standard jump-diffusion process
            - "ornstein-uhlenbeck": Mean-reverting process
            - "fractional-brownian": Long-range dependent process

    Returns:
        Dictionary containing:
            - "parameters": Fitted model parameters
            - "trajectories": Simulated developmental trajectories
            - "statistics": Summary statistics and goodness-of-fit
            - "diagnostics": Model diagnostic information

    Raises:
        ValueError: If data contains NaN values or invalid time points
        RuntimeError: If model fitting fails to converge

    Examples:
        >>> data = np.random.randn(100, 3)
        >>> times = np.linspace(0, 10, 100)
        >>> results = analyze_trajectory(data, times, model_type="jump-diffusion")
        >>> print(results["parameters"])
    """
    # Implementation here
    pass
```

#### Module Documentation
Every module should include:
- Module-level docstring describing purpose
- Import statements organized by type
- Clear section comments for major code blocks

## Making Contributions

### Branching Strategy

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the guidelines above

3. **Commit your changes**:
```bash
git add .
git commit -m "Add feature: brief description

Detailed description of changes:
- What was added/changed/fixed
- Why the change was necessary
- Any relevant issue numbers (#123)
"
```

4. **Push your branch**:
```bash
git push origin feature/your-feature-name
```

5. **Create a Pull Request** on GitHub

### Pull Request Guidelines

When submitting a pull request:

1. **Ensure all tests pass**:
```bash
python run_all_tests.py --all
```

2. **Verify code quality**:
```bash
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

3. **Update documentation** as needed

4. **Write a clear PR description**:
   - What does this PR do?
   - Why is this change needed?
   - How has it been tested?
   - Any breaking changes?
   - Related issue numbers

5. **Request review** from maintainers

### What to Contribute

We welcome contributions in many forms:

#### Bug Fixes
- Report bugs via GitHub Issues
- Include minimal reproducible example
- Submit fix with comprehensive tests

#### New Features
- Discuss major features in Issues first
- Implement with TDD approach
- Include comprehensive documentation
- Add examples demonstrating usage

#### Documentation Improvements
- Fix typos and clarify explanations
- Add usage examples
- Improve API documentation
- Create tutorials

#### Performance Improvements
- Profile code to identify bottlenecks
- Benchmark improvements
- Maintain numerical accuracy
- Document performance gains

#### Test Coverage
- Add tests for uncovered code
- Improve edge case testing
- Add integration tests
- Enhance performance tests

## Scientific Computing Guidelines

### Numerical Methods
- Use established numerical libraries (NumPy, SciPy)
- Validate algorithms with known test cases
- Handle floating-point precision appropriately
- Document numerical stability considerations
- Include references to scientific literature

### Statistical Methods
- Use appropriate statistical tests
- Validate statistical assumptions
- Report effect sizes and confidence intervals
- Apply multiple testing corrections when needed
- Document statistical power considerations

### Biological Data
- Support standard biological data formats
- Handle missing data appropriately
- Validate data quality and integrity
- Support metadata tracking
- Enable reproducible workflows

## Testing Guidelines

### Writing Tests

1. **Test Structure**:
```python
class TestFeatureName:
    """Tests for feature_name functionality."""

    def test_basic_functionality(self):
        """Test basic operation with standard input."""
        # Arrange
        data = create_test_data()
        
        # Act
        result = feature_function(data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
        np.testing.assert_allclose(result, expected_values)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty input
        with pytest.raises(ValueError):
            feature_function(np.array([]))
        
        # Test with single value
        result = feature_function(np.array([1.0]))
        assert result is not None

    def test_error_handling(self):
        """Test error conditions and exception handling."""
        with pytest.raises(ValueError, match="Invalid input"):
            feature_function(invalid_data)
```

2. **Test Data**:
   - Use synthetic biological data that mimics real patterns
   - Include edge cases (empty, single value, large datasets)
   - Test with various data distributions
   - Use fixed random seeds for reproducibility

3. **Test Coverage**:
   - Aim for 95%+ coverage on new code
   - Test both success and failure paths
   - Include integration tests
   - Test performance with large datasets

### Running Tests

```bash
# Quick test run (no coverage)
python run_all_tests.py --quick

# Full test with coverage
python run_all_tests.py --coverage --verbose

# All checks (tests + linting + docs)
python run_all_tests.py --all

# Performance benchmarks
python run_all_tests.py --benchmark

# Specific test file
pytest tests/test_datacore.py -v

# Specific test method
pytest tests/test_jumprope.py::TestJumpRope::test_fit_model -v
```

## Release Process

Releases are managed by project maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release tag
5. Build and upload to PyPI
6. Update documentation

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Documentation**: https://evojump.readthedocs.io/
- **Examples**: See `examples/` directory

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for each release
- GitHub contributors page
- Academic publications using contributed features

## License

By contributing to EvoJump, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to EvoJump!** Your work helps advance research in developmental and evolutionary biology.

