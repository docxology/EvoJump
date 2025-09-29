Contributing Guide
==================

We welcome contributions to EvoJump! This guide explains how to contribute effectively to the project.

Getting Started
---------------

**Prerequisites**
  * Python 3.8+ development environment
  * Git version control system
  * Familiarity with scientific Python ecosystem
  * Understanding of evolutionary biology concepts (helpful but not required)

**Development Setup**

1. **Fork the Repository**

   .. code-block:: bash

      git clone https://github.com/your-username/evojump.git
      cd evojump

2. **Create Development Environment**

   .. code-block:: bash

      # Create virtual environment
      python -m venv evojump_dev
      source evojump_dev/bin/activate  # Linux/macOS
      # or
      evojump_dev\Scripts\activate     # Windows

      # Install in development mode
      pip install -e ".[dev]"

3. **Verify Setup**

   .. code-block:: python

      import evojump as ej
      print(f"EvoJump version: {ej.__version__}")
      print("Development environment ready!")

Development Workflow
--------------------

**Branch Strategy**

* **main**: Stable release branch (protected)
* **develop**: Integration branch for new features
* **feature/***: Feature development branches
* **bugfix/***: Bug fix branches
* **hotfix/***: Critical bug fix branches

**Creating Feature Branches**

.. code-block:: bash

   # Create and switch to feature branch
   git checkout -b feature/amazing-new-feature

   # Or for bug fixes
   git checkout -b bugfix/issue-description

**Code Style Guidelines**

Follow PEP 8 and project-specific conventions:

**Imports**
  * Standard library imports first
  * Third-party imports second
  * Local imports last
  * Group related imports together

.. code-block:: python

   # Correct import order
   import os
   import sys
   from pathlib import Path

   import numpy as np
   import pandas as pd
   import scipy.stats as stats

   from evojump.datacore import DataCore
   from evojump.jumprope import JumpRope

**Function and Variable Names**
  * Use descriptive names
  * Follow snake_case convention
  * Use ALL_CAPS for constants

.. code-block:: python

   # Good naming
   def calculate_evolutionary_rate(population_data, time_points):
       EVOLUTIONARY_CONSTANT = 0.01
       rate = EVOLUTIONARY_CONSTANT * len(population_data)
       return rate

   # Avoid
   def calc(x, y):
       c = 0.01
       return c * len(x)

**Documentation**
  * Use Google/NumPy style docstrings
  * Include parameter and return value documentation
  * Add examples for complex functions

.. code-block:: python

   def analyze_developmental_trajectory(data, model_type='jump-diffusion'):
       """Analyze developmental trajectory using stochastic modeling.

       This function performs comprehensive analysis of developmental
       trajectories using jump-diffusion stochastic processes.

       Parameters
       ----------
       data : DataCore
           Input developmental data
       model_type : str, optional
           Type of stochastic model ('jump-diffusion', 'ornstein-uhlenbeck')

       Returns
       -------
       dict
           Dictionary containing analysis results including:
           - model_parameters: Fitted model parameters
           - trajectories: Generated sample trajectories
           - cross_sections: Cross-sectional analysis results

       Examples
       --------
       >>> data = ej.DataCore.load_from_csv('developmental_data.csv')
       >>> results = analyze_developmental_trajectory(data)
       >>> print(f"Model fit: {results['model_parameters']['equilibrium']:.3f}")
       """
       pass

**Testing Requirements**

**Test-Driven Development**
  * Write tests before implementing features
  * Maintain >95% test coverage
  * Use real data in tests (no mocks)
  * Test both success and failure cases

.. code-block:: python

   def test_analyze_developmental_trajectory():
       """Test developmental trajectory analysis."""
       # Create test data
       test_data = create_synthetic_developmental_data()

       # Test successful analysis
       result = analyze_developmental_trajectory(test_data)
       assert 'model_parameters' in result
       assert 'trajectories' in result
       assert result['model_parameters']['equilibrium'] > 0

       # Test error handling
       with pytest.raises(ValueError):
           analyze_developmental_trajectory(None)

**Test Data Creation**
  Create realistic synthetic data for testing:

.. code-block:: python

   def create_synthetic_developmental_data():
       """Create synthetic developmental data for testing."""
       np.random.seed(42)

       # Generate realistic developmental pattern
       time_points = np.linspace(0, 20, 21)
       base_pattern = 10 + 5 * np.sin(time_points * 0.3) + time_points * 0.2
       noise = np.random.normal(0, 1, len(time_points))

       data = pd.DataFrame({
           'time': time_points,
           'phenotype': base_pattern + noise
       })

       return ej.DataCore.load_from_csv(
           pd.io.common.StringIO(data.to_csv()),
           time_column='time'
       )

**Running Tests**

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=evojump --cov-report=html

   # Run specific test file
   pytest tests/test_datacore.py

   # Run with verbose output
   pytest -v

   # Run performance tests
   pytest tests/ -k "performance" --benchmark-only

**Code Quality Tools**

**Linting**
  * Use flake8 for code style checking
  * Use black for code formatting
  * Use isort for import sorting

.. code-block:: bash

   # Check code style
   flake8 src/evojump/

   # Format code
   black src/evojump/

   # Sort imports
   isort src/evojump/

**Type Checking**
  * Use mypy for static type checking

.. code-block:: bash

   # Check types
   mypy src/evojump/

**Documentation**
  * Use sphinx for documentation building
  * Keep docstrings up to date

.. code-block:: bash

   # Build documentation
   sphinx-build docs/ docs/_build/html

   # Check documentation links
   sphinx-build docs/ docs/_build/html -b linkcheck

Contributing Process
--------------------

**1. Choose an Issue**
  * Look for issues labeled "good first issue" or "help wanted"
  * Check existing issues for unassigned tasks
  * Create new issues for bugs or feature requests

**2. Create Feature Branch**
  * Create a descriptive branch name
  * Base your branch on the develop branch

.. code-block:: bash

   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name

**3. Implement Changes**
  * Write tests first (TDD approach)
  * Implement the feature
  * Update documentation
  * Ensure all tests pass

**4. Submit Pull Request**
  * Push your branch to GitHub
  * Create a pull request against the develop branch
  * Fill out the pull request template

**5. Code Review**
  * Address reviewer feedback
  * Update code as needed
  * Ensure CI/CD passes

**6. Merge**
  * Maintainers will merge approved pull requests
  * Delete your feature branch after merge

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

**PR Title**
  * Use clear, descriptive titles
  * Follow conventional commit format

.. code-block::

   # Good titles
   feat: add Bayesian analysis methods
   fix: resolve memory leak in trajectory generation
   docs: update installation guide
   test: add comprehensive test coverage

**PR Description**
  * Explain what the PR does
  * Include motivation and context
  * Reference related issues
  * Document breaking changes

.. code-block::

   ## Description

   This PR adds comprehensive Bayesian analysis methods to the AnalyticsEngine,
   including posterior sampling, credible interval calculation, and model
   comparison using BIC and AIC criteria.

   ## Motivation

   Users need advanced statistical inference capabilities for analyzing
   developmental data with uncertainty quantification.

   ## Changes

   - Added `bayesian_analysis()` method to AnalyticsEngine
   - Implemented `BayesianResult` dataclass
   - Added model comparison utilities
   - Updated documentation and examples

   ## Testing

   - Added comprehensive test suite for Bayesian methods
   - All existing tests still pass
   - Test coverage maintained above 95%

**Code Changes**
  * Keep PRs focused and atomic
  * Avoid mixing multiple features in one PR
  * Ensure backward compatibility

**Documentation Updates**
  * Update docstrings for new/modified APIs
  * Add examples demonstrating new features
  * Update README and guides as needed

Types of Contributions
-----------------------

**🐛 Bug Reports**
  * Clear description of the bug
  * Steps to reproduce
  * Expected vs actual behavior
  * Environment information

**✨ Feature Requests**
  * Detailed feature description
  * Use cases and motivation
  * Proposed API design
  * Alternative solutions considered

**📚 Documentation Improvements**
  * Fix typos and grammatical errors
  * Improve clarity and examples
  * Add missing documentation
  * Update outdated information

**🧪 Test Contributions**
  * Add tests for untested functionality
  * Improve test coverage
  * Add performance benchmarks
  * Create integration tests

**🎨 Code Style Improvements**
  * Refactor for better readability
  * Optimize performance
  * Improve error handling
  * Add type hints

**🔧 Tooling and Infrastructure**
  * CI/CD improvements
  * Development tool enhancements
  * Documentation build improvements
  * Testing infrastructure

**📖 Examples and Tutorials**
  * Create new examples
  * Improve existing examples
  * Add tutorial notebooks
  * Create use case demonstrations

**🔬 Research Contributions**
  * Implement new statistical methods
  * Add novel analysis algorithms
  * Contribute domain-specific features
  * Validate methods with real data

**🌐 Community Support**
  * Answer user questions
  * Help with issue triage
  * Improve user experience
  * Build community resources

Development Best Practices
--------------------------

**Scientific Computing Standards**
  * Validate numerical algorithms with known test cases
  * Handle floating-point precision appropriately
  * Document convergence criteria and stability
  * Include references to scientific literature

**Performance Optimization**
  * Profile code before optimizing
  * Use vectorized operations when possible
  * Consider memory usage for large datasets
  * Implement efficient caching strategies

**Error Handling**
  * Provide informative error messages
  * Handle edge cases gracefully
  * Allow users to recover from errors
  * Log errors appropriately

**Backward Compatibility**
  * Maintain API compatibility when possible
  * Deprecate old interfaces before removing
  * Provide migration guides for breaking changes
  * Version APIs appropriately

**Testing Best Practices**
  * Test with real biological data when possible
  * Include edge cases and error conditions
  * Test integration between modules
  * Use descriptive test names

**Documentation Standards**
  * Write comprehensive docstrings
  * Include examples for complex functions
  * Document limitations and assumptions
  * Keep documentation synchronized with code

Code Review Process
-------------------

**Reviewer Responsibilities**
  * Review code for correctness and style
  * Check test coverage and quality
  * Verify documentation completeness
  * Ensure backward compatibility

**Author Responsibilities**
  * Address all reviewer comments
  * Update code as requested
  * Ensure CI/CD passes
  * Keep PR updated with develop branch

**Review Checklist**
  * [ ] Code follows style guidelines
  * [ ] Tests are comprehensive and pass
  * [ ] Documentation is complete and accurate
  * [ ] Performance impact is acceptable
  * [ ] Backward compatibility is maintained
  * [ ] New features have examples
  * [ ] Error handling is robust

**Merging Guidelines**
  * PRs must have at least one approval
  * All CI/CD checks must pass
  * No merge conflicts with target branch
  * Maintainers have final merge authority

Community Guidelines
--------------------

**Be Respectful**
  * Treat all contributors with respect
  * Use inclusive language
  * Give constructive feedback
  * Acknowledge good work

**Be Collaborative**
  * Work together to solve problems
  * Share knowledge and expertise
  * Help newcomers get started
  * Build on each other's ideas

**Be Professional**
  * Use appropriate language
  * Stay on topic in discussions
  * Respect different opinions
  * Focus on technical merit

**Communication**
  * Use clear and concise language
  * Provide context for questions
  * Share relevant information
  * Respond promptly to inquiries

**Quality Standards**
  * Maintain high code quality
  * Follow established patterns
  * Write comprehensive tests
  * Keep documentation current

Recognition and Rewards
-----------------------

**Contributor Recognition**
  * Contributors are listed in README
  * Special recognition for major contributions
  * Feature naming opportunities for significant contributions
  * Conference presentation opportunities

**Hall of Fame**
  * Contributors with 10+ merged PRs
  * Maintainers and core developers
  * Early adopters and supporters
  * Community leaders and advocates

**Contribution Badges**
  * Bug Hunter: 5+ bug reports
  * Code Contributor: 3+ merged PRs
  * Documentation Hero: 5+ documentation improvements
  * Test Master: 10+ new tests added

**Special Roles**
  * **Core Developer**: Full write access to repository
  * **Maintainer**: Code review and merge authority
  * **Community Manager**: User support and outreach
  * **Documentation Lead**: Documentation maintenance

Getting Help
------------

**Development Questions**
  * Check existing documentation
  * Search GitHub issues and discussions
  * Ask in GitHub discussions
  * Create new issue if needed

**Technical Issues**
  * Provide minimal reproducible example
  * Include environment information
  * Describe expected vs actual behavior
  * Reference related issues/PRs

**Community Support**
  * GitHub Discussions for questions and help
  * Stack Overflow with "evojump" tag
  * Scientific Python community resources
  * Domain-specific forums and mailing lists

**Mentorship Program**
  * Pair new contributors with experienced developers
  * Provide guidance on contribution process
  * Help with technical challenges
  * Regular check-ins and feedback

Success Stories
---------------

**Example Contributions**

*"Added GPU acceleration support for trajectory simulation, improving performance by 10x for large datasets."*

*"Implemented robust statistical methods for handling outliers in developmental data analysis."*

*"Created comprehensive documentation with tutorials and examples, helping new users get started quickly."*

*"Fixed memory leak in animation generation, preventing crashes during long-running analyses."*

*"Added support for real-time data streaming, enabling live developmental monitoring."*

Your contributions help make EvoJump better for everyone! Thank you for being part of the community. 🎉
