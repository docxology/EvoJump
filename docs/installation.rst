Installation Guide
==================

This guide provides comprehensive instructions for installing EvoJump and its dependencies.

Requirements
------------

**Python Version**
  * Python 3.9 or higher required (packaging bounds ``requires-python`` to ``>=3.9,<3.15``)
  * Tested primarily on Python 3.12 (project venv)

**System Requirements**
  * Operating System: Linux, macOS, Windows
  * RAM: Minimum 4GB recommended, 8GB+ for large datasets
  * Storage: 500MB free space for installation and examples
  * Network: Required for package downloads

Core Dependencies
-----------------

The following packages are installed automatically as runtime dependencies
(mirroring ``pyproject.toml``):

.. code-block::

   numpy>=1.21.0          # Numerical computing
   scipy>=1.7.0           # Scientific computing
   pandas>=1.3.0          # Data manipulation
   matplotlib>=3.5.0      # Plotting and visualization
   plotly>=5.0.0          # Interactive plots
   scikit-learn>=1.0.0    # Machine learning
   numba>=0.56.0          # JIT compilation
   dask>=2022.0.0         # Parallel computing
   h5py>=3.7.0            # HDF5 file support
   sqlalchemy>=1.4.0      # Database operations
   pyyaml>=6.0            # YAML support
   tqdm>=4.62.0           # Progress bars
   PyWavelets>=1.3.0      # Wavelet analysis
   networkx>=2.6.0        # Graph analysis
   statsmodels>=0.13.0    # Statistical models
   seaborn>=0.11.0        # Statistical visualization

Test and documentation tooling (also declared in ``pyproject.toml``):

.. code-block::

   pytest>=7.0.0          # Testing framework
   coverage>=7.0          # Coverage gate (direct coverage run)
   sphinx>=5.0.0          # Documentation
   sphinx-rtd-theme>=1.0.0 # Documentation theme

Quick Installation
------------------

Install the latest stable version using UV:

.. code-block:: bash

   uv add evojump

Install with optional dependencies:

.. code-block:: bash

   # Development dependencies
   uv sync --group dev

   # GPU acceleration (Linux/macOS)
   uv add evojump --extra gpu

   # Web interface
   uv add evojump --extra web

   # R integration
   uv add evojump --extra r-integration

Development Installation
------------------------

For contributors and advanced users:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/docxology/EvoJump.git
   cd EvoJump

   # Sync dependencies and install in development mode
   uv sync

   # Include development dependencies
   uv sync --group dev

   # Run tests
   pytest

   # Build documentation
   sphinx-build docs/ docs/_build/html

Verifying Installation
----------------------

Check that EvoJump is properly installed:

.. code-block:: python

   import evojump as ej
   print(f"EvoJump version: {ej.__version__}")

   # Test basic functionality
   import pandas as pd
   import numpy as np

   # Create test data
   data = pd.DataFrame({
       'time': [1, 2, 3, 4, 5] * 10,
       'phenotype1': np.random.normal(10, 2, 50)
   })

   # Test core functionality
   data_core = ej.DataCore.load_from_csv(pd.io.common.StringIO(data.to_csv()))
   print("✓ DataCore working")

   model = ej.JumpRope.fit(data_core)
   print("✓ JumpRope modeling working")

   analyzer = ej.LaserPlaneAnalyzer(model)
   result = analyzer.analyze_cross_section(3.0)
   print("✓ Cross-sectional analysis working")

   visualizer = ej.TrajectoryVisualizer()
   print("✓ Visualization system working")

   print("✓ All core functionality verified!")

Installation Troubleshooting
----------------------------

**ImportError: No module named 'evojump'**

Solution: Ensure the package is installed and the Python path includes the installation directory.

.. code-block:: bash

   # Check installation
   uv tree | grep evojump

   # If not found, reinstall
   uv add evojump

**ModuleNotFoundError: Specific dependency missing**

Solution: Install missing dependencies manually:

.. code-block:: bash

   uv add numpy scipy pandas matplotlib

**Permission denied during installation**

Solution: Use virtual environment with uv (recommended):

.. code-block:: bash

   # Create virtual environment with UV
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in virtual environment
   uv add evojump

**Memory errors during installation**

Solution: UV handles installation more efficiently than traditional pip. If issues persist, ensure sufficient system memory is available.

**Platform-specific issues**

**Linux:**
  - Ensure build tools are installed: ``sudo apt-get install build-essential``
  - For GPU support: Install CUDA toolkit from NVIDIA

**macOS:**
  - Install Xcode command line tools: ``xcode-select --install``
  - For GPU support: Install via conda (pip GPU packages may not work)

**Windows:**
  - Install Visual Studio Build Tools
  - Use conda for better compatibility
  - GPU support requires specific CUDA versions

Advanced Configuration
----------------------

.. note::
   EvoJump v0.2.0 supports no environment variables (no ``EVOJUMP_*``
   variables), no configuration file (no ``~/.evojump/config.yaml``), and
   no global configuration API (no ``evojump.config`` module). All
   behavior is controlled through function and method parameters.

**Performance Optimization**

For large datasets, chunk the data yourself before constructing analyzers
and parallelize independent fits with worker processes — see
:doc:`advanced_usage` for worked examples.

Getting Help
------------

**Documentation**: https://evojump.readthedocs.io/

**GitHub Issues**: https://github.com/docxology/EvoJump/issues

**Discussions**: https://github.com/docxology/EvoJump/discussions

**Email Support**: support@evojump.org

.. note::
   For the most up-to-date installation instructions, check the GitHub repository.

.. warning::
   Always install in a virtual environment to avoid dependency conflicts.
