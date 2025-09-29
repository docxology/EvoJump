Installation Guide
==================

This guide provides comprehensive instructions for installing EvoJump and its dependencies.

Requirements
------------

**Python Version**
  * Python 3.8 or higher required
  * Tested on Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

**System Requirements**
  * Operating System: Linux, macOS, Windows
  * RAM: Minimum 4GB recommended, 8GB+ for large datasets
  * Storage: 500MB free space for installation and examples
  * Network: Required for package downloads

Core Dependencies
-----------------

The following packages are automatically installed:

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
   pyyaml>=6.0            # YAML configuration
   tqdm>=4.62.0           # Progress bars
   pytest>=7.0.0          # Testing framework
   pytest-cov>=3.0.0      # Coverage reporting
   sphinx>=5.0.0          # Documentation
   sphinx-rtd-theme>=1.0.0 # Documentation theme

Quick Installation
------------------

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install evojump

Install with optional dependencies:

.. code-block:: bash

   # Development dependencies
   pip install evojump[dev]

   # GPU acceleration (Linux/macOS)
   pip install evojump[gpu]

   # Web interface
   pip install evojump[web]

   # R integration
   pip install evojump[r-integration]

Development Installation
-----------------------

For contributors and advanced users:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/evojump/evojump.git
   cd evojump

   # Install in development mode
   pip install -e .

   # Install development dependencies
   pip install -e ".[dev]"

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
   pip list | grep evojump

   # If not found, reinstall
   pip install evojump

**ModuleNotFoundError: Specific dependency missing**

Solution: Install missing dependencies manually:

.. code-block:: bash

   pip install numpy scipy pandas matplotlib

**Permission denied during installation**

Solution: Use virtual environment or install with user flag:

.. code-block:: bash

   # Create virtual environment
   python -m venv evojump_env
   source evojump_env/bin/activate  # On Windows: evojump_env\Scripts\activate

   # Install in virtual environment
   pip install evojump

   # Or install for current user only
   pip install --user evojump

**Memory errors during installation**

Solution: Install with reduced parallelization:

.. code-block:: bash

   pip install evojump --no-build-isolation
   # Or use conda for better dependency management
   conda install -c conda-forge evojump

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

**Environment Variables**

Set environment variables for custom configuration:

.. code-block:: bash

   export EVOJUMP_LOG_LEVEL=DEBUG
   export EVOJUMP_CACHE_DIR=/path/to/cache
   export EVOJUMP_PLOT_BACKEND=plotly  # or matplotlib
   export EVOJUMP_NUM_THREADS=4

**Configuration File**

Create a configuration file for persistent settings:

.. code-block:: yaml
   # ~/.evojump/config.yaml
   logging:
     level: INFO
     file: /path/to/evojump.log

   plotting:
     backend: plotly
     style: ggplot
     dpi: 150

   computation:
     num_threads: 4
     cache_enabled: true
     cache_dir: /tmp/evojump_cache

**Performance Optimization**

For large datasets:

.. code-block:: python

   import evojump as ej

   # Enable parallel processing
   ej.config.set_num_threads(8)

   # Enable caching
   ej.config.enable_cache('/path/to/cache')

   # Set memory limits
   ej.config.set_memory_limit('4GB')

Getting Help
------------

**Documentation**: https://evojump.readthedocs.io/

**GitHub Issues**: https://github.com/evojump/evojump/issues

**Discussions**: https://github.com/evojump/evojump/discussions

**Email Support**: support@evojump.org

.. note::
   For the most up-to-date installation instructions, check the GitHub repository.

.. warning::
   Always install in a virtual environment to avoid dependency conflicts.
