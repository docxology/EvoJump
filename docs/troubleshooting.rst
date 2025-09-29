Troubleshooting Guide
=====================

This guide provides solutions to common issues encountered when using EvoJump.

Installation Issues
-------------------

**ModuleNotFoundError: No module named 'evojump'**

**Symptoms:**
  * ImportError when trying to import evojump
  * Package not found in Python path

**Solutions:**

1. **Verify Installation**

   .. code-block:: bash

      pip list | grep evojump

   If not found, reinstall:

   .. code-block:: bash

      pip install evojump

2. **Check Python Path**

   .. code-block:: python

      import sys
      print(sys.path)

   Ensure the site-packages directory containing evojump is in the path.

3. **Virtual Environment Issues**

   If using a virtual environment, ensure it's activated:

   .. code-block:: bash

      source venv/bin/activate  # Linux/macOS
      # or
      venv\Scripts\activate     # Windows

**ImportError: Missing dependencies**

**Symptoms:**
  * ImportError for specific modules (numpy, scipy, pandas, etc.)
  * Runtime errors due to missing packages

**Solutions:**

1. **Install Missing Dependencies**

   .. code-block:: bash

      pip install numpy scipy pandas matplotlib plotly scikit-learn

2. **Check Version Compatibility**

   Ensure you have compatible versions:

   .. code-block:: python

      import evojump
      print(evojump.__version__)

3. **Use Requirements File**

   .. code-block:: bash

      pip install -r requirements.txt

**Permission denied during installation**

**Symptoms:**
  * Permission denied when installing packages
  * Cannot write to system directories

**Solutions:**

1. **Use Virtual Environment**

   .. code-block:: bash

      python -m venv evojump_env
      source evojump_env/bin/activate
      pip install evojump

2. **Install for Current User**

   .. code-block:: bash

      pip install --user evojump

3. **Use sudo (not recommended)**

   .. code-block:: bash

      sudo pip install evojump

Data Loading Issues
-------------------

**File not found errors**

**Symptoms:**
  * FileNotFoundError when loading data files
  * Path-related errors

**Solutions:**

1. **Check File Path**

   .. code-block:: python

      from pathlib import Path
      data_file = Path("data.csv")
      if not data_file.exists():
          print(f"File not found: {data_file.absolute()}")

2. **Use Absolute Paths**

   .. code-block:: python

      import os
      absolute_path = os.path.abspath("data.csv")
      data = ej.DataCore.load_from_csv(absolute_path)

3. **Check Working Directory**

   .. code-block:: python

      import os
      print(f"Current working directory: {os.getcwd()}")

**Encoding errors**

**Symptoms:**
  * UnicodeDecodeError when reading CSV files
  * Character encoding issues

**Solutions:**

1. **Specify Encoding**

   .. code-block:: python

      data = ej.DataCore.load_from_csv("data.csv", encoding='utf-8')

2. **Check File Encoding**

   Use a text editor to check the file encoding and convert if necessary.

**Column not found errors**

**Symptoms:**
  * KeyError or AttributeError for missing columns
  * Column name mismatches

**Solutions:**

1. **Check Column Names**

   .. code-block:: python

      import pandas as pd
      df = pd.read_csv("data.csv")
      print("Available columns:", df.columns.tolist())

2. **Case Sensitivity**

   Ensure column names match exactly (case-sensitive).

3. **Specify Column Names Explicitly**

   .. code-block:: python

      data = ej.DataCore.load_from_csv(
          "data.csv",
          time_column='Time',  # Note the capital T
          phenotype_columns=['Phenotype1', 'Phenotype2']
      )

Model Fitting Issues
--------------------

**Convergence failures**

**Symptoms:**
  * Optimization fails to converge
  * Warning messages about failed optimization
  * Poor model fit quality

**Solutions:**

1. **Check Data Quality**

   .. code-block:: python

      quality = data_core.validate_data_quality()
      print(f"Missing data: {quality['missing_data_percentage']['dataset_0']:.1f}%")
      print(f"Outliers: {quality['outlier_percentage']['dataset_0']:.1f}%")

2. **Preprocess Data**

   .. code-block:: python

      data_core.preprocess_data(
          normalize=True,
          remove_outliers=True,
          interpolate_missing=True
      )

3. **Try Different Model Types**

   .. code-block:: python

      # Try Ornstein-Uhlenbeck instead of jump-diffusion
      model = ej.JumpRope.fit(data_core, model_type='ornstein-uhlenbeck')

4. **Adjust Optimization Parameters**

   .. code-block:: python

      # Custom optimization settings
      model = ej.JumpRope.fit(data_core, max_iter=10000, tolerance=1e-8)

**Poor model fit**

**Symptoms:**
  * Model parameters don't make biological sense
  * Simulated trajectories don't match data
  * High residual errors

**Solutions:**

1. **Check Data Scale**

   .. code-block:: python

      print(f"Data range: {data.min():.2f} to {data.max():.2f}")
      print(f"Data standard deviation: {data.std():.2f}")

2. **Transform Data if Needed**

   .. code-block:: python

      # Log transform for multiplicative processes
      import numpy as np
      log_data = np.log(data + 1)  # Add small constant to avoid log(0)

3. **Use Geometric Jump-Diffusion for Growth Processes**

   .. code-block:: python

      model = ej.JumpRope.fit(data_core, model_type='geometric-jump-diffusion')

**Memory issues with large datasets**

**Symptoms:**
  * MemoryError during model fitting
  * OutOfMemory exceptions
  * System becomes unresponsive

**Solutions:**

1. **Reduce Dataset Size**

   .. code-block:: python

      # Sample subset of data
      subset_data = data.sample(frac=0.1, random_state=42)

2. **Use Chunked Processing**

   .. code-block:: python

      # Process data in chunks
      chunk_size = 1000
      for i in range(0, len(data), chunk_size):
          chunk = data.iloc[i:i+chunk_size]
          # Process chunk...

3. **Enable Memory-Efficient Algorithms**

   .. code-block:: python

      # Use memory-efficient fitting
      model = ej.JumpRope.fit(data_core, memory_efficient=True)

4. **Increase System Memory**

   Close other applications or upgrade system memory.

Analysis Issues
---------------

**Statistical test failures**

**Symptoms:**
  * RuntimeWarning about test assumptions
  * NaN or infinite p-values
  * Test statistic errors

**Solutions:**

1. **Check Test Assumptions**

   .. code-block:: python

      # Check normality
      from scipy import stats
      statistic, p_value = stats.shapiro(data)
      print(f"Normality test p-value: {p_value:.3f}")

   # Check equal variances
      statistic, p_value = stats.levene(group1, group2)
      print(f"Equal variance test p-value: {p_value:.3f}")

2. **Use Non-Parametric Tests**

   .. code-block:: python

      # Use Mann-Whitney instead of t-test
      result = ej.LaserPlaneAnalyzer(model).compare_distributions(
          data1, data2, test='mann_whitney'
      )

3. **Transform Data**

   .. code-block:: python

      # Log transform for skewed data
      log_data1 = np.log(data1 + 1)
      log_data2 = np.log(data2 + 1)

**Bootstrap confidence interval failures**

**Symptoms:**
  * Bootstrap failures with small datasets
  * Warning messages about insufficient data

**Solutions:**

1. **Increase Sample Size**

   .. code-block:: python

      # Generate more trajectories
      trajectories = model.generate_trajectories(n_samples=1000)

2. **Use Parametric Methods**

   .. code-block:: python

      # Use analytical confidence intervals
      result = analyzer.analyze_cross_section(
          time_point=5.0,
          bootstrap_samples=0  # Disable bootstrap
      )

3. **Reduce Confidence Level**

   .. code-block:: python

      # Use 90% instead of 95% confidence
      result = analyzer.analyze_cross_section(
          time_point=5.0,
          confidence_level=0.90
      )

Visualization Issues
--------------------

**Plot generation failures**

**Symptoms:**
  * Matplotlib errors during plotting
  * Empty or corrupted plot files
  * Display issues

**Solutions:**

1. **Check Matplotlib Backend**

   .. code-block:: python

      import matplotlib
      print(f"Backend: {matplotlib.get_backend()}")

   # Use non-interactive backend
      matplotlib.use('Agg')

2. **Install Missing Dependencies**

   .. code-block:: bash

      pip install matplotlib plotly

3. **Update Matplotlib**

   .. code-block:: bash

      pip install --upgrade matplotlib

**Animation generation failures**

**Symptoms:**
  * Animation creation fails
  * Corrupted GIF files
  * Memory issues during animation

**Solutions:**

1. **Reduce Animation Complexity**

   .. code-block:: python

      # Reduce number of frames
      anim = visualizer.create_animation(
          model,
          n_frames=20,  # Reduced from default
          fps=10        # Reduced from default
      )

2. **Use Alternative Writers**

   .. code-block:: python

      # Use different animation writer
      anim.save('animation.gif', writer='ffmpeg', fps=10)

3. **Increase Memory**

   Close other applications or increase system memory.

**Interactive plot issues**

**Symptoms:**
  * Plotly plots not displaying
  * JavaScript errors in browser
  * Performance issues with large datasets

**Solutions:**

1. **Check Plotly Installation**

   .. code-block:: python

      import plotly
      print(f"Plotly version: {plotly.__version__}")

2. **Use Static Plots**

   .. code-block:: python

      # Fall back to matplotlib for static plots
      fig = visualizer.plot_trajectories(model, interactive=False)
      fig.savefig('trajectories.png')

3. **Optimize for Performance**

   .. code-block:: python

      # Reduce data points for interactive plots
      fig = visualizer.plot_trajectories(model, n_trajectories=20)

Performance Issues
------------------

**Slow model fitting**

**Symptoms:**
  * Model fitting takes excessive time
  * Optimization algorithms don't converge

**Solutions:**

1. **Profile Performance**

   .. code-block:: python

      import cProfile
      profiler = cProfile.Profile()
      profiler.enable()

      model = ej.JumpRope.fit(data_core)

      profiler.disable()
      profiler.print_stats()

2. **Use Faster Algorithms**

   .. code-block:: python

      # Use analytical fitting when possible
      model = ej.JumpRope.fit(data_core, method='analytical')

3. **Reduce Data Size**

   .. code-block:: python

      # Subsample data for faster fitting
      small_data = data_core.time_series_data[0].data.sample(frac=0.5)
      small_data_core = ej.DataCore([ej.TimeSeriesData(small_data, 'time', ['phenotype'])])

**Memory usage issues**

**Symptoms:**
  * High memory consumption
  * Memory errors during processing
  * System slowdown

**Solutions:**

1. **Monitor Memory Usage**

   .. code-block:: python

      import psutil
      import os

      process = psutil.Process(os.getpid())
      print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

2. **Use Memory-Efficient Methods**

   .. code-block:: python

      # Use chunked processing
      for chunk in pd.read_csv('large_file.csv', chunksize=10000):
          process_chunk(chunk)

3. **Enable Garbage Collection**

   .. code-block:: python

      import gc
      gc.collect()

**Parallel processing issues**

**Symptoms:**
  * Threading errors
  * Race conditions
  * Inconsistent results

**Solutions:**

1. **Check Thread Safety**

   Ensure code is thread-safe before parallelizing.

2. **Use Process-Based Parallelism**

   .. code-block:: python

      from multiprocessing import Pool

      def process_data(data_chunk):
          return ej.JumpRope.fit(data_chunk)

      with Pool(4) as pool:
          results = pool.map(process_data, data_chunks)

3. **Debug Threading Issues**

   .. code-block:: python

      import threading
      import time

      def worker():
          print(f"Thread {threading.current_thread().name} starting")
          # Your parallel code here
          time.sleep(1)
          print(f"Thread {threading.current_thread().name} finished")

      threads = [threading.Thread(target=worker) for _ in range(4)]
      for t in threads:
          t.start()
      for t in threads:
          t.join()

Error Messages
--------------

**Common Error Patterns**

**"Insufficient data for analysis"**

**Cause:** Not enough data points for statistical analysis
**Solution:** Ensure you have at least 10-20 data points per time point

**"Model parameters out of bounds"**

**Cause:** Optimization algorithm found invalid parameter values
**Solution:** Check data quality and try different model types or initial values

**"Singular matrix"**

**Cause:** Data matrix is not invertible (collinear variables)
**Solution:** Remove highly correlated variables or use regularization

**"Convergence failed"**

**Cause:** Optimization algorithm couldn't find optimal parameters
**Solution:** Try different optimization methods or initial parameter values

**"Memory allocation failed"**

**Cause:** Insufficient memory for the operation
**Solution:** Reduce data size, use memory-efficient algorithms, or add more RAM

Getting Help
------------

**Debug Information**

When reporting issues, include:

.. code-block:: python

   import evojump as ej
   import sys

   print("=== EvoJump Debug Information ===")
   print(f"EvoJump version: {ej.__version__}")
   print(f"Python version: {sys.version}")
   print(f"Platform: {sys.platform}")

   # Check dependencies
   try:
       import numpy as np
       print(f"NumPy: {np.__version__}")
   except ImportError:
       print("NumPy: Not installed")

   try:
       import scipy
       print(f"SciPy: {scipy.__version__}")
   except ImportError:
       print("SciPy: Not installed")

   try:
       import pandas
       print(f"Pandas: {pandas.__version__}")
   except ImportError:
       print("Pandas: Not installed")

   try:
       import matplotlib
       print(f"Matplotlib: {matplotlib.__version__}")
   except ImportError:
       print("Matplotlib: Not installed")

   try:
       import plotly
       print(f"Plotly: {plotly.__version__}")
   except ImportError:
       print("Plotly: Not installed")

**Reporting Issues**

When reporting bugs or issues:

1. **Include Debug Information** (see above)
2. **Describe the Problem** - What were you trying to do?
3. **Include Error Messages** - Copy the full error traceback
4. **Provide Minimal Example** - Smallest code that reproduces the issue
5. **Expected vs Actual Behavior** - What should happen vs what happened

**Support Channels**

* **GitHub Issues**: https://github.com/evojump/evojump/issues
* **Discussions**: https://github.com/evojump/evojump/discussions
* **Documentation**: https://evojump.readthedocs.io/
* **Email**: support@evojump.org

**Known Issues and Limitations**

**Large Dataset Performance**
  * Datasets > 1M samples may require chunked processing
  * GPU acceleration recommended for intensive computations
  * Consider using Dask for distributed processing

**Statistical Test Assumptions**
  * Some tests assume normality or equal variances
  * Check assumptions before interpreting results
  * Use robust methods when assumptions are violated

**Model Selection**
  * No single model fits all biological systems
  * Try multiple model types and compare fits
  * Use domain knowledge to guide model selection

**Visualization Limitations**
  * 3D plots may be slow with many trajectories
  * Interactive plots require compatible browsers
  * Animation generation can be memory-intensive

**Platform-Specific Issues**

**Linux:**
  * Ensure build tools are installed for compilation
  * GPU support requires NVIDIA drivers and CUDA

**macOS:**
  * Some packages may require Homebrew installation
  * GPU support may be limited

**Windows:**
  * Use Anaconda for better dependency management
  * Some performance optimizations may not be available

**Workarounds for Common Issues**

**"Method not implemented" errors**

Some advanced features may show placeholder implementations. Use alternative methods or wait for full implementation.

**"Feature not available" warnings**

Some optional dependencies may not be available. Install optional packages or use alternative features.

**"Performance warning" messages**

EvoJump may suggest optimizations for better performance. Consider the recommendations for large datasets.

**Version Compatibility**

EvoJump follows semantic versioning. Breaking changes are only introduced in major version updates (e.g., 1.0.0).

This troubleshooting guide covers the most common issues. If you encounter problems not listed here, please report them to the development team.
