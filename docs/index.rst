EvoJump Documentation
=====================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/evojump/badge/?version=latest
   :target: https://evojump.readthedocs.io/
   :alt: Documentation Status

A Comprehensive Framework for Evolutionary Ontogenetic Analysis
----------------------------------------------------------------

EvoJump represents a groundbreaking analytical framework that conceptualizes evolutionary and developmental biology through a novel "cross-sectional laser" metaphor. This system treats ontogenetic development as a temporal progression where a "jumprope-like" distribution sweeps across a fixed analytical plane (the laser), generating dynamic cross-sectional views of phenotypic distributions throughout an organism's developmental timeline.

Core Features
~~~~~~~~~~~~~

**🔬 Scientific Analysis**
  * Advanced stochastic process modeling for developmental trajectories
  * Cross-sectional distribution analysis at specific timepoints
  * Jump detection and characterization for discrete developmental transitions
  * Evolutionary pattern recognition using machine learning approaches
  * Predictive modeling for developmental outcomes

**📊 Comprehensive Analytics**
  * Time series analysis with trend detection and seasonality analysis
  * Multivariate statistics including PCA, CCA, and cluster analysis
  * Bayesian inference with credible intervals and model comparison
  * Network analysis with centrality measures and community detection
  * Causal inference using Granger causality testing
  * Advanced dimensionality reduction (FastICA, t-SNE)
  * Spectral analysis for frequency domain insights
  * Nonlinear dynamics analysis (Lyapunov exponents, chaos detection)
  * Information theory analysis (entropy measures, mutual information)
  * Robust statistical methods resistant to outliers
  * Spatial analysis (Moran's I autocorrelation)

**🎨 Visualization & Animation**
  * Interactive 3D phenotypic landscape visualization
  * Animated developmental process sequences
  * Comparative multi-condition trajectory visualization
  * Real-time statistical analysis interface
  * Publication-quality plots and figures

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import evojump as ej
   import pandas as pd

   # Load developmental data
   data = ej.DataCore.load_from_csv("developmental_data.csv", time_column='time')

   # Fit jump-diffusion model
   model = ej.JumpRope.fit(data, model_type='jump-diffusion')

   # Analyze cross-sections
   analyzer = ej.LaserPlaneAnalyzer(model)
   results = analyzer.analyze_cross_section(time_point=10.0)

   # Create visualizations
   visualizer = ej.TrajectoryVisualizer()
   visualizer.plot_trajectories(model)

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install evojump

For development:

.. code-block:: bash

   git clone https://github.com/evojump/evojump.git
   cd evojump
   pip install -e .

Documentation Contents
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples
   api_reference
   advanced_usage
   advanced_methods
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   architecture
   contributing
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
