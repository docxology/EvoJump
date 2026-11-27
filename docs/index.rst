EvoJump Documentation
=====================

.. image:: https://img.shields.io/badge/python-3.11_to_3.14-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

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

   import numpy as np
   from evojump.jumprope import ModelParameters, OrnsteinUhlenbeckJump

   parameters = ModelParameters(diffusion=0.15, equilibrium=0.5)
   process = OrnsteinUhlenbeckJump(parameters, rng=np.random.default_rng(27))
   paths = process.simulate(2.0, np.linspace(0.0, 2.0, 21), n_paths=8)
   print(paths.shape)

The complete experiment and publication source are in the
`manuscript directory <https://github.com/docxology/EvoJump/tree/main/docs/manuscript>`_.

Installation
~~~~~~~~~~~~

EvoJump is not published on PyPI; install it from the repository:

.. code-block:: bash

   pip install "git+https://github.com/docxology/EvoJump.git"

For development:

.. code-block:: bash

   git clone https://github.com/docxology/EvoJump.git
   cd EvoJump
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
