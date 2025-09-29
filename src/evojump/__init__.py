"""
EvoJump: A Comprehensive Framework for Evolutionary Ontogenetic Analysis

This package provides a groundbreaking analytical framework for conceptualizing
evolutionary and developmental biology through a novel "cross-sectional laser" metaphor.
It treats ontogenetic development as a temporal progression where a "jumprope-like"
distribution sweeps across a fixed analytical plane (the laser), generating dynamic
cross-sectional views of phenotypic distributions throughout an organism's developmental
timeline.

The package enables researchers to visualize, analyze, and model the continuous
transformation of phenotypic landscapes during ontogenetic development, with particular
applications in evolutionary developmental biology, quantitative genetics, and systems
biology research.

Main Modules:
- DataCore: Data ingestion, validation, and preprocessing
- JumpRope: Jump-diffusion modeling for developmental trajectories
- LaserPlane: Cross-sectional analysis algorithms
- TrajectoryVisualizer: Advanced visualization system
- EvolutionSampler: Population-level analysis
- AnalyticsEngine: Comprehensive statistical analysis

Examples:
    >>> import evojump as ej
    >>> # Load developmental data
    >>> data = ej.DataCore.load_from_csv("developmental_data.csv")
    >>> # Create jump rope model
    >>> model = ej.JumpRope.fit(data)
    >>> # Analyze cross-sections
    >>> analyzer = ej.LaserPlaneAnalyzer(model)
    >>> # Visualize results
    >>> ej.TrajectoryVisualizer.plot_trajectories(model)
"""

__version__ = "0.1.0"
__author__ = "EvoJump Development Team"

# Import main modules for easy access
from . import datacore
from . import jumprope
from . import laserplane
from . import trajectory_visualizer
from . import evolution_sampler
from . import analytics_engine
from . import cli

__all__ = [
    "datacore",
    "jumprope",
    "laserplane",
    "trajectory_visualizer",
    "evolution_sampler",
    "analytics_engine",
    "cli",
]
