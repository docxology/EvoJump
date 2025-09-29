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

# Import main classes for direct access
from .datacore import DataCore, TimeSeriesData, MetadataManager
from .jumprope import JumpRope, ModelParameters
from .laserplane import LaserPlaneAnalyzer, CrossSectionResult
from .trajectory_visualizer import TrajectoryVisualizer, PlotConfig
from .evolution_sampler import EvolutionSampler, PopulationStatistics
from .analytics_engine import AnalyticsEngine, TimeSeriesResult

__all__ = [
    # Modules
    "datacore",
    "jumprope",
    "laserplane",
    "trajectory_visualizer",
    "evolution_sampler",
    "analytics_engine",
    "cli",
    # Classes
    "DataCore",
    "TimeSeriesData",
    "MetadataManager",
    "JumpRope",
    "ModelParameters",
    "LaserPlaneAnalyzer",
    "CrossSectionResult",
    "TrajectoryVisualizer",
    "PlotConfig",
    "EvolutionSampler",
    "PopulationStatistics",
    "AnalyticsEngine",
    "TimeSeriesResult",
]

