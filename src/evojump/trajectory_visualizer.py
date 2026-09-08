"""
Trajectory Visualizer: Advanced Visualization System

This module provides advanced visualization capabilities for developmental trajectories
including interactive plotting, animation sequences, and comparative visualization tools
for multi-sample analysis. Supports both 2D and 3D visualization of phenotypic landscapes
showing how distributions change over developmental time.

Classes:
    TrajectoryVisualizer: Main visualization class
    PlotConfig: Configuration for plot appearance and behavior
    AnimationController: Controls animation sequences

Examples:
    >>> # Create basic trajectory plot
    >>> TrajectoryVisualizer.plot_trajectories(model)
    >>> # Create interactive 3D landscape
    >>> TrajectoryVisualizer.plot_landscapes(model, interactive=True)
    >>> # Generate animation
    >>> TrajectoryVisualizer.create_animation(model, output_dir="animations/")
Figure ownership: plotting methods return an open matplotlib (or Plotly)
figure. Pass ``close=True`` to have the visualizer close the figure right
``plt.close(fig)`` so long plotting sessions do not accumulate figures.

"""

import os
import numpy as np
import textwrap
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field
from pathlib import Path
import warnings

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

logger = logging.getLogger(__name__)

# Default to a non-interactive backend for headless rendering, but never
# override a backend that the host application already configured.
if os.environ.get('MPLBACKEND') is None and matplotlib.get_backend().lower() != 'agg':
    matplotlib.use('Agg')


def _mean_ci_band(trajectories: np.ndarray,
                  n_std: float = 1.96,
                  kind: str = 'ci') -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute the mean trajectory with a cross-trajectory band.

    Single convention for every mean-band overlay in this module:
        kind='ci': mean +/- n_std * SEM -- a confidence interval on the
            mean (n_std=1.96 gives the 95% CI). Label: '95% CI'.
        kind='sd': mean +/- 1 standard deviation -- a spread band, not a
            confidence interval. Label: '±1 SD'.

    Returns (mean, band_lower, band_upper, label).
    """
    mean = np.mean(trajectories, axis=0)
    std = np.std(trajectories, axis=0)
    if kind == 'sd':
        return mean, mean - std, mean + std, '±1 SD'
    sem = std / np.sqrt(trajectories.shape[0])
    return mean, mean - n_std * sem, mean + n_std * sem, '95% CI'


def _apply_plot_style() -> None:
    """Apply the shared matplotlib rcParams used by every static panel.

    Single source of truth for the module's polished defaults: readable
    font sizes, a subtle dotted grid, despined axes, constrained layout,
    and a DPI floor of 120. Called at the top of each public plot and
    re-asserted in ``_save`` so figures created elsewhere also inherit
    the style.
    """
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 120,
        'figure.constrained_layout.use': True,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': ':',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.framealpha': 0.9,
    })


def _model_param_note(model: Any) -> str:
    """Return a compact description of a fitted model for panel corners.

    Combines the model type (an explicit ``model_type`` attribute when
    present, else the stochastic process name) with the key numeric
    fitted parameters, so comparison panels can be read without opening
    the parameter objects. Returns an empty string when the model
    exposes no usable identity or parameters.
    """
    model_type = getattr(model, 'model_type', None)
    if not model_type:
        model_type = getattr(getattr(model, 'stochastic_process', None),
                             'process_name', None)
    params = (getattr(model, 'fitted_parameters', None)
              or getattr(model, 'parameters', None))
    pieces = []
    if params is not None and hasattr(params, '__dict__'):
        for name, value in vars(params).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                pieces.append(f"{name}={value:.3g}")
    prefix = f"{model_type}: " if model_type else ""
    return prefix + ", ".join(pieces)


def _annotate_model_params(ax: Axes, models: List[Any]) -> None:
    """Draw fitted-parameter notes for each model in a small corner text.

    Long one-line notes are wrapped so the box stays inside the axes.
    """
    wrapped = []
    for note in (_model_param_note(m) for m in models):
        if note:
            wrapped.extend(textwrap.wrap(note, width=44) or [''])
    if not wrapped:
        return
    ax.text(0.98, 0.02, "\n".join(wrapped), transform=ax.transAxes,
            fontsize=7, ha='right', va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _aicc_winner_note(fits: Any) -> Optional[str]:
    """Return the AICc winner annotation for distribution-fit results.

    ``fits`` may be a mapping of candidate name -> fit mapping (each
    carrying an ``aicc`` entry) or a single fit mapping with
    ``distribution`` and ``aicc`` keys. Returns None when no finite
    AICc information is present, so panels render unchanged.
    """
    if not isinstance(fits, dict) or not fits:
        return None
    if isinstance(fits.get('aicc'), (int, float)) and np.isfinite(fits['aicc']):
        best = fits.get('distribution', 'fit')
        return f"Best fit: {best} (AICc={fits['aicc']:.2f})"
    scored = [
        (fit['aicc'], name)
        for name, fit in fits.items()
        if isinstance(fit, dict)
        and isinstance(fit.get('aicc'), (int, float))
        and np.isfinite(fit['aicc'])
    ]
    if not scored:
        return None
    best_aicc, best_name = min(scored)
    return f"Best fit: {best_name} (AICc={best_aicc:.2f})"


def _draw_fit_note(ax: Axes, source: Any) -> Optional[str]:
    """Annotate ``ax`` with the AICc-winning distribution when known.

    ``source`` is an optional-fit carrier (e.g. a model exposing
    ``distribution_fits`` or ``distribution_fit``). Returns the drawn
    note, or None when the source carries no AICc information.
    """
    fits = (getattr(source, 'distribution_fits', None)
            or getattr(source, 'distribution_fit', None))
    note = _aicc_winner_note(fits)
    if note:
        ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8,
                ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    return note


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior."""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 120  # matches the _apply_plot_style() DPI floor
    style: str = 'default'
    palette: str = 'viridis'
    alpha: float = 0.7
    linewidth: float = 2.0
    markersize: float = 6.0
    show_grid: bool = True
    show_legend: bool = True
    show_confidence_intervals: bool = True
    n_std: float = 1.96  # 95% confidence interval
    # GIF playback rate is derived at save time as 1000 / animation_interval.
    animation_interval: int = 50
    # Deterministic 3D landscape camera angles (degrees) and colour-axis units.
    landscape_elevation: float = 30.0
    landscape_azimuth: float = -60.0
    phenotype_units: str = 'units'
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])


@dataclass
class AnimationFrame:
    """Container for animation frame data.

    ``confidence_interval`` is the mean +/- n_std*SEM of the cross-section
    (a CI on the mean). ``metadata['distribution_quantiles']`` holds the
    2.5/97.5 percentiles of the cross-sectional distribution -- that is the
    band create_animation draws over the histogram.
    """
    time_point: float
    trajectories: np.ndarray
    cross_section: np.ndarray
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnimationController:
    """Controls animation sequences for developmental processes."""

    def __init__(self, jump_rope_model, config: PlotConfig):
        """Initialize animation controller."""
        self.model = jump_rope_model
        self.config = config
        self.frames: List[AnimationFrame] = []

    def generate_frames(self,
                       n_frames: Optional[int] = None,
                       time_range: Optional[Tuple[float, float]] = None) -> List[AnimationFrame]:
        """
        Generate animation frames.

        Parameters:
            n_frames: Number of frames to generate
            time_range: Time range for animation

        Returns:
            List of AnimationFrame objects
        """
        if time_range is None:
            time_range = (self.model.time_points[0], self.model.time_points[-1])

        if n_frames is None:
            n_frames = len(self.model.time_points)

        # Generate time points for animation
        if len(self.model.time_points) <= n_frames:
            frame_times = self.model.time_points
        else:
            frame_times = np.linspace(time_range[0], time_range[1], n_frames)

        self.frames = []

        for time_point in frame_times:
            try:
                # Get trajectories up to this time point
                time_idx = np.argmin(np.abs(self.model.time_points - time_point))
                trajectories = self.model.trajectories[:, :time_idx+1]

                # Get cross-section at this time point
                cross_section = self.model.compute_cross_sections(time_idx)

                # confidence_interval is the mean +/- n_std*SEM of the
                # cross-section (a CI on the mean). The distribution's own
                # 2.5/97.5 percentiles go into metadata; create_animation
                # draws those over the histogram because SEM lines hug the
                # mean and read as a bug against the full distribution.
                mean_val = np.mean(cross_section)
                std_val = np.std(cross_section)
                sem = std_val / np.sqrt(len(cross_section)) if len(cross_section) > 0 else 0.0
                ci = (
                    mean_val - self.config.n_std * sem,
                    mean_val + self.config.n_std * sem
                )
                if len(cross_section) > 0:
                    lo_q, hi_q = np.percentile(cross_section, [2.5, 97.5])
                    quantiles = (float(lo_q), float(hi_q))
                else:
                    quantiles = ci

                frame = AnimationFrame(
                    time_point=time_point,
                    trajectories=trajectories,
                    cross_section=cross_section,
                    confidence_interval=ci,
                    metadata={'distribution_quantiles': quantiles}
                )

                self.frames.append(frame)

            except Exception as e:
                logger.warning(f"Failed to generate frame for time {time_point}: {e}")
                continue

        logger.info(f"Generated {len(self.frames)} animation frames")
        return self.frames


class TrajectoryVisualizer:
    """Main visualization class for developmental trajectories."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize visualizer with configuration."""
        self.config = config or PlotConfig()
        self.animation_controller: Optional[AnimationController] = None

        # Set matplotlib style
        plt.style.use(self.config.style)

        logger.info("Initialized Trajectory Visualizer")

    def _save(self, fig, output_dir: Optional[Path], filename: str,
              close: bool = False) -> None:
        """Save a figure under output_dir and optionally close it.

        Interactive (Plotly) figures are written as HTML and never closed.
        Matplotlib figures are saved with the configured DPI and closed only
        when ``close`` is True, so long plotting sessions do not accumulate
        pyplot figures. A closed Figure object can still be re-saved by the
        caller (``fig.savefig`` keeps working).
        """
        if output_dir is None:
            return
        _apply_plot_style()  # figures created elsewhere still inherit the style
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        if hasattr(fig, 'write_html'):
            fig.write_html(out_path)
        else:
            fig.savefig(out_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {out_path}")
        if close and hasattr(fig, 'savefig'):
            plt.close(fig)

    def plot_trajectories(self,
                         jump_rope_model,
                         n_trajectories: Optional[int] = None,
                         output_dir: Optional[Path] = None,
                         interactive: bool = False,
                         show_ci: bool = True,
                         close: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot developmental trajectories.

        Parameters:
            jump_rope_model: JumpRope model with trajectories
            n_trajectories: Number of trajectories to plot
            output_dir: Directory to save plots
            interactive: Create interactive plot
            show_ci: Show confidence intervals
            close: If True, close the matplotlib figure right after saving
                it (static plots only); see module docstring.

        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating trajectory plot")

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        if n_trajectories is None:
            n_trajectories = min(100, trajectories.shape[0])
        elif n_trajectories > trajectories.shape[0]:
            n_trajectories = trajectories.shape[0]

        # Select subset of trajectories
        _apply_plot_style()
        selected_trajectories = trajectories[:n_trajectories]

        if interactive:
            return self._plot_trajectories_interactive(selected_trajectories, time_points)
        else:
            return self._plot_trajectories_static(selected_trajectories, time_points, output_dir, show_ci, close)

    def _plot_trajectories_static(self,
                                 trajectories: np.ndarray,
                                 time_points: np.ndarray,
                                 output_dir: Optional[Path] = None,
                                 show_ci: bool = True,
                                 close: bool = False) -> Figure:
        """Create static matplotlib plot of trajectories."""
        _apply_plot_style()
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Plot individual trajectories
        for i in range(trajectories.shape[0]):
            ax.plot(time_points, trajectories[i],
                   alpha=self.config.alpha,
                   linewidth=self.config.linewidth * 0.5,
                   color=self.config.colors[i % len(self.config.colors)])

        # Plot mean trajectory plus the 95% CI band on the mean (single
        # convention shared with the other band overlays in this module).
        mean_trajectory, ci_lower, ci_upper, ci_label = _mean_ci_band(
            trajectories, n_std=self.config.n_std)
        ax.plot(time_points, mean_trajectory,
               linewidth=self.config.linewidth * 2,
               color='black',
               label='Mean', linestyle='--')

        # Plot confidence intervals
        if show_ci and trajectories.shape[0] > 1:
            ax.fill_between(time_points, ci_lower, ci_upper,
                           alpha=0.3, color='gray', label=ci_label)

        # Formatting
        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Phenotype Value')
        ax.set_title('Developmental Trajectories')
        ax.grid(self.config.show_grid, alpha=0.3)
        ax.legend() if self.config.show_legend else None

        self._save(fig, output_dir, 'trajectories.png', close=close)

        return fig

    def _plot_trajectories_interactive(self,
                                     trajectories: np.ndarray,
                                     time_points: np.ndarray) -> go.Figure:
        """Create interactive Plotly plot of trajectories."""
        fig = go.Figure()

        # Add individual trajectories
        for i in range(min(50, trajectories.shape[0])):  # Limit for performance
            fig.add_trace(go.Scatter(
                x=time_points,
                y=trajectories[i],
                mode='lines',
                line=dict(width=1, color=self.config.colors[i % len(self.config.colors)]),
                opacity=self.config.alpha,
                showlegend=False
            ))

        # Add mean trajectory
        mean_trajectory = np.mean(trajectories, axis=0)
        fig.add_trace(go.Scatter(
            x=time_points,
            y=mean_trajectory,
            mode='lines',
            line=dict(width=3, color='black', dash='dash'),
            name='Mean Trajectory'
        ))

        # Add confidence intervals
        if trajectories.shape[0] > 1:
            _, ci_lower, ci_upper, _ = _mean_ci_band(
                trajectories, n_std=self.config.n_std)

            fig.add_trace(go.Scatter(
                x=np.concatenate([time_points, time_points[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title='Developmental Trajectories',
            xaxis_title='Developmental Time',
            yaxis_title='Phenotype Value',
            showlegend=True,
            hovermode='x unified'
        )

        return fig

    def plot_cross_sections(self,
                           jump_rope_model,
                           time_points: Optional[List[float]] = None,
                           output_dir: Optional[Path] = None,
                           interactive: bool = False,
                           show_kde: bool = False,
                           close: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot cross-sectional distributions at specific time points.

        Parameters:
            jump_rope_model: JumpRope model
            time_points: Time points to analyze
            output_dir: Directory to save plots
            interactive: Create interactive plot
            show_kde: Overlay a kernel-density estimate curve on each
                histogram (static plots only). Bandwidth uses Scott's rule
                via scipy.stats.gaussian_kde, which adapts to sample size
                and spread; the curve is labelled 'KDE (Scott's rule)' so a
                reader can distinguish it from the fitted-Normal overlay.
            close: If True, close the matplotlib figure right after saving
                it (static plots only); see module docstring.

        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating cross-section plot")
        _apply_plot_style()

        if time_points is None:
            time_points = jump_rope_model.time_points[::max(1, len(jump_rope_model.time_points)//5)]

        if interactive:
            return self._plot_cross_sections_interactive(jump_rope_model, time_points)
        else:
            return self._plot_cross_sections_static(jump_rope_model, time_points, output_dir, show_kde=show_kde, close=close)

    def _plot_cross_sections_static(self,
                                   jump_rope_model,
                                   time_points: List[float],
                                   output_dir: Optional[Path] = None,
                                   show_kde: bool = False,
                                   close: bool = False) -> Figure:
        """Create static matplotlib plot of cross-sections."""
        _apply_plot_style()
        n_plots = len(time_points)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figsize, dpi=self.config.dpi)
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, time_point in enumerate(time_points):
            row, col = i // n_cols, i % n_cols

            # Get cross-section data
            time_idx = np.argmin(np.abs(jump_rope_model.time_points - time_point))
            cross_section = jump_rope_model.compute_cross_sections(time_idx)

            # Plot histogram
            ax = axes[row, col]
            ax.hist(cross_section, bins=30, alpha=self.config.alpha,
                   density=True, color=self.config.colors[i % len(self.config.colors)])

            # Plot fitted distribution if available
            if jump_rope_model.fitted_parameters:
                from scipy.stats import norm
                x_vals = np.linspace(np.min(cross_section), np.max(cross_section), 100)
                y_vals = norm.pdf(x_vals, np.mean(cross_section), np.std(cross_section))
                ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Fitted Normal')

            # Optional kernel-density estimate over the histogram
            if show_kde and len(cross_section) >= 2 and np.std(cross_section) > 0:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(cross_section, bw_method='scott')
                x_kde = np.linspace(np.min(cross_section), np.max(cross_section), 200)
                ax.plot(x_kde, kde(x_kde), color='black', linestyle='--',
                        linewidth=1.5, label="KDE (Scott's rule)")

            ax.set_title(f'Time: {time_point:.2f}')
            ax.set_xlabel('Phenotype Value')
            ax.set_ylabel('Density')
            ax.grid(self.config.show_grid, alpha=0.3)
            # Every panel with labelled overlays gets its own legend so a
            # reader never has to infer which curve is which.
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='best', fontsize='small')
            _draw_fit_note(ax, jump_rope_model)


        self._save(fig, output_dir, 'cross_sections.png', close=close)

        return fig

    def _plot_cross_sections_interactive(self,
                                       jump_rope_model,
                                       time_points: List[float]) -> go.Figure:
        """Create interactive Plotly plot of cross-sections."""
        fig = make_subplots(
            rows=len(time_points), cols=1,
            subplot_titles=[f'Time: {t:.2f}' for t in time_points],
            shared_xaxes=True
        )

        for i, time_point in enumerate(time_points):
            # Get cross-section data
            time_idx = np.argmin(np.abs(jump_rope_model.time_points - time_point))
            cross_section = jump_rope_model.compute_cross_sections(time_idx)

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=cross_section,
                    nbinsx=30,
                    name=f'Time {time_point:.2f}',
                    showlegend=False,
                    opacity=self.config.alpha
                ),
                row=i+1, col=1
            )

            # Add fitted distribution if available
            if jump_rope_model.fitted_parameters:
                from scipy.stats import norm
                x_vals = np.linspace(np.min(cross_section), np.max(cross_section), 100)
                y_vals = norm.pdf(x_vals, np.mean(cross_section), np.std(cross_section))
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        name='Fitted Distribution',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )

        fig.update_layout(
            title='Cross-Sectional Distributions',
            height=300 * len(time_points),
            showlegend=False
        )

        return fig

    def plot_landscapes(self,
                       jump_rope_model,
                       output_dir: Optional[Path] = None,
                       interactive: bool = False,
                       close: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot 3D phenotypic landscapes showing distribution evolution.

        Parameters:
            jump_rope_model: JumpRope model
            output_dir: Directory to save plots
            interactive: Create interactive plot
            close: If True, close the matplotlib figure right after saving
                it (static plots only); see module docstring. The z-axis
                label carries ``config.phenotype_units`` when set to
                anything other than the default 'units'.

        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating landscape plot")
        _apply_plot_style()

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        if interactive:
            return self._plot_landscapes_interactive(jump_rope_model)
        else:
            return self._plot_landscapes_static(jump_rope_model, output_dir, close=close)

    def _plot_landscapes_static(self,
                               jump_rope_model,
                               output_dir: Optional[Path] = None,
                               close: bool = False) -> Figure:
        """Create static matplotlib 3D landscape plot."""
        _apply_plot_style()
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("3D plotting requires mpl_toolkits.mplot3d")

        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')

        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Plot trajectories in 3D (time, trajectory_id, phenotype)
        n_trajectories = min(50, trajectories.shape[0])  # Limit for performance

        for i in range(n_trajectories):
            ax.plot(time_points, [i] * len(time_points), trajectories[i],
                   alpha=self.config.alpha, linewidth=self.config.linewidth * 0.5)

        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Individual')
        z_label = 'Phenotype Value'
        if self.config.phenotype_units and self.config.phenotype_units != 'units':
            z_label = f'Phenotype Value ({self.config.phenotype_units})'
        ax.set_zlabel(z_label)
        ax.set_title('Phenotypic Landscape')


        self._save(fig, output_dir, 'landscape.png', close=close)

        return fig

    def _plot_landscapes_interactive(self, jump_rope_model) -> go.Figure:
        """Create interactive Plotly 3D landscape plot.

        Camera defaults mirror the static plot's deterministic
        elevation/azimuth so both formats show the same viewpoint.
        """
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Limit number of trajectories for performance
        n_trajectories = min(100, trajectories.shape[0])

        fig = go.Figure()

        for i in range(n_trajectories):
            fig.add_trace(go.Scatter3d(
                x=time_points,
                y=[i] * len(time_points),
                z=trajectories[i],
                mode='lines',
                line=dict(width=2, color=self.config.colors[i % len(self.config.colors)]),
                opacity=self.config.alpha,
                name=f'Trajectory {i}'
            ))

        z_label = 'Phenotype Value'
        if self.config.phenotype_units and self.config.phenotype_units != 'units':
            z_label = f'Phenotype Value ({self.config.phenotype_units})'
        fig.update_layout(
            title='Phenotypic Landscape',
            scene=dict(
                xaxis_title='Developmental Time',
                yaxis_title='Individual',
                zaxis_title=z_label,
                camera=dict(
                    eye=dict(
                        x=1.6 * np.cos(np.deg2rad(30)) * np.cos(np.deg2rad(-60)),
                        y=1.6 * np.cos(np.deg2rad(30)) * np.sin(np.deg2rad(-60)),
                        z=1.6 * np.sin(np.deg2rad(30)),
                    )
                ),
            ),
            showlegend=False
        )

        return fig

    def create_animation(self,
                        jump_rope_model,
                        n_frames: Optional[int] = None,
                        time_range: Optional[Tuple[float, float]] = None,
                        output_dir: Optional[Path] = None,
                        trailing_window: Optional[int] = None,
                        close: bool = False) -> animation.FuncAnimation:
        """
        Create animation of developmental process.

        Parameters:
            jump_rope_model: JumpRope model
            n_frames: Number of animation frames
            time_range: Time range for animation
            output_dir: Directory to save animation
            trailing_window: Optional k; when set, each frame shows only the
                last k time points of every trajectory instead of the full
                history. Long runs stay legible because old segments slide
                out of view. Axis limits stay fixed on the full time range
                (v0.2.0 behaviour), so the moving window is visible against
                the stationary frame.
            close: If True, close the animation figure after saving it;
                see module docstring. The cross-section panel draws the
                2.5/97.5 percentiles of the distribution (stored in each
                AnimationFrame's metadata) rather than the much narrower
                CI-on-the-mean lines.

        Returns:
            Matplotlib animation object
        """
        logger.info("Creating animation")
        _apply_plot_style()

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        # Create animation controller
        self.animation_controller = AnimationController(jump_rope_model, self.config)
        frames = self.animation_controller.generate_frames(n_frames, time_range)

        if not frames:
            raise ValueError("No frames generated for animation")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Fix axes across frames so the animation does not rescale per frame
        all_traj = jump_rope_model.trajectories
        ax1.set_xlim(jump_rope_model.time_points[0], jump_rope_model.time_points[-1])
        y_min, y_max = float(np.min(all_traj)), float(np.max(all_traj))
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ax1.set_ylim(y_min - pad, y_max + pad)
        pooled = all_traj.flatten()
        ax2.set_xlim(float(np.min(pooled)), float(np.max(pooled)))
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        def animate(frame_idx):
            frame = frames[frame_idx]

            # Clear axes
            ax1.clear()
            ax2.clear()

            # Plot trajectories up to current time; with trailing_window=k,
            # only the last k time points of each trajectory are drawn so
            # long runs stay legible (fixed axes show the sliding window).
            current_time_idx = np.argmin(np.abs(jump_rope_model.time_points - frame.time_point))
            current_trajectories = frame.trajectories

            start_idx = 0
            if trailing_window is not None and trailing_window > 0:
                start_idx = max(0, current_time_idx + 1 - int(trailing_window))

            for i in range(current_trajectories.shape[0]):
                ax1.plot(jump_rope_model.time_points[start_idx:current_time_idx+1],
                        current_trajectories[i, start_idx:current_time_idx+1],
                        alpha=self.config.alpha, linewidth=self.config.linewidth * 0.5,
                        label='Individual trajectories' if i == 0 else None)

            ax1.set_xlabel('Developmental Time')
            ax1.set_ylabel('Phenotype Value')
            window_label = (f', trailing window: {int(trailing_window)}'
                            if trailing_window else '')
            ax1.set_title(f'Developmental Trajectories '
                          f'(Time: {frame.time_point:.2f}{window_label})')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize='small')
            # Re-assert fixed limits each frame: ax1.clear() wipes them.
            ax1.set_xlim(jump_rope_model.time_points[0], jump_rope_model.time_points[-1])
            ax1.set_ylim(y_min - pad, y_max + pad)

            # Plot cross-section
            ax2.hist(frame.cross_section, bins=30, alpha=self.config.alpha, density=True)
            q_lo, q_hi = frame.metadata.get(
                'distribution_quantiles', frame.confidence_interval)
            ax2.axvline(q_lo, color='red', linestyle='--', alpha=0.7,
                        label='95% distribution quantiles')
            ax2.axvline(q_hi, color='red', linestyle='--', alpha=0.7)
            ax2.legend(loc='best')
            ax2.set_xlabel('Phenotype Value')
            ax2.set_ylabel('Density')
            ax2.set_title(f'Cross-Section Distribution (Mean: {np.mean(frame.cross_section):.3f})')
            ax2.grid(True, alpha=0.3)

            return ax1, ax2

        anim = animation.FuncAnimation(
            fig, animate,
            frames=len(frames),
            interval=self.config.animation_interval,
            blit=False,
            repeat=True
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Playback rate must match the frame interval (ms), otherwise
            # the saved GIF runs at the wrong speed.
            fps = int(round(1000.0 / self.config.animation_interval))
            anim.save(output_dir / 'animation.gif', writer='pillow', fps=fps)
            logger.info(f"Saved animation to {output_dir / 'animation.gif'} at {fps} fps")
            if close:
                plt.close(fig)

        return anim

    def plot_model_comparison(self,
                             models: List[Any],
                             model_names: List[str],
                             output_dir: Optional[Path] = None,
                             close: bool = False) -> Figure:
        """
        Create comprehensive multi-panel model comparison visualization.

        Parameters:
            models: List of JumpRope models
            model_names: Names for each model
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure with multiple panels
        """
        logger.info("Creating comprehensive model comparison visualization")
        _apply_plot_style()

        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of names")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Panel 1: Mean trajectories comparison
        ax1 = plt.subplot(3, 3, 1)
        colors = self.config.colors[:len(models)]

        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is None:
                continue

            color = colors[i % len(colors)]
            mean_traj = np.mean(model.trajectories, axis=0)
            ax1.plot(model.time_points, mean_traj,
                    label=name, color=color, linewidth=self.config.linewidth)

            # ±1 SD spread band (a spread band, not a confidence interval;
            # single convention shared with the other band overlays).
            _, lo_sd, hi_sd, _ = _mean_ci_band(model.trajectories, kind='sd')
            ax1.fill_between(model.time_points, lo_sd, hi_sd,
                           alpha=0.2, color=color)

        ax1.set_xlabel('Developmental Time')
        ax1.set_ylabel('Phenotype Value')
        ax1.set_title('Mean Trajectories\nwith ±1 SD Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Corner note: model type + key fitted parameters per model.
        _annotate_model_params(ax1, models)

        # Panel 2: Final distribution comparison
        ax2 = plt.subplot(3, 3, 2)
        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is None:
                continue

            color = colors[i % len(colors)]
            final_dist = model.compute_cross_sections(-1)
            ax2.hist(final_dist, bins=30, alpha=0.6,
                    label=name, color=color, density=True)

        ax2.set_xlabel('Phenotype Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Final Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # AICc winner when any model carries distribution-fit results.
        for _m in models:
            if _draw_fit_note(ax2, _m):
                break

        # Panel 3: Jump pattern comparison
        ax3 = plt.subplot(3, 3, 3)
        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is None:
                continue

            jump_times = model.estimate_jump_times()
            if len(jump_times) > 0:
                ax3.scatter(jump_times, [i] * len(jump_times),
                           label=name, color=colors[i % len(colors)],
                           s=self.config.markersize * 10, alpha=0.7)

        ax3.set_xlabel('Time')
        ax3.set_ylabel('Model')
        ax3.set_title('Jump Pattern Detection')
        ax3.set_yticks(range(len(model_names)))
        ax3.set_yticklabels(model_names)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Statistical properties comparison
        ax4 = plt.subplot(3, 3, 4)
        stats_data = []
        for model, name in zip(models, model_names):
            if model.trajectories is None:
                continue

            final_dist = model.compute_cross_sections(-1)
            stats = {
                'Model': name,
                'Mean': np.mean(final_dist),
                'Std': np.std(final_dist),
                'CV': np.std(final_dist) / np.mean(final_dist) if np.mean(final_dist) != 0 else 0,
                'Skewness': self._compute_skewness(final_dist),
                'Kurtosis': self._compute_kurtosis(final_dist)
            }
            stats_data.append(stats)

        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            metrics = ['Mean', 'Std', 'CV', 'Skewness', 'Kurtosis']
            x_pos = np.arange(len(metrics))

            for i, (_, row) in enumerate(df_stats.iterrows()):
                ax4.plot(x_pos, [row[metric] for metric in metrics],
                        marker='o', label=row['Model'],
                        color=colors[i % len(colors)])

            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(metrics, rotation=45)
            ax4.set_ylabel('Value')
            ax4.set_title('Statistical Properties')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Panel 5: Trajectory variability
        ax5 = plt.subplot(3, 3, 5)
        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is None:
                continue

            std_over_time = np.std(model.trajectories, axis=0)
            ax5.plot(model.time_points, std_over_time,
                    label=name, color=colors[i % len(colors)],
                    linewidth=self.config.linewidth)

        ax5.set_xlabel('Time')
        ax5.set_ylabel('Standard Deviation')
        ax5.set_title('Trajectory Variability')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Panel 6: Model parameters comparison
        ax6 = plt.subplot(3, 3, 6)
        param_comparison = []
        for model, name in zip(models, model_names):
            if hasattr(model, 'fitted_parameters') and model.fitted_parameters:
                params = model.fitted_parameters
                if hasattr(params, '__dict__'):
                    for param_name, param_value in params.__dict__.items():
                        if isinstance(param_value, (int, float)):
                            param_comparison.append({
                                'Model': name,
                                'Parameter': param_name,
                                'Value': param_value
                            })

        if param_comparison:
            df_params = pd.DataFrame(param_comparison)
            models_in_plot = df_params['Model'].unique()

            for i, model_name in enumerate(models_in_plot):
                model_params = df_params[df_params['Model'] == model_name]
                param_names = model_params['Parameter'].tolist()
                param_values = model_params['Value'].tolist()

                y_pos = [j + i*0.2 for j in range(len(param_names))]
                ax6.barh(y_pos, param_values, alpha=0.7,
                        label=model_name, color=colors[i % len(colors)],
                        height=0.15)

            ax6.set_yticks([j + 0.1 for j in range(len(param_names))])
            ax6.set_yticklabels(param_names)
            ax6.set_xlabel('Parameter Value')
            ax6.set_title('Model Parameters')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # Panel 7: Trajectory clustering
        ax7 = plt.subplot(3, 3, 7)
        all_trajectories = []
        all_labels = []

        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is not None:
                # Take a subset for clustering visualization
                subset = model.trajectories[:min(20, model.trajectories.shape[0])]
                all_trajectories.extend(subset)
                all_labels.extend([name] * len(subset))

        if len(all_trajectories) > 0:
            trajectories_array = np.array(all_trajectories)

            # Simple clustering based on final values
            final_values = trajectories_array[:, -1]
            clusters = np.argsort(final_values)

            for i, traj in enumerate(trajectories_array[clusters]):
                color_idx = i % len(colors)
                ax7.plot(model.time_points, traj,
                        alpha=0.6, color=colors[color_idx], linewidth=1)

        ax7.set_xlabel('Time')
        ax7.set_ylabel('Phenotype')
        ax7.set_title('Trajectory Clustering\nby Final Value')
        ax7.grid(True, alpha=0.3)

        # Panel 8: Model performance metrics
        ax8 = plt.subplot(3, 3, 8)
        performance_data = []

        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is not None:
                trajectories = model.trajectories
                final_dist = model.compute_cross_sections(-1)

                # Compute some basic performance metrics
                stability = 1 / (1 + np.std(final_dist))  # Higher is more stable
                predictability = 1 / (1 + np.mean(np.abs(np.diff(trajectories, axis=1))))  # Lower variation

                performance_data.append({
                    'Model': name,
                    'Stability': stability,
                    'Predictability': predictability,
                    'Complexity': len(model.estimate_jump_times())
                })

        if performance_data:
            df_perf = pd.DataFrame(performance_data)

            # Normalize for radar plot
            metrics = ['Stability', 'Predictability', 'Complexity']
            for metric in metrics:
                df_perf[metric] = (df_perf[metric] - df_perf[metric].min()) / (df_perf[metric].max() - df_perf[metric].min())

            # Simple bar chart instead of radar for now
            x_pos = np.arange(len(metrics))
            width = 0.8 / len(models)

            for i, (_, row) in enumerate(df_perf.iterrows()):
                ax8.bar(x_pos + i*width, [row[metric] for metric in metrics],
                       width=width, label=row['Model'],
                       color=colors[i % len(colors)], alpha=0.7)

            ax8.set_xticks(x_pos + width/2)
            ax8.set_xticklabels(metrics)
            ax8.set_ylabel('Normalized Score')
            ax8.set_title('Model Performance')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # Panel 9: Summary comparison
        ax9 = plt.subplot(3, 3, 9)
        comparison_text = "Model Comparison Summary:\n\n"

        for model, name in zip(models, model_names):
            if model.trajectories is not None:
                trajectories = model.trajectories
                final_dist = model.compute_cross_sections(-1)

                summary_stats = {
                    'Final Mean': f"{np.mean(final_dist):.2f}",
                    'Final SD': f"{np.std(final_dist):.2f}",
                    'Jumps': str(len(model.estimate_jump_times())),
                    'Trend': 'Increasing' if np.mean(np.diff(trajectories, axis=1)) > 0 else 'Decreasing'
                }

                comparison_text += f"{name}:\n"
                for stat, value in summary_stats.items():
                    comparison_text += f"  {stat}: {value}\n"
                comparison_text += "\n"

        ax9.text(0.05, 0.95, comparison_text,
                transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax9.set_title('Summary Comparison')
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')

        self._save(fig, output_dir, 'model_comparison.png', close=close)

        return fig

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            return 0.0

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0

    def plot_comparison(self,
                       models: List[Any],
                       model_names: List[str],
                       output_dir: Optional[Path] = None,
                       close: bool = False) -> Figure:
        """
        Plot comparison of multiple models.

        Parameters:
            models: List of JumpRope models
            model_names: Names for each model
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating model comparison plot")
        _apply_plot_style()

        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of names")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison', fontsize=16)

        colors = self.config.colors[:len(models)]

        for i, (model, name) in enumerate(zip(models, model_names)):
            if model.trajectories is None:
                continue

            color = colors[i % len(colors)]

            # Mean trajectories
            mean_traj = np.mean(model.trajectories, axis=0)
            axes[0, 0].plot(model.time_points, mean_traj,
                           label=name, color=color, linewidth=self.config.linewidth)

            # Final distributions
            final_dist = model.compute_cross_sections(-1)
            axes[0, 1].hist(final_dist, bins=30, alpha=self.config.alpha,
                           label=name, color=color, density=True)

            # Jump detection
            jump_times = model.estimate_jump_times()
            axes[1, 0].scatter(jump_times, [i] * len(jump_times),
                             label=name, color=color, s=self.config.markersize * 10)

            # Parameter summary (placeholder)
            axes[1, 1].bar([i], [len(jump_times)], label=name, color=color, alpha=self.config.alpha)

        # Formatting
        axes[0, 0].set_title('Mean Trajectories')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Phenotype')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        # Corner note: model type + key fitted parameters per model.
        _annotate_model_params(axes[0, 0], models)

        axes[0, 1].set_title('Final Distributions')
        axes[0, 1].set_xlabel('Phenotype Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        # AICc winner when any model carries distribution-fit results.
        for _m in models:
            if _draw_fit_note(axes[0, 1], _m):
                break

        axes[1, 0].set_title('Estimated Jump Times')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Model')
        axes[1, 0].set_yticks(range(len(model_names)))
        axes[1, 0].set_yticklabels(model_names)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title('Number of Jumps')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Jump Count')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)


        self._save(fig, output_dir, 'model_comparison.png', close=close)

        return fig

    def plot_bayesian_analysis(self,
                              bayesian_result,
                              output_dir: Optional[Path] = None,
                              close: bool = False) -> Figure:
        """
        Plot Bayesian analysis results.

        Parameters:
            bayesian_result: BayesianResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating Bayesian analysis plot")
        _apply_plot_style()

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle('Bayesian Analysis Results', fontsize=16)

        # Posterior samples distribution
        if len(bayesian_result.posterior_samples) > 0:
            axes[0, 0].hist(bayesian_result.posterior_samples, bins=50,
                           alpha=self.config.alpha, density=True)
            axes[0, 0].set_title('Posterior Distribution')
            axes[0, 0].set_xlabel('Parameter Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].grid(True, alpha=0.3)

        # Credible intervals
        if bayesian_result.credible_intervals:
            intervals = list(bayesian_result.credible_intervals.values())
            if intervals:
                ci_plot = axes[0, 1]
                # matplotlib >= 3.9 renamed boxplot's labels kwarg to
                # tick_labels (and removed it in 3.11); support both.
                interval_labels = list(bayesian_result.credible_intervals.keys())
                if tuple(int(p) for p in matplotlib.__version__.split('.')[:2]) >= (3, 9):
                    ci_plot.boxplot(intervals, tick_labels=interval_labels)
                else:
                    ci_plot.boxplot(intervals, labels=interval_labels)
                ci_plot.set_title('Credible Intervals')
                ci_plot.set_ylabel('Parameter Range')
                ci_plot.grid(True, alpha=0.3)

        # Convergence diagnostics
        if bayesian_result.convergence_diagnostics:
            diag_names = list(bayesian_result.convergence_diagnostics.keys())
            diag_values = list(bayesian_result.convergence_diagnostics.values())

            axes[1, 0].bar(range(len(diag_names)), diag_values)
            axes[1, 0].set_xticks(range(len(diag_names)))
            axes[1, 0].set_xticklabels(diag_names, rotation=45)
            axes[1, 0].set_title('Convergence Diagnostics')
            axes[1, 0].set_ylabel('Diagnostic Value')
            axes[1, 0].grid(True, alpha=0.3)

        # Model evidence
        axes[1, 1].text(0.5, 0.5, f'Model Evidence: {bayesian_result.model_evidence:.4f}',
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Model Evidence')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')


        self._save(fig, output_dir, 'bayesian_analysis.png', close=close)

        return fig

    def plot_network_analysis(self,
                            network_result,
                            output_dir: Optional[Path] = None,
                            close: bool = False) -> Figure:
        """
        Plot network analysis results.

        Parameters:
            network_result: NetworkResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating network analysis plot")
        _apply_plot_style()

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle('Network Analysis Results', fontsize=16)

        if network_result.graph is not None:
            G = network_result.graph

            # Network graph
            try:
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, ax=axes[0, 0], node_size=300, alpha=0.7,
                       node_color='lightblue', with_labels=True, font_size=8)
                axes[0, 0].set_title('Network Graph')
                axes[0, 0].axis('off')
            except (ValueError, MemoryError):
                axes[0, 0].text(0.5, 0.5, 'Network too complex to display',
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Network Graph')

            # Centrality measures
            if network_result.centrality_measures:
                centrality_names = list(network_result.centrality_measures.keys())
                centrality_values = list(network_result.centrality_measures.values())

                if isinstance(centrality_values[0], dict):
                    # Multiple nodes
                    nodes = list(centrality_values[0].keys())[:10]  # Show top 10
                    values = [centrality_values[0][node] for node in nodes]

                    axes[0, 1].bar(range(len(nodes)), values)
                    axes[0, 1].set_xticks(range(len(nodes)))
                    axes[0, 1].set_xticklabels(nodes, rotation=45)
                    axes[0, 1].set_title('Node Centrality (Top 10)')
                    axes[0, 1].set_ylabel('Centrality Value')
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].bar(range(len(centrality_names)), centrality_values)
                    axes[0, 1].set_xticks(range(len(centrality_names)))
                    axes[0, 1].set_xticklabels(centrality_names, rotation=45)
                    axes[0, 1].set_title('Centrality Measures')
                    axes[0, 1].set_ylabel('Centrality Value')
                    axes[0, 1].grid(True, alpha=0.3)

            # Network metrics
            if network_result.network_metrics:
                metrics_names = list(network_result.network_metrics.keys())
                metrics_values = list(network_result.network_metrics.values())

                axes[1, 0].bar(range(len(metrics_names)), metrics_values)
                axes[1, 0].set_xticks(range(len(metrics_names)))
                axes[1, 0].set_xticklabels(metrics_names, rotation=45)
                axes[1, 0].set_title('Network Metrics')
                axes[1, 0].set_ylabel('Metric Value')
                axes[1, 0].grid(True, alpha=0.3)

            # Community structure
            if network_result.community_structure and 'num_communities' in network_result.community_structure:
                num_communities = network_result.community_structure['num_communities']
                axes[1, 1].text(0.5, 0.5, f'Communities: {num_communities}\nModularity: {network_result.community_structure.get("modularity", "N/A"):.3f}',
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Community Structure')
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].axis('off')


        self._save(fig, output_dir, 'network_analysis.png', close=close)

        return fig

    def plot_dimensionality_reduction(self,
                                    dimensionality_result,
                                    output_dir: Optional[Path] = None,
                                    close: bool = False) -> Figure:
        """
        Plot dimensionality reduction results.

        Parameters:
            dimensionality_result: DimensionalityResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating dimensionality reduction plot")
        _apply_plot_style()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Dimensionality Reduction Analysis', fontsize=16)

        if dimensionality_result.embeddings.size > 0:
            embeddings = dimensionality_result.embeddings

            # Scatter plot of embeddings
            if embeddings.shape[1] >= 2:
                scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1],
                                        alpha=self.config.alpha, c=range(len(embeddings)),
                                        cmap='viridis', s=self.config.markersize * 10)
                axes[0].set_xlabel('Component 1')
                axes[0].set_ylabel('Component 2')
                axes[0].set_title('Dimensionality Reduction Embeddings')
                axes[0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0], label='Data Point Index')

            # Explained variance
            if dimensionality_result.explained_variance.size > 0:
                axes[1].bar(range(len(dimensionality_result.explained_variance)),
                           dimensionality_result.explained_variance)
                axes[1].set_xlabel('Component')
                axes[1].set_ylabel('Explained Variance')
                axes[1].set_title('Explained Variance by Component')
                axes[1].grid(True, alpha=0.3)

            # Reconstruction error
            axes[2].text(0.5, 0.5, f'Reconstruction Error: {dimensionality_result.reconstruction_error:.4f}\nIntrinsic Dimension: {dimensionality_result.intrinsic_dimensionality}',
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Model Quality Metrics')
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
            axes[2].axis('off')


        self._save(fig, output_dir, 'dimensionality_reduction.png', close=close)

        return fig

    def plot_spectral_analysis(self,
                             spectral_result,
                             output_dir: Optional[Path] = None,
                             close: bool = False) -> Figure:
        """
        Plot spectral analysis results.

        Parameters:
            spectral_result: SpectralResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating spectral analysis plot")
        _apply_plot_style()

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle('Spectral Analysis Results', fontsize=16)

        # Power spectrum
        if spectral_result.power_spectrum.size > 0:
            power_data = np.array(spectral_result.power_spectrum)
            if power_data.ndim > 1:
                axes[0, 0].plot(power_data[:, 0], power_data[:, 1])
                axes[0, 0].set_xlabel('Frequency')
                axes[0, 0].set_ylabel('Power')
                axes[0, 0].set_title('Power Spectrum')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_yscale('log')

        # Spectral peaks
        if spectral_result.frequency_peaks.size > 0:
            peak_data = np.array(spectral_result.frequency_peaks)
            if peak_data.ndim > 1:
                axes[0, 1].scatter(peak_data[:, 0], peak_data[:, 1], alpha=self.config.alpha)
                axes[0, 1].set_xlabel('Frequency')
                axes[0, 1].set_ylabel('Peak Power')
                axes[0, 1].set_title('Spectral Peaks')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_yscale('log')

        # Dominant frequencies
        if len(spectral_result.dominant_frequencies) > 0:
            axes[1, 0].bar(range(len(spectral_result.dominant_frequencies)),
                          spectral_result.dominant_frequencies)
            axes[1, 0].set_xlabel('Rank')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Dominant Frequencies')
            axes[1, 0].grid(True, alpha=0.3)

        # Spectral entropy
        axes[1, 1].text(0.5, 0.5, f'Spectral Entropy: {spectral_result.spectral_entropy:.4f}',
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Spectral Entropy')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')


        self._save(fig, output_dir, 'spectral_analysis.png', close=close)

        return fig

    def plot_nonlinear_dynamics(self,
                              nonlinear_result,
                              output_dir: Optional[Path] = None,
                              close: bool = False) -> Figure:
        """
        Plot nonlinear dynamics analysis results.

        Parameters:
            nonlinear_result: NonlinearResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating nonlinear dynamics plot")
        _apply_plot_style()

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle('Nonlinear Dynamics Analysis', fontsize=16)

        # Lyapunov exponents
        if nonlinear_result.lyapunov_exponents.size > 0:
            axes[0, 0].bar(range(len(nonlinear_result.lyapunov_exponents)),
                          nonlinear_result.lyapunov_exponents)
            axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Exponent Index')
            axes[0, 0].set_ylabel('Lyapunov Exponent')
            axes[0, 0].set_title('Lyapunov Spectrum')
            axes[0, 0].grid(True, alpha=0.3)

        # Correlation dimensions
        if nonlinear_result.correlation_dimensions.size > 0:
            axes[0, 1].bar(range(len(nonlinear_result.correlation_dimensions)),
                          nonlinear_result.correlation_dimensions)
            axes[0, 1].set_xlabel('Dimension Index')
            axes[0, 1].set_ylabel('Correlation Dimension')
            axes[0, 1].set_title('Correlation Dimensions')
            axes[0, 1].grid(True, alpha=0.3)

        # Chaos quantifiers
        if nonlinear_result.chaos_quantifiers:
            chaos_names = list(nonlinear_result.chaos_quantifiers.keys())
            chaos_values = list(nonlinear_result.chaos_quantifiers.values())

            axes[1, 0].bar(range(len(chaos_names)), chaos_values)
            axes[1, 0].set_xticks(range(len(chaos_names)))
            axes[1, 0].set_xticklabels(chaos_names, rotation=45)
            axes[1, 0].set_title('Chaos Quantifiers')
            axes[1, 0].set_ylabel('Quantifier Value')
            axes[1, 0].grid(True, alpha=0.3)

        # Attractor properties
        if nonlinear_result.attractor_properties:
            attractor_text = '\n'.join([f'{k}: {v}' for k, v in nonlinear_result.attractor_properties.items()])

            axes[1, 1].text(0.5, 0.5, attractor_text,
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].set_title('Attractor Properties')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')


        self._save(fig, output_dir, 'nonlinear_dynamics.png', close=close)

        return fig

    def plot_information_theory(self,
                              information_result,
                              output_dir: Optional[Path] = None,
                              close: bool = False) -> Figure:
        """
        Plot information theory analysis results.

        Parameters:
            information_result: InformationResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating information theory plot")
        _apply_plot_style()

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle('Information Theory Analysis', fontsize=16)

        # Entropy measures
        if information_result.entropy_measures:
            entropy_names = list(information_result.entropy_measures.keys())
            entropy_values = list(information_result.entropy_measures.values())

            axes[0, 0].bar(range(len(entropy_names)), entropy_values)
            axes[0, 0].set_xticks(range(len(entropy_names)))
            axes[0, 0].set_xticklabels(entropy_names, rotation=45)
            axes[0, 0].set_title('Entropy Measures')
            axes[0, 0].set_ylabel('Entropy Value')
            axes[0, 0].grid(True, alpha=0.3)

        # Mutual information
        if information_result.mutual_information.size > 0:
            mi_data = information_result.mutual_information
            if mi_data.ndim > 1:
                im = axes[0, 1].imshow(mi_data, cmap='viridis', aspect='auto')
                axes[0, 1].set_title('Mutual Information Matrix')
                axes[0, 1].grid(False)  # dotted grid over an image reads as noise
                axes[0, 1].set_xlabel('Variable Index')
                axes[0, 1].set_ylabel('Variable Index')
                plt.colorbar(im, ax=axes[0, 1])

        # Complexity measures
        if information_result.complexity_measures:
            complexity_names = list(information_result.complexity_measures.keys())
            complexity_values = list(information_result.complexity_measures.values())

            axes[1, 0].bar(range(len(complexity_names)), complexity_values)
            axes[1, 0].set_xticks(range(len(complexity_names)))
            axes[1, 0].set_xticklabels(complexity_names, rotation=45)
            axes[1, 0].set_title('Complexity Measures')
            axes[1, 0].set_ylabel('Complexity Value')
            axes[1, 0].grid(True, alpha=0.3)

        # Information flow
        if information_result.information_flow:
            flow_text = '\n'.join([f'{k}: {v}' for k, v in information_result.information_flow.items()])

            axes[1, 1].text(0.5, 0.5, flow_text,
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].set_title('Information Flow')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')


        self._save(fig, output_dir, 'information_theory.png', close=close)

        return fig

    def plot_robust_statistics(self,
                              robust_result,
                              output_dir: Optional[Path] = None,
                              close: bool = False) -> Figure:
        """
        Plot robust statistics analysis results.

        Parameters:
            robust_result: RobustResult from analytics engine
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure
        """
        logger.info("Creating robust statistics plot")
        _apply_plot_style()

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle('Robust Statistics Analysis', fontsize=16)

        # Location estimates comparison
        if robust_result.robust_estimates:
            location_names = list(robust_result.robust_estimates.keys())
            location_values = list(robust_result.robust_estimates.values())

            axes[0, 0].bar(range(len(location_names)), location_values)
            axes[0, 0].set_xticks(range(len(location_names)))
            axes[0, 0].set_xticklabels(location_names, rotation=45)
            axes[0, 0].set_title('Location Estimates Comparison')
            axes[0, 0].set_ylabel('Estimate Value')
            axes[0, 0].grid(True, alpha=0.3)

        # Outlier analysis
        if robust_result.outlier_analysis:
            outlier_names = list(robust_result.outlier_analysis.keys())
            outlier_values = list(robust_result.outlier_analysis.values())

            axes[0, 1].bar(range(len(outlier_names)), outlier_values)
            axes[0, 1].set_xticks(range(len(outlier_names)))
            axes[0, 1].set_xticklabels(outlier_names, rotation=45)
            axes[0, 1].set_title('Outlier Analysis')
            axes[0, 1].set_ylabel('Outlier Count')
            axes[0, 1].grid(True, alpha=0.3)

        # Influence measures
        if robust_result.influence_measures:
            influence_names = list(robust_result.influence_measures.keys())
            influence_values = list(robust_result.influence_measures.values())

            axes[1, 0].bar(range(len(influence_names)), influence_values)
            axes[1, 0].set_xticks(range(len(influence_names)))
            axes[1, 0].set_xticklabels(influence_names, rotation=45)
            axes[1, 0].set_title('Influence Measures')
            axes[1, 0].set_ylabel('Influence Value')
            axes[1, 0].grid(True, alpha=0.3)

        # Efficiency comparison
        if robust_result.efficiency_comparison:
            efficiency_names = list(robust_result.efficiency_comparison.keys())
            efficiency_values = list(robust_result.efficiency_comparison.values())

            axes[1, 1].bar(range(len(efficiency_names)), efficiency_values)
            axes[1, 1].set_xticks(range(len(efficiency_names)))
            axes[1, 1].set_xticklabels(efficiency_names, rotation=45)
            axes[1, 1].set_title('Efficiency Comparison')
            axes[1, 1].set_ylabel('Relative Efficiency')
            axes[1, 1].grid(True, alpha=0.3)


        self._save(fig, output_dir, 'robust_statistics.png', close=close)

        return fig
    
    def plot_comprehensive_trajectories(self,
                                      jump_rope_model,
                                      time_points: Optional[List[float]] = None,
                                      output_dir: Optional[Path] = None,
                                      close: bool = False) -> Figure:
        """
        Create comprehensive multi-panel trajectory visualization.

        Parameters:
            jump_rope_model: JumpRope model with trajectories
            time_points: Time points for cross-sectional analysis
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.

        Returns:
            Matplotlib figure with multiple panels
        """
        logger.info("Creating comprehensive trajectory visualization")
        _apply_plot_style()

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Panel 1: Individual trajectories
        ax1 = plt.subplot(3, 3, 1)
        trajectories = jump_rope_model.trajectories
        time_points_all = jump_rope_model.time_points

        n_trajectories = min(50, trajectories.shape[0])
        for i in range(n_trajectories):
            ax1.plot(time_points_all, trajectories[i],
                    alpha=0.3, linewidth=0.5,
                    color=self.config.colors[i % len(self.config.colors)])

        mean_trajectory, lo_sd, hi_sd, sd_label = _mean_ci_band(
            trajectories, kind='sd')
        ax1.plot(time_points_all, mean_trajectory, 'k-', linewidth=3, label='Mean')
        ax1.fill_between(time_points_all, lo_sd, hi_sd,
                        alpha=0.3, color='gray', label=sd_label)
        ax1.set_xlabel('Developmental Time')
        ax1.set_ylabel('Phenotype Value')
        ax1.set_title('Individual Trajectories\nwith Mean ± SD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Density heatmap
        ax2 = plt.subplot(3, 3, 2)
        self._plot_heatmap_panel(jump_rope_model, ax2)

        # Panel 3: Cross-sectional distributions
        ax3 = plt.subplot(3, 3, 3)
        self._plot_cross_section_panel(jump_rope_model, time_points, ax3)

        # Panel 4: Violin plots
        ax4 = plt.subplot(3, 3, 4)
        self._plot_violin_panel(jump_rope_model, ax4)

        # Panel 5: Ridge plot
        ax5 = plt.subplot(3, 3, 5)
        self._plot_ridge_panel(jump_rope_model, ax5)

        # Panel 6: Phase portrait
        ax6 = plt.subplot(3, 3, 6)
        self._plot_phase_portrait_panel(jump_rope_model, ax6)

        # Panel 7: Statistical summary
        ax7 = plt.subplot(3, 3, 7)
        self._plot_statistical_summary(jump_rope_model, ax7)

        # Panel 8: Model diagnostics
        ax8 = plt.subplot(3, 3, 8)
        self._plot_model_diagnostics(jump_rope_model, ax8)

        # Panel 9: Evolution summary
        ax9 = plt.subplot(3, 3, 9)
        self._plot_evolution_summary(jump_rope_model, ax9)


        self._save(fig, output_dir, 'comprehensive_trajectories.png', close=close)

        return fig

    def _plot_heatmap_panel(self, jump_rope_model, ax):
        """Plot heatmap in a subplot panel."""
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Create simple heatmap
        heatmap_data = np.zeros((50, len(time_points)))
        for i, t in enumerate(time_points):
            values = trajectories[:, i]
            hist, _ = np.histogram(values, bins=50, range=(trajectories.min(), trajectories.max()))
            heatmap_data[:, i] = hist

        im = ax.imshow(heatmap_data.T, aspect='auto', origin='lower',
                      extent=[trajectories.min(), trajectories.max(), time_points.min(), time_points.max()],
                      cmap='viridis', alpha=0.8)
        ax.set_xlabel('Phenotype')
        ax.set_ylabel('Time')
        ax.set_title('Density Heatmap')
        ax.grid(False)  # dotted grid over an image reads as noise
        plt.colorbar(im, ax=ax, shrink=0.8)

    def _plot_cross_section_panel(self, jump_rope_model, time_points, ax):
        """Plot cross-sectional distributions."""
        trajectories = jump_rope_model.trajectories
        all_time_points = jump_rope_model.time_points

        if time_points is None:
            time_points = [all_time_points[len(all_time_points)//4],
                          all_time_points[len(all_time_points)//2],
                          all_time_points[3*len(all_time_points)//4]]

        colors = ['red', 'green', 'blue']
        for i, t in enumerate(time_points):
            time_idx = np.argmin(np.abs(all_time_points - t))
            values = trajectories[:, time_idx]
            ax.hist(values, bins=20, alpha=0.6, color=colors[i],
                   label=f't = {t:.1f}', density=True)

        ax.set_xlabel('Phenotype Value')
        ax.set_ylabel('Density')
        ax.set_title('Cross-Sectional\nDistributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _draw_fit_note(ax, jump_rope_model)

    def _plot_violin_panel(self, jump_rope_model, ax):
        """Plot violin plot panel."""
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Select key time points
        n_points = min(5, len(time_points))
        indices = np.linspace(0, len(time_points)-1, n_points, dtype=int)
        selected_times = time_points[indices]

        data = [trajectories[:, idx] for idx in indices]
        positions = range(len(data))

        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        # Label the summary markers so the panel explains itself.
        parts['cmeans'].set_label('Mean')
        parts['cmedians'].set_label('Median')

        ax.set_xticks(positions)
        ax.set_xticklabels([f'{t:.1f}' for t in selected_times])
        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Phenotype Value')
        ax.set_title('Violin Plots')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')

    def _plot_ridge_panel(self, jump_rope_model, ax):
        """Plot ridge plot panel."""
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Select evenly spaced time points
        n_distributions = min(8, len(time_points))
        indices = np.linspace(0, len(time_points)-1, n_distributions, dtype=int)

        global_min, global_max = trajectories.min(), trajectories.max()
        x_range = np.linspace(global_min, global_max, 100)

        for i, idx in enumerate(indices):
            time = time_points[idx]
            values = trajectories[:, idx]

            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(values)
                density = kde(x_range)

                # Offset each distribution; each layer is labelled with its
                # time point so the ridge panel carries its own legend.
                y_offset = i * 0.1
                ax.fill_between(x_range, y_offset, y_offset + density * 0.1,
                              alpha=0.6, color=self.config.colors[i % len(self.config.colors)],
                              label=f't = {time:.1f}')
                ax.plot(x_range, y_offset + density * 0.1,
                       color=self.config.colors[i % len(self.config.colors)], linewidth=1)
            except:
                pass

        ax.set_xlabel('Phenotype Value')
        ax.set_ylabel('Time')
        ax.set_title('Ridge Plot')
        ax.set_ylim(-0.1, (n_distributions-1) * 0.1 + 0.1)
        ax.set_xlim(global_min, global_max)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize='small', ncol=2, loc='upper left')

    def _plot_phase_portrait_panel(self, jump_rope_model, ax):
        """Plot phase portrait panel."""
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Compute derivatives using finite differences
        dt = np.diff(time_points)
        derivatives = np.diff(trajectories, axis=1) / dt
        phenotype_values = (trajectories[:, :-1] + trajectories[:, 1:]) / 2

        # Sample for visualization
        n_samples = min(1000, phenotype_values.size)
        # Seeded generator: the comprehensive figure stays reproducible
        # run-to-run, consistent with the fixed landscape camera angles.
        indices = np.random.default_rng(0).choice(
            phenotype_values.size, n_samples, replace=False)

        scatter = ax.scatter(phenotype_values.flatten()[indices],
                           derivatives.flatten()[indices],
                           c=time_points[:-1][indices // trajectories.shape[0]],
                           cmap='viridis', alpha=0.6, s=10)

        ax.set_xlabel('Phenotype Value')
        ax.set_ylabel('Rate of Change (dP/dt)')
        ax.set_title('Phase Portrait')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='Time')

    def _plot_statistical_summary(self, jump_rope_model, ax):
        """Plot statistical summary."""
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        # Compute statistics over time
        means = np.mean(trajectories, axis=0)
        stds = np.std(trajectories, axis=0)
        cv = stds / means  # Coefficient of variation

        ax2 = ax.twinx()

        line1 = ax.plot(time_points, means, 'b-', linewidth=2, label='Mean')
        ax.fill_between(time_points, means - stds, means + stds,
                       alpha=0.3, color='blue', label='±1 SD')

        line2 = ax2.plot(time_points, cv, 'r-', linewidth=2, label='CV')

        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Phenotype Value', color='blue')
        ax2.set_ylabel('Coefficient of Variation', color='red')
        ax.set_title('Statistical Summary')
        ax.grid(True, alpha=0.3)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels)

    def _plot_model_diagnostics(self, jump_rope_model, ax):
        """Plot model diagnostics."""
        if not hasattr(jump_rope_model, 'fitted_parameters'):
            ax.text(0.5, 0.5, 'No model\nparameters\navailable',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Diagnostics')
            return

        params = jump_rope_model.fitted_parameters

        # Plot parameter distributions or values
        if hasattr(params, '__dict__'):
            param_names = []
            param_values = []

            for name, value in params.__dict__.items():
                if value is not None and isinstance(value, (int, float)):
                    param_names.append(name)
                    param_values.append(float(value))

            if len(param_names) > 0:
                y_pos = np.arange(len(param_names))
                ax.barh(y_pos, param_values, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(param_names)
                ax.set_xlabel('Parameter Value')
                ax.set_title('Fitted Parameters')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid\nparameters\navailable',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Model Diagnostics')

    def _plot_evolution_summary(self, jump_rope_model, ax):
        """Plot evolution summary."""
        trajectories = jump_rope_model.trajectories

        # Compute evolutionary metrics
        final_values = trajectories[:, -1]
        initial_values = trajectories[:, 0]

        # Plot initial vs final distribution
        ax.scatter(initial_values, final_values, alpha=0.6, s=20)

        # Add diagonal line
        min_val = min(initial_values.min(), final_values.min())
        max_val = max(initial_values.max(), final_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='No Evolution')

        ax.set_xlabel('Initial Phenotype')
        ax.set_ylabel('Final Phenotype')
        ax.set_title('Evolutionary Change')
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _trajectory_sort_order(trajectories: np.ndarray,
                               statistic: str) -> np.ndarray:
        """Return the permutation ordering trajectories by ``statistic``.

        Supported statistics: 'final_value', 'mean_value', 'max_value',
        'min_value' (NaNs are ignored for the mean/max/min statistics).
        """
        if statistic == 'final_value':
            stat = trajectories[:, -1]
        elif statistic == 'mean_value':
            stat = np.nanmean(trajectories, axis=1)
        elif statistic == 'max_value':
            stat = np.nanmax(trajectories, axis=1)
        elif statistic == 'min_value':
            stat = np.nanmin(trajectories, axis=1)
        else:
            raise ValueError(f"Unknown row_sort_statistic: {statistic!r}")
        return np.argsort(stat)

    def plot_heatmap(self,
                    jump_rope_model,
                    time_resolution: int = 50,
                    phenotype_resolution: int = 50,
                    output_dir: Optional[Path] = None,
                    interactive: bool = False,
                    sort_rows: bool = True,
                    row_sort_statistic: str = 'final_value',
                    x_label: Optional[str] = None,
                    y_label: Optional[str] = None,
                    close: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot density heatmap of trajectory evolution.

        Parameters:
            jump_rope_model: JumpRope model with trajectories
            time_resolution: Number of time bins
            phenotype_resolution: Number of phenotype bins
            output_dir: Directory to save plots
            interactive: Create interactive plot
            sort_rows: Sort heatmap rows by ``row_sort_statistic`` (default:
                'final_value' — the trajectory value at the last time point)
                so monotone structure such as graded outcomes reads as
                ordered bands instead of visual noise. Set False to keep the
                original trajectory order. Supported statistics:
                'final_value', 'mean_value', 'max_value', 'min_value'.
            row_sort_statistic: Statistic used when ``sort_rows`` is True.
            x_label: X-axis label; defaults to the model's time-column name
                when discoverable, else 'Developmental Time'.
            y_label: Y-axis label; defaults to the model's phenotype-column
                name when discoverable, else 'Phenotype Value'.
            close: If True, close the matplotlib figure right after saving
                it (static plots only); see module docstring.

        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating trajectory density heatmap")
        _apply_plot_style()

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        supported_stats = ('final_value', 'mean_value', 'max_value', 'min_value')
        if row_sort_statistic not in supported_stats:
            raise ValueError(
                f"row_sort_statistic must be one of {supported_stats}, "
                f"got {row_sort_statistic!r}")

        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points

        if sort_rows:
            # Reorder trajectories by the documented statistic so readers can
            # track cohort structure across the heatmap.
            order = self._trajectory_sort_order(trajectories, row_sort_statistic)
            trajectories = trajectories[order]

        time_label = x_label
        pheno_label = y_label
        if time_label is None:
            ts_list = (getattr(jump_rope_model, '_source_time_series', None)
                       or getattr(jump_rope_model, 'time_series_data', None)
                       or [])
            for ts in ts_list:
                time_label = getattr(ts, 'time_column', None)
                if time_label:
                    break
            time_label = time_label or 'Developmental Time'
        if pheno_label is None:
            pheno_label = 'Phenotype Value'

        # Drop non-finite values instead of imputing them (e.g. to 0):
        # imputation fabricates data mass at phenotype 0, drags the phenotype
        # extent toward 0, and pollutes every histogram column.
        finite_values = trajectories[np.isfinite(trajectories)]
        if finite_values.size == 0:
            raise ValueError("No finite phenotype values available for heatmap.")

        # Create time and phenotype grids
        time_edges = np.linspace(time_points.min(), time_points.max(), time_resolution + 1)
        phenotype_min = float(finite_values.min())
        phenotype_max = float(finite_values.max())

        # Handle case where all values are the same
        if np.isclose(phenotype_min, phenotype_max):
            phenotype_min -= 1.0
            phenotype_max += 1.0

        phenotype_edges = np.linspace(phenotype_min, phenotype_max, phenotype_resolution + 1)

        # Compute 2D histogram for each time bin
        density_map = np.zeros((phenotype_resolution, time_resolution))

        for t_idx in range(time_resolution):
            # Find closest time point
            t_center = (time_edges[t_idx] + time_edges[t_idx + 1]) / 2
            closest_time_idx = np.argmin(np.abs(time_points - t_center))

            # Keep only finite trajectory values at this time
            values_at_time = trajectories[:, closest_time_idx]
            values_at_time = values_at_time[np.isfinite(values_at_time)]
            if values_at_time.size == 0:
                continue

            # Compute histogram
            hist, _ = np.histogram(values_at_time, bins=phenotype_edges)
            density_map[:, t_idx] = hist

        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=density_map,
                x=0.5 * (time_edges[:-1] + time_edges[1:]),
                y=0.5 * (phenotype_edges[:-1] + phenotype_edges[1:]),
                colorscale='Viridis',
                colorbar=dict(title='Trajectory Density')
            ))

            fig.update_layout(
                title='Trajectory Density Heatmap',
                xaxis_title=time_label,
                yaxis_title=pheno_label,
                width=800,
                height=600
            )

            self._save(fig, output_dir, 'density_heatmap.html')

            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 8))

            im = ax.imshow(density_map, aspect='auto', origin='lower',
                          extent=[time_points.min(), time_points.max(), phenotype_min, phenotype_max],
                          cmap='viridis', interpolation='bilinear')

            ax.set_xlabel(time_label)
            ax.set_ylabel(pheno_label)
            title = 'Trajectory Density Heatmap'
            if sort_rows:
                title += f' (rows sorted by {row_sort_statistic.replace("_", " ")})'
            ax.set_title(title)
            ax.grid(False)  # dotted grid over an image reads as noise

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Trajectory Density', rotation=270, labelpad=20)

            self._save(fig, output_dir, 'density_heatmap.png', close=close)

            return fig
    
    def plot_violin(self,
                   jump_rope_model,
                   time_points: Optional[List[float]] = None,
                   output_dir: Optional[Path] = None,
                   close: bool = False) -> Figure:
        """
        Plot violin plots showing distribution at multiple time points.
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            time_points: Specific time points to plot (if None, use evenly spaced)
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.
        
        Returns:
            Matplotlib figure
        """
        logger.info("Creating violin plots")
        _apply_plot_style()
        
        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")
        
        trajectories = jump_rope_model.trajectories
        all_time_points = jump_rope_model.time_points
        
        # Select time points
        if time_points is None:
            n_points = min(8, len(all_time_points))
            indices = np.linspace(0, len(all_time_points) - 1, n_points, dtype=int)
            time_points = all_time_points[indices]
        else:
            # Find closest time points
            indices = [np.argmin(np.abs(all_time_points - t)) for t in time_points]
            time_points = all_time_points[indices]
        
        # Collect data for violin plots
        data_for_violin = []
        labels = []
        for idx in indices:
            data_for_violin.append(trajectories[:, idx])
            labels.append(f't={time_points[len(labels)]:.2f}')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                             showmeans=True, showmedians=True)
        
        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            color = self.config.colors[i % len(self.config.colors)]
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        # Label the summary markers so the panel explains itself.
        parts['cmeans'].set_label('Mean')
        parts['cmedians'].set_label('Median')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Phenotype Value')
        ax.set_title('Phenotype Distribution Evolution (Violin Plots)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
        
        
        self._save(fig, output_dir, 'violin_plots.png', close=close)
        
        return fig
    
    def plot_ridge(self,
                  jump_rope_model,
                  n_distributions: int = 10,
                  output_dir: Optional[Path] = None,
                  close: bool = False) -> Figure:
        """
        Plot ridge plot (joyplot) showing distribution evolution over time.
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            n_distributions: Number of distributions to show
            output_dir: Directory to save plots
            close: If True, close the figure after saving it; see module
                docstring.
        
        Returns:
            Matplotlib figure
        """
        logger.info("Creating ridge plot")
        _apply_plot_style()
        
        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")
        
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points
        
        # Select evenly spaced time points
        n_distributions = min(n_distributions, len(time_points))
        indices = np.linspace(0, len(time_points) - 1, n_distributions, dtype=int)
        
        fig, axes = plt.subplots(n_distributions, 1, figsize=(12, 2 * n_distributions),
                                sharex=True)
        
        if n_distributions == 1:
            axes = [axes]
        
        # Find global min/max for consistent x-axis
        global_min = trajectories.min()
        global_max = trajectories.max()
        x_range = np.linspace(global_min, global_max, 200)
        
        for i, (ax, idx) in enumerate(zip(axes, indices)):
            time = time_points[idx]
            values = trajectories[:, idx]
            
            # Compute KDE
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(values)
                density = kde(x_range)
                
                # Fill the area under the curve
                color = self.config.colors[i % len(self.config.colors)]
                ax.fill_between(x_range, 0, density, alpha=0.7, color=color)
                ax.plot(x_range, density, color=color, linewidth=2)
                
                # Add time label
                ax.text(0.02, 0.75, f't = {time:.2f}', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Formatting
                ax.set_xlim(global_min, global_max)
                ax.set_ylim(0, None)
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if i < len(axes) - 1:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticks([])
            except Exception as e:
                logger.warning(f"Could not create KDE for time {time}: {e}")
                # Use a fallback color if not defined
                fallback_color = self.config.colors[i % len(self.config.colors)] if hasattr(self, 'config') and hasattr(self.config, 'colors') else 'blue'
                ax.hist(values, bins=30, alpha=0.7, color=fallback_color, density=True)
        
        # Only show x-axis label on bottom plot
        axes[-1].set_xlabel('Phenotype Value')
        axes[-1].spines['bottom'].set_visible(True)
        
        fig.suptitle('Phenotype Distribution Evolution (Ridge Plot)', fontsize=16, y=0.995)
        
        self._save(fig, output_dir, 'ridge_plot.png', close=close)
        
        return fig
    
    def plot_phase_portrait(self,
                           jump_rope_model,
                           derivative_method: str = 'finite_difference',
                           output_dir: Optional[Path] = None,
                           interactive: bool = False,
                           close: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot phase portrait (phenotype vs. rate of change).
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            derivative_method: Method to compute derivatives ('finite_difference', 'spline')
            output_dir: Directory to save plots
            interactive: Create interactive plot
            close: If True, close the figure after saving it; see module
                docstring.
        
        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating phase portrait")
        _apply_plot_style()
        
        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")
        
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points
        
        # Compute derivatives
        if derivative_method == 'finite_difference':
            dt = np.diff(time_points)
            derivatives = np.diff(trajectories, axis=1) / dt
            # Use midpoint values for phenotype
            phenotype_values = (trajectories[:, :-1] + trajectories[:, 1:]) / 2
        elif derivative_method == 'spline':
            from scipy.interpolate import UnivariateSpline
            derivatives = np.zeros_like(trajectories)
            phenotype_values = trajectories
            
            for i in range(trajectories.shape[0]):
                try:
                    spline = UnivariateSpline(time_points, trajectories[i, :], s=0.1)
                    derivatives[i, :] = spline.derivative()(time_points)
                except:
                    # Fallback to finite differences
                    dt = np.diff(time_points)
                    derivatives[i, :-1] = np.diff(trajectories[i, :]) / dt
                    derivatives[i, -1] = derivatives[i, -2]
        else:
            raise ValueError(f"Unknown derivative method: {derivative_method}")
        
        if interactive:
            # Create interactive scatter plot with color gradient for time
            time_colors = np.repeat(time_points[:phenotype_values.shape[1]], phenotype_values.shape[0])
            
            fig = go.Figure(data=go.Scattergl(
                x=phenotype_values.flatten(),
                y=derivatives.flatten(),
                mode='markers',
                marker=dict(
                    size=3,
                    color=time_colors,
                    colorscale='Viridis',
                    colorbar=dict(title='Time'),
                    opacity=0.5
                )
            ))
            
            fig.update_layout(
                title='Phase Portrait: Phenotype vs. Rate of Change',
                xaxis_title='Phenotype Value',
                yaxis_title='Rate of Change (dP/dt)',
                width=800,
                height=600
            )
            
            self._save(fig, output_dir, 'phase_portrait.html')
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot with color gradient
            for i in range(0, phenotype_values.shape[0], max(1, phenotype_values.shape[0] // 100)):
                scatter = ax.scatter(phenotype_values[i, :], derivatives[i, :],
                                   c=time_points[:phenotype_values.shape[1]],
                                   cmap='viridis', alpha=0.5, s=20)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Developmental Time', rotation=270, labelpad=20)
            
            ax.set_xlabel('Phenotype Value')
            ax.set_ylabel('Rate of Change (dP/dt)')
            ax.set_title('Phase Portrait: Phenotype vs. Rate of Change')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            self._save(fig, output_dir, 'phase_portrait.png', close=close)
            
            return fig