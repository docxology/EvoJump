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
"""

import numpy as np
import pandas as pd
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

# Set matplotlib backend for non-interactive plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
except:
    pass


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior."""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    style: str = 'default'
    palette: str = 'viridis'
    alpha: float = 0.7
    linewidth: float = 2.0
    markersize: float = 6.0
    show_grid: bool = True
    show_legend: bool = True
    show_confidence_intervals: bool = True
    n_std: float = 1.96  # 95% confidence interval
    animation_fps: int = 30
    animation_interval: int = 50
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])


@dataclass
class AnimationFrame:
    """Container for animation frame data."""
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

                # Compute confidence interval
                mean_val = np.mean(cross_section)
                std_val = np.std(cross_section)
                ci = (
                    mean_val - self.config.n_std * std_val / np.sqrt(len(cross_section)),
                    mean_val + self.config.n_std * std_val / np.sqrt(len(cross_section))
                )

                frame = AnimationFrame(
                    time_point=time_point,
                    trajectories=trajectories,
                    cross_section=cross_section,
                    confidence_interval=ci
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

    def plot_trajectories(self,
                         jump_rope_model,
                         n_trajectories: Optional[int] = None,
                         output_dir: Optional[Path] = None,
                         interactive: bool = False,
                         show_ci: bool = True) -> Union[Figure, go.Figure]:
        """
        Plot developmental trajectories.

        Parameters:
            jump_rope_model: JumpRope model with trajectories
            n_trajectories: Number of trajectories to plot
            output_dir: Directory to save plots
            interactive: Create interactive plot
            show_ci: Show confidence intervals

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
        selected_trajectories = trajectories[:n_trajectories]

        if interactive:
            return self._plot_trajectories_interactive(selected_trajectories, time_points)
        else:
            return self._plot_trajectories_static(selected_trajectories, time_points, output_dir, show_ci)

    def _plot_trajectories_static(self,
                                 trajectories: np.ndarray,
                                 time_points: np.ndarray,
                                 output_dir: Optional[Path] = None,
                                 show_ci: bool = True) -> Figure:
        """Create static matplotlib plot of trajectories."""
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Plot individual trajectories
        for i in range(trajectories.shape[0]):
            ax.plot(time_points, trajectories[i],
                   alpha=self.config.alpha,
                   linewidth=self.config.linewidth * 0.5,
                   color=self.config.colors[i % len(self.config.colors)])

        # Plot mean trajectory
        mean_trajectory = np.mean(trajectories, axis=0)
        ax.plot(time_points, mean_trajectory,
               linewidth=self.config.linewidth * 2,
               color='black',
               label='Mean', linestyle='--')

        # Plot confidence intervals
        if show_ci and trajectories.shape[0] > 1:
            std_trajectory = np.std(trajectories, axis=0)
            ci_lower = mean_trajectory - self.config.n_std * std_trajectory / np.sqrt(trajectories.shape[0])
            ci_upper = mean_trajectory + self.config.n_std * std_trajectory / np.sqrt(trajectories.shape[0])

            ax.fill_between(time_points, ci_lower, ci_upper,
                           alpha=0.3, color='gray', label='95% CI')

        # Formatting
        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Phenotype Value')
        ax.set_title('Developmental Trajectories')
        ax.grid(self.config.show_grid, alpha=0.3)
        ax.legend() if self.config.show_legend else None

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'trajectories.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved trajectory plot to {output_dir / 'trajectories.png'}")

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
            std_trajectory = np.std(trajectories, axis=0)
            ci_lower = mean_trajectory - self.config.n_std * std_trajectory / np.sqrt(trajectories.shape[0])
            ci_upper = mean_trajectory + self.config.n_std * std_trajectory / np.sqrt(trajectories.shape[0])

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
                           interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot cross-sectional distributions at specific time points.

        Parameters:
            jump_rope_model: JumpRope model
            time_points: Time points to analyze
            output_dir: Directory to save plots
            interactive: Create interactive plot

        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating cross-section plot")

        if time_points is None:
            time_points = jump_rope_model.time_points[::max(1, len(jump_rope_model.time_points)//5)]

        if interactive:
            return self._plot_cross_sections_interactive(jump_rope_model, time_points)
        else:
            return self._plot_cross_sections_static(jump_rope_model, time_points, output_dir)

    def _plot_cross_sections_static(self,
                                   jump_rope_model,
                                   time_points: List[float],
                                   output_dir: Optional[Path] = None) -> Figure:
        """Create static matplotlib plot of cross-sections."""
        n_plots = len(time_points)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figsize, dpi=self.config.dpi)
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

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

            ax.set_title(f'Time: {time_point:.2f}')
            ax.set_xlabel('Phenotype Value')
            ax.set_ylabel('Density')
            ax.grid(self.config.show_grid, alpha=0.3)
            ax.legend() if i == 0 else None

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'cross_sections.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved cross-section plot to {output_dir / 'cross_sections.png'}")

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
                       interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot 3D phenotypic landscapes showing distribution evolution.

        Parameters:
            jump_rope_model: JumpRope model
            output_dir: Directory to save plots
            interactive: Create interactive plot

        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating landscape plot")

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        if interactive:
            return self._plot_landscapes_interactive(jump_rope_model)
        else:
            return self._plot_landscapes_static(jump_rope_model, output_dir)

    def _plot_landscapes_static(self,
                               jump_rope_model,
                               output_dir: Optional[Path] = None) -> Figure:
        """Create static matplotlib 3D landscape plot."""
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
        ax.set_zlabel('Phenotype Value')
        ax.set_title('Phenotypic Landscape')

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'landscape.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved landscape plot to {output_dir / 'landscape.png'}")

        return fig

    def _plot_landscapes_interactive(self, jump_rope_model) -> go.Figure:
        """Create interactive Plotly 3D landscape plot."""
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

        fig.update_layout(
            title='Phenotypic Landscape',
            scene=dict(
                xaxis_title='Developmental Time',
                yaxis_title='Individual',
                zaxis_title='Phenotype Value'
            ),
            showlegend=False
        )

        return fig

    def create_animation(self,
                        jump_rope_model,
                        n_frames: Optional[int] = None,
                        time_range: Optional[Tuple[float, float]] = None,
                        output_dir: Optional[Path] = None) -> animation.FuncAnimation:
        """
        Create animation of developmental process.

        Parameters:
            jump_rope_model: JumpRope model
            n_frames: Number of animation frames
            time_range: Time range for animation
            output_dir: Directory to save animation

        Returns:
            Matplotlib animation object
        """
        logger.info("Creating animation")

        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")

        # Create animation controller
        self.animation_controller = AnimationController(jump_rope_model, self.config)
        frames = self.animation_controller.generate_frames(n_frames, time_range)

        if not frames:
            raise ValueError("No frames generated for animation")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def animate(frame_idx):
            frame = frames[frame_idx]

            # Clear axes
            ax1.clear()
            ax2.clear()

            # Plot trajectories up to current time
            current_time_idx = np.argmin(np.abs(jump_rope_model.time_points - frame.time_point))
            current_trajectories = frame.trajectories

            for i in range(current_trajectories.shape[0]):
                ax1.plot(jump_rope_model.time_points[:current_time_idx+1],
                        current_trajectories[i, :current_time_idx+1],
                        alpha=self.config.alpha, linewidth=self.config.linewidth * 0.5)

            ax1.set_xlabel('Developmental Time')
            ax1.set_ylabel('Phenotype Value')
            ax1.set_title(f'Developmental Trajectories (Time: {frame.time_point:.2f})')
            ax1.grid(True, alpha=0.3)

            # Plot cross-section
            ax2.hist(frame.cross_section, bins=30, alpha=self.config.alpha, density=True)
            ax2.axvline(frame.confidence_interval[0], color='red', linestyle='--', alpha=0.7)
            ax2.axvline(frame.confidence_interval[1], color='red', linestyle='--', alpha=0.7)
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
            anim.save(output_dir / 'animation.gif', writer='pillow', fps=self.config.animation_fps)
            logger.info(f"Saved animation to {output_dir / 'animation.gif'}")

        return anim

    def plot_comparison(self,
                       models: List[Any],
                       model_names: List[str],
                       output_dir: Optional[Path] = None) -> Figure:
        """
        Plot comparison of multiple models.

        Parameters:
            models: List of JumpRope models
            model_names: Names for each model
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating model comparison plot")

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

        axes[0, 1].set_title('Final Distributions')
        axes[0, 1].set_xlabel('Phenotype Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'model_comparison.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {output_dir / 'model_comparison.png'}")

        return fig

    def plot_bayesian_analysis(self,
                              bayesian_result,
                              output_dir: Optional[Path] = None) -> Figure:
        """
        Plot Bayesian analysis results.

        Parameters:
            bayesian_result: BayesianResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating Bayesian analysis plot")

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
                ci_plot.boxplot(intervals, labels=list(bayesian_result.credible_intervals.keys()))
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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'bayesian_analysis.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved Bayesian analysis plot to {output_dir / 'bayesian_analysis.png'}")

        return fig

    def plot_network_analysis(self,
                            network_result,
                            output_dir: Optional[Path] = None) -> Figure:
        """
        Plot network analysis results.

        Parameters:
            network_result: NetworkResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating network analysis plot")

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
            except:
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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'network_analysis.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved network analysis plot to {output_dir / 'network_analysis.png'}")

        return fig

    def plot_dimensionality_reduction(self,
                                    dimensionality_result,
                                    output_dir: Optional[Path] = None) -> Figure:
        """
        Plot dimensionality reduction results.

        Parameters:
            dimensionality_result: DimensionalityResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating dimensionality reduction plot")

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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'dimensionality_reduction.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved dimensionality reduction plot to {output_dir / 'dimensionality_reduction.png'}")

        return fig

    def plot_spectral_analysis(self,
                             spectral_result,
                             output_dir: Optional[Path] = None) -> Figure:
        """
        Plot spectral analysis results.

        Parameters:
            spectral_result: SpectralResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating spectral analysis plot")

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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'spectral_analysis.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved spectral analysis plot to {output_dir / 'spectral_analysis.png'}")

        return fig

    def plot_nonlinear_dynamics(self,
                              nonlinear_result,
                              output_dir: Optional[Path] = None) -> Figure:
        """
        Plot nonlinear dynamics analysis results.

        Parameters:
            nonlinear_result: NonlinearResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating nonlinear dynamics plot")

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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'nonlinear_dynamics.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved nonlinear dynamics plot to {output_dir / 'nonlinear_dynamics.png'}")

        return fig

    def plot_information_theory(self,
                              information_result,
                              output_dir: Optional[Path] = None) -> Figure:
        """
        Plot information theory analysis results.

        Parameters:
            information_result: InformationResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating information theory plot")

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
                axes[0, 1].imshow(mi_data, cmap='viridis', aspect='auto')
                axes[0, 1].set_title('Mutual Information Matrix')
                axes[0, 1].set_xlabel('Variable Index')
                axes[0, 1].set_ylabel('Variable Index')
                plt.colorbar(axes[0, 1].imshow(mi_data, cmap='viridis', aspect='auto'), ax=axes[0, 1])

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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'information_theory.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved information theory plot to {output_dir / 'information_theory.png'}")

        return fig

    def plot_robust_statistics(self,
                              robust_result,
                              output_dir: Optional[Path] = None) -> Figure:
        """
        Plot robust statistics analysis results.

        Parameters:
            robust_result: RobustResult from analytics engine
            output_dir: Directory to save plots

        Returns:
            Matplotlib figure
        """
        logger.info("Creating robust statistics plot")

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

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'robust_statistics.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved robust statistics plot to {output_dir / 'robust_statistics.png'}")

        return fig
    
    def plot_heatmap(self,
                    jump_rope_model,
                    time_resolution: int = 50,
                    phenotype_resolution: int = 50,
                    output_dir: Optional[Path] = None,
                    interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot density heatmap of trajectory evolution.
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            time_resolution: Number of time bins
            phenotype_resolution: Number of phenotype bins
            output_dir: Directory to save plots
            interactive: Create interactive plot
        
        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating trajectory density heatmap")
        
        if jump_rope_model.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")
        
        trajectories = jump_rope_model.trajectories
        time_points = jump_rope_model.time_points
        
        # Remove NaN values
        trajectories_clean = np.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create time and phenotype grids
        time_edges = np.linspace(time_points.min(), time_points.max(), time_resolution + 1)
        phenotype_min = np.nanmin(trajectories_clean)
        phenotype_max = np.nanmax(trajectories_clean)
        
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
            
            # Get trajectory values at this time
            values_at_time = trajectories_clean[:, closest_time_idx]
            
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
                xaxis_title='Developmental Time',
                yaxis_title='Phenotype Value',
                width=800,
                height=600
            )
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.write_html(output_dir / 'density_heatmap.html')
                logger.info(f"Saved interactive heatmap to {output_dir / 'density_heatmap.html'}")
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            im = ax.imshow(density_map, aspect='auto', origin='lower',
                          extent=[time_points.min(), time_points.max(), phenotype_min, phenotype_max],
                          cmap='viridis', interpolation='bilinear')
            
            ax.set_xlabel('Developmental Time')
            ax.set_ylabel('Phenotype Value')
            ax.set_title('Trajectory Density Heatmap')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Trajectory Density', rotation=270, labelpad=20)
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / 'density_heatmap.png', dpi=self.config.dpi, bbox_inches='tight')
                logger.info(f"Saved heatmap to {output_dir / 'density_heatmap.png'}")
            
            return fig
    
    def plot_violin(self,
                   jump_rope_model,
                   time_points: Optional[List[float]] = None,
                   output_dir: Optional[Path] = None) -> Figure:
        """
        Plot violin plots showing distribution at multiple time points.
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            time_points: Specific time points to plot (if None, use evenly spaced)
            output_dir: Directory to save plots
        
        Returns:
            Matplotlib figure
        """
        logger.info("Creating violin plots")
        
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
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel('Developmental Time')
        ax.set_ylabel('Phenotype Value')
        ax.set_title('Phenotype Distribution Evolution (Violin Plots)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'violin_plots.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved violin plots to {output_dir / 'violin_plots.png'}")
        
        return fig
    
    def plot_ridge(self,
                  jump_rope_model,
                  n_distributions: int = 10,
                  output_dir: Optional[Path] = None) -> Figure:
        """
        Plot ridge plot (joyplot) showing distribution evolution over time.
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            n_distributions: Number of distributions to show
            output_dir: Directory to save plots
        
        Returns:
            Matplotlib figure
        """
        logger.info("Creating ridge plot")
        
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
                ax.hist(values, bins=30, alpha=0.7, color=color, density=True)
        
        # Only show x-axis label on bottom plot
        axes[-1].set_xlabel('Phenotype Value')
        axes[-1].spines['bottom'].set_visible(True)
        
        fig.suptitle('Phenotype Distribution Evolution (Ridge Plot)', fontsize=16, y=0.995)
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'ridge_plot.png', dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved ridge plot to {output_dir / 'ridge_plot.png'}")
        
        return fig
    
    def plot_phase_portrait(self,
                           jump_rope_model,
                           derivative_method: str = 'finite_difference',
                           output_dir: Optional[Path] = None,
                           interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot phase portrait (phenotype vs. rate of change).
        
        Parameters:
            jump_rope_model: JumpRope model with trajectories
            derivative_method: Method to compute derivatives ('finite_difference', 'spline')
            output_dir: Directory to save plots
            interactive: Create interactive plot
        
        Returns:
            Matplotlib or Plotly figure
        """
        logger.info("Creating phase portrait")
        
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
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.write_html(output_dir / 'phase_portrait.html')
                logger.info(f"Saved interactive phase portrait to {output_dir / 'phase_portrait.html'}")
            
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
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / 'phase_portrait.png', dpi=self.config.dpi, bbox_inches='tight')
                logger.info(f"Saved phase portrait to {output_dir / 'phase_portrait.png'}")
            
            return fig