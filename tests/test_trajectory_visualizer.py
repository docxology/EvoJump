"""
Test suite for TrajectoryVisualizer module.

This module tests the advanced visualization functionality of the TrajectoryVisualizer
using real data and methods.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope, trajectory_visualizer


class TestPlotConfig:
    """Test PlotConfig class."""

    def test_plot_config_default_values(self):
        """Test PlotConfig with default values."""
        config = trajectory_visualizer.PlotConfig()

        assert config.figsize == (12, 8)
        assert config.dpi == 100
        assert config.style == 'default'
        assert config.alpha == 0.7
        assert config.linewidth == 2.0
        assert bool(config.show_confidence_intervals) is True
        assert config.animation_fps == 30

    def test_plot_config_custom_values(self):
        """Test PlotConfig with custom values."""
        config = trajectory_visualizer.PlotConfig(
            figsize=(16, 10),
            dpi=150,
            style='ggplot',
            alpha=0.5,
            linewidth=3.0,
            show_confidence_intervals=False
        )

        assert config.figsize == (16, 10)
        assert config.dpi == 150
        assert config.style == 'ggplot'
        assert config.alpha == 0.5
        assert config.linewidth == 3.0
        assert config.show_confidence_intervals is False


class TestTrajectoryVisualizer:
    """Test TrajectoryVisualizer class."""

    def create_test_model(self):
        """Create test JumpRope model for visualization tests."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 9, 11, 13, 15, 17]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.array([1, 2, 3, 4, 5])
        )

        model.generate_trajectories(n_samples=20, x0=10.0)

        return model

    def test_trajectory_visualizer_initialization(self):
        """Test TrajectoryVisualizer initialization."""
        visualizer = trajectory_visualizer.TrajectoryVisualizer()

        assert visualizer.config is not None
        assert isinstance(visualizer.config, trajectory_visualizer.PlotConfig)

    def test_trajectory_visualizer_with_custom_config(self):
        """Test TrajectoryVisualizer with custom configuration."""
        config = trajectory_visualizer.PlotConfig(figsize=(10, 6), dpi=120)
        visualizer = trajectory_visualizer.TrajectoryVisualizer(config)

        assert visualizer.config.figsize == (10, 6)
        assert visualizer.config.dpi == 120

    def test_plot_trajectories_static(self):
        """Test static trajectory plotting."""
        model = self.create_test_model()

        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        fig = visualizer.plot_trajectories(model, n_trajectories=5, interactive=False)

        assert fig is not None
        assert hasattr(fig, 'get_axes')
        plt.close(fig)  # Clean up

    def test_plot_trajectories_with_confidence_intervals(self):
        """Test trajectory plotting with confidence intervals."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig(show_confidence_intervals=True)
        visualizer = trajectory_visualizer.TrajectoryVisualizer(config)

        fig = visualizer.plot_trajectories(model, n_trajectories=5, interactive=False, show_ci=True)

        assert fig is not None
        plt.close(fig)

    def test_plot_cross_sections_static(self):
        """Test static cross-section plotting."""
        model = self.create_test_model()

        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        fig = visualizer.plot_cross_sections(
            model,
            time_points=[1.0, 3.0, 5.0],
            interactive=False
        )

        assert fig is not None
        assert hasattr(fig, 'get_axes')
        plt.close(fig)

    def test_plot_landscapes_static(self):
        """Test static landscape plotting."""
        model = self.create_test_model()

        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        fig = visualizer.plot_landscapes(model, interactive=False)

        assert fig is not None
        assert hasattr(fig, 'get_axes')
        plt.close(fig)

    def test_plot_comparison(self):
        """Test model comparison plotting."""
        model1 = self.create_test_model()
        model2 = self.create_test_model()

        # Modify second model to be different
        model2.fitted_parameters.equilibrium = 15.0

        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        fig = visualizer.plot_comparison([model1, model2], ['Model1', 'Model2'])

        assert fig is not None
        assert hasattr(fig, 'get_axes')
        plt.close(fig)

    def test_animation_controller_generation(self):
        """Test animation frame generation."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig()
        controller = trajectory_visualizer.AnimationController(model, config)

        frames = controller.generate_frames(n_frames=5, time_range=(1.0, 5.0))

        assert len(frames) > 0
        assert all(isinstance(frame, trajectory_visualizer.AnimationFrame) for frame in frames)
        assert all(frame.time_point >= 1.0 and frame.time_point <= 5.0 for frame in frames)

    def test_animation_controller_empty_generation(self):
        """Test animation frame generation with no valid frames."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig()
        controller = trajectory_visualizer.AnimationController(model, config)

        frames = controller.generate_frames(n_frames=0)

        assert len(frames) == 0

    def test_create_animation(self):
        """Test animation creation."""
        model = self.create_test_model()

        visualizer = trajectory_visualizer.TrajectoryVisualizer()

        # Test with small number of frames for speed
        model.trajectories = model.trajectories[:3]  # Reduce trajectories for speed
        model.time_points = model.time_points[:3]   # Reduce time points for speed

        anim = visualizer.create_animation(model, n_frames=3)

        assert anim is not None
        assert hasattr(anim, '_func')
        # Check that animation has the basic structure (frames attribute may not always be present)
        assert hasattr(anim, '_iter_gen') or hasattr(anim, '_frames')

    def test_save_plots_to_directory(self):
        """Test saving plots to directory."""
        model = self.create_test_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            visualizer = trajectory_visualizer.TrajectoryVisualizer()
            fig = visualizer.plot_trajectories(model, interactive=False)

            # Should not raise exception
            try:
                fig.savefig(output_dir / 'test_plot.png')
                assert (output_dir / 'test_plot.png').exists()
            finally:
                plt.close(fig)

    def test_plot_with_insufficient_trajectories(self):
        """Test plotting with insufficient trajectories."""
        model = self.create_test_model()

        # Remove trajectories
        model.trajectories = None

        visualizer = trajectory_visualizer.TrajectoryVisualizer()

        with pytest.raises(ValueError, match="No trajectories available"):
            visualizer.plot_trajectories(model)

    def test_plot_with_insufficient_data_points(self):
        """Test plotting with insufficient data points."""
        model = self.create_test_model()

        # Reduce data points
        model.time_points = np.array([1, 2])
        model.trajectories = model.trajectories[:, :2]

        visualizer = trajectory_visualizer.TrajectoryVisualizer()
        fig = visualizer.plot_trajectories(model, interactive=False)

        assert fig is not None
        plt.close(fig)

    def test_plot_config_colors(self):
        """Test plot configuration colors."""
        config = trajectory_visualizer.PlotConfig()

        assert len(config.colors) >= 10  # Should have at least 10 colors
        assert all(isinstance(color, str) for color in config.colors)
        assert all(color.startswith('#') for color in config.colors)

    def test_plot_with_custom_style(self):
        """Test plotting with custom matplotlib style."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig(style='default')
        visualizer = trajectory_visualizer.TrajectoryVisualizer(config)

        # Test that matplotlib style is applied
        original_style = plt.style.available[0] if plt.style.available else 'default'

        try:
            fig = visualizer.plot_trajectories(model, interactive=False)
            assert fig is not None
            plt.close(fig)
        except Exception:
            # If style doesn't exist, should still work with default
            config.style = 'default'
            visualizer = trajectory_visualizer.TrajectoryVisualizer(config)
            fig = visualizer.plot_trajectories(model, interactive=False)
            plt.close(fig)

    def test_animation_frame_properties(self):
        """Test AnimationFrame properties."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig()
        controller = trajectory_visualizer.AnimationController(model, config)

        frames = controller.generate_frames(n_frames=3)

        if len(frames) > 0:
            frame = frames[0]
            assert hasattr(frame, 'time_point')
            assert hasattr(frame, 'trajectories')
            assert hasattr(frame, 'cross_section')
            assert hasattr(frame, 'confidence_interval')
            assert isinstance(frame.confidence_interval, tuple)
            assert len(frame.confidence_interval) == 2

    def test_trajectory_visualizer_seaborn_availability(self):
        """Test that TrajectoryVisualizer handles seaborn availability."""
        visualizer = trajectory_visualizer.TrajectoryVisualizer()

        # Should work regardless of seaborn availability
        assert hasattr(trajectory_visualizer, 'HAS_SEABORN')

        # Test that plotting still works
        model = self.create_test_model()
        fig = visualizer.plot_trajectories(model, interactive=False)
        assert fig is not None
        plt.close(fig)
