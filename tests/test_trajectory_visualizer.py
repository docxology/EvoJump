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
from evojump import analytics_engine, datacore, jumprope, trajectory_visualizer


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
        assert config.animation_interval == 50

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
        """Test plotting with a custom matplotlib style (no exception fallback)."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig(style='default')
        visualizer = trajectory_visualizer.TrajectoryVisualizer(config)

        # The exception-free path must hold: plotting with a valid style
        # raises nothing and returns a figure.
        fig = visualizer.plot_trajectories(model, interactive=False)
        assert fig is not None
        plt.close(fig)

    def test_animation_frame_properties(self):
        """Test AnimationFrame properties."""
        model = self.create_test_model()

        config = trajectory_visualizer.PlotConfig()
        controller = trajectory_visualizer.AnimationController(model, config)

        frames = controller.generate_frames(n_frames=3)

        assert frames, "expected at least one generated frame"
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


class TestAnalyticsAndPanelLanes:
    """Agg lane tests for previously untested public plot methods.

    Follows the test_viz_lane_* pattern: real models and real analytics
    results (never mocks), rendered figures saved to tmp_path with
    non-trivial PNG sizes, and assertions on titles, labels, and panel
    structure.
    """

    def make_model(self, seed: int = 42, n_samples: int = 25):
        """Build a seeded JumpRope model with generated trajectories."""
        rng = np.random.default_rng(seed)
        n_times = 10
        rows = []
        for i in range(n_samples):
            for t in range(1, n_times + 1):
                rows.append((float(t), 10.0 + 0.6 * t + 0.25 * i + rng.normal(0, 0.5)))
        data = pd.DataFrame(rows, columns=['time', 'phenotype1'])
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore([ts_data])
        model = jumprope.JumpRope.fit(
            data_core, model_type='jump-diffusion',
            time_points=np.arange(1, n_times + 1, dtype=float), seed=seed)
        model.generate_trajectories(n_samples=n_samples, x0=10.0, seed=seed)
        return model

    def _save_and_check(self, fig, out_path: Path, min_bytes: int = 10000):
        fig.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        assert out_path.exists()
        size = out_path.stat().st_size
        assert size > min_bytes, f"PNG too small: {size} bytes"

    def test_plot_model_comparison_panels(self, tmp_path):
        m1 = self.make_model(seed=42)
        m2 = self.make_model(seed=43)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_model_comparison([m1, m2], ['Model A', 'Model B'])
        assert len(fig.axes) == 9
        ax1 = fig.axes[0]
        assert 'Mean Trajectories' in ax1.get_title()
        # The comparison band is a +/-1 SD spread, not a confidence interval.
        assert '±1 SD' in ax1.get_title()
        self._save_and_check(fig, tmp_path / 'model_comparison.png')

    def test_plot_bayesian_analysis_lane(self, tmp_path):
        rng = np.random.default_rng(42)
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0 + rng.normal(0, 1.0, x.size)
        result = analytics_engine.BayesianAnalyzer(
            pd.DataFrame({'x': x, 'y': y})).bayesian_linear_regression(x, y)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_bayesian_analysis(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Posterior Distribution' in titles
        assert 'Credible Intervals' in titles
        self._save_and_check(fig, tmp_path / 'bayesian_analysis.png')

    def test_plot_network_analysis_renders_graph(self, tmp_path):
        rng = np.random.default_rng(42)
        base = rng.normal(0.0, 1.0, 60)
        df = pd.DataFrame({
            'a': base,
            'b': base + rng.normal(0, 0.1, 60),
            'c': base + rng.normal(0, 0.2, 60),
            'd': rng.normal(0, 1.0, 60),
        })
        result = analytics_engine.NetworkAnalyzer(df).construct_correlation_network(
            threshold=0.6)
        assert result.graph.number_of_edges() > 0
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_network_analysis(result)
        ax0 = fig.get_axes()[0]
        # Regression: a missing networkx import used to be swallowed by a
        # bare except, rendering the placeholder text instead of the graph.
        texts = [t.get_text() for t in ax0.texts]
        assert not any('too complex' in t for t in texts), (
            "network graph fell back to placeholder text")
        assert len(ax0.collections) >= 2, "node/edge collections not drawn"
        assert any(t for t in texts), "no node labels drawn"
        self._save_and_check(fig, tmp_path / 'network_analysis.png')

    def test_plot_heatmap_static_lane(self, tmp_path):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=12,
                               phenotype_resolution=12, interactive=False)
        ax = fig.get_axes()[0]
        assert 'Trajectory Density Heatmap' in ax.get_title()
        self._save_and_check(fig, tmp_path / 'density_heatmap.png')

    def test_plot_heatmap_interactive_lane(self):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=12,
                               phenotype_resolution=12, interactive=True)
        assert hasattr(fig, 'to_html')  # plotly figure
        assert fig.layout.xaxis.title.text is not None
        z = fig.data[0].z
        assert len(z) == 12 and len(z[0]) == 12

    def test_plot_heatmap_ignores_nan(self, tmp_path):
        # A fully missing visit must be dropped, not imputed to 0: the
        # phenotype extent must not be dragged toward 0 and the affected
        # time bin stays empty.
        model = self.make_model(seed=42)
        trajs = model.trajectories.copy()
        assert np.nanmin(trajs) > 5.0, "test data must sit far from 0"
        trajs[:, 3] = np.nan
        model.trajectories = trajs
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=10,
                               phenotype_resolution=4, interactive=False)
        im = fig.get_axes()[0].images[0]
        extent = im.get_extent()
        assert extent[2] > 5.0, f"phenotype extent dragged toward imputed zeros: {extent}"
        arr = np.asarray(im.get_array())
        assert np.sum(~arr.any(axis=0)) == 1, (
            "exactly one (the NaN) time bin should be empty")
        self._save_and_check(fig, tmp_path / 'heatmap_nan.png')

    def test_plot_violin_lane(self, tmp_path):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_violin(model)
        assert 'Violin' in fig.get_axes()[0].get_title()
        self._save_and_check(fig, tmp_path / 'violin_plots.png')

    def test_plot_ridge_lane(self, tmp_path):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_ridge(model, n_distributions=4)
        assert len(fig.get_axes()) == 4
        self._save_and_check(fig, tmp_path / 'ridge_plot.png')

    def test_plot_phase_portrait_lane(self, tmp_path):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_phase_portrait(model, interactive=False)
        assert 'Phase Portrait' in fig.get_axes()[0].get_title()
        self._save_and_check(fig, tmp_path / 'phase_portrait.png')

    def test_plot_phase_portrait_interactive_lane(self):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_phase_portrait(model, interactive=True)
        assert hasattr(fig, 'to_html')

    def test_plot_dimensionality_reduction_lane(self, tmp_path):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (100, 4)),
                            columns=['f1', 'f2', 'f3', 'f4'])
        result = analytics_engine.DimensionalityReducer(data).fast_ica(n_components=2)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_dimensionality_reduction(result)
        assert 'Dimensionality Reduction Analysis' in fig._suptitle.get_text()
        self._save_and_check(fig, tmp_path / 'dimensionality_reduction.png')

    def test_plot_spectral_analysis_lane(self, tmp_path):
        rng = np.random.default_rng(42)
        t = np.arange(128, dtype=float)
        signal = np.sin(2 * np.pi * 0.1 * t) + 0.5 * rng.normal(0, 1, t.size)
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': t, 'signal': signal}))
        result = engine.spectral_analysis('signal')
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_spectral_analysis(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Power Spectrum' in titles
        self._save_and_check(fig, tmp_path / 'spectral_analysis.png')

    def test_plot_nonlinear_dynamics_lane(self, tmp_path):
        rng = np.random.default_rng(42)
        t = np.arange(120, dtype=float)
        series = np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.normal(0, 1, t.size)
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': t, 'series': series}))
        raw = engine.nonlinear_dynamics_analysis('series', embedding_dim=3)
        assert 'error' not in raw
        result = analytics_engine.NonlinearResult(
            lyapunov_exponents=np.array([raw['largest_lyapunov_exponent']]),
            correlation_dimensions=np.asarray(raw['correlation_dimensions']),
            attractor_properties=raw['attractor_properties'],
            chaos_quantifiers=raw['chaos_quantifiers'],
            recurrence_properties=raw['recurrence_properties'])
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_nonlinear_dynamics(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Lyapunov Spectrum' in titles
        self._save_and_check(fig, tmp_path / 'nonlinear_dynamics.png')

    def test_plot_information_theory_lane(self, tmp_path):
        rng = np.random.default_rng(42)
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'values': rng.normal(0, 1, 200)}))
        raw = engine.information_theory_analysis('values')
        result = analytics_engine.InformationResult(
            entropy_measures=raw,
            mutual_information=np.array([]),
            transfer_entropy=np.array([]),
            complexity_measures={},
            information_flow={})
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_information_theory(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Entropy Measures' in titles
        self._save_and_check(fig, tmp_path / 'information_theory.png')

    def test_plot_robust_statistics_lane(self, tmp_path):
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(0, 1, 100), [50.0]])  # one outlier
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'values': values}))
        raw = engine.robust_statistical_analysis('values')
        result = analytics_engine.RobustResult(
            robust_estimates=raw['location_estimates'],
            outlier_analysis={},
            influence_measures=raw['scale_estimates'],
            breakdown_properties={},
            efficiency_comparison={})
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_robust_statistics(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Location Estimates Comparison' in titles
        self._save_and_check(fig, tmp_path / 'robust_statistics.png')

    def test_close_param_closes_saved_figure(self, tmp_path):
        model = self.make_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        before = plt.get_fignums()
        fig = viz.plot_trajectories(model, interactive=False,
                                    output_dir=tmp_path, close=True)
        assert (tmp_path / 'trajectories.png').exists()
        assert fig is not None
        assert plt.get_fignums() == before, "figure was not closed"
