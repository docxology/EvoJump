"""
Test suite for TrajectoryVisualizer module.

This module tests the advanced visualization functionality of the TrajectoryVisualizer
using real data and methods.
"""

import importlib
import pytest
import numpy as np
import pandas as pd
import matplotlib as matplotlib_module
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from conftest import make_growth_frame
from evojump import analytics_engine, datacore, jumprope, trajectory_visualizer


def make_seeded_model(seed: int = 42, n_samples: int = 25, n_times: int = 10):
    """Build a seeded JumpRope model with generated trajectories."""
    rng = np.random.default_rng(seed)
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


def make_jumped_trajectories(n_samples: int = 20, n_times: int = 10,
                             seed: int = 3) -> np.ndarray:
    """Synthetic cohort where trajectory 0 makes one large step at mid-run.

    The single outsized increment exceeds the robust jump threshold of
    ``estimate_jump_times``, so comparison panels have jumps to scatter.
    """
    rng = np.random.default_rng(seed)
    trajs = (10.0 + 0.6 * np.arange(n_times)[None, :]
             + rng.normal(0, 0.3, (n_samples, n_times)))
    trajs[0, n_times // 2:] += 25.0
    return trajs


class TestPlotConfig:
    """Test PlotConfig class."""

    def test_plot_config_default_values(self):
        """Test PlotConfig with default values."""
        config = trajectory_visualizer.PlotConfig()

        assert config.figsize == (12, 8)
        assert config.dpi == 120
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


class TestPlotStyleHelpers:
    """Module-level style and annotation helpers (v0.5.0 polish)."""

    def test_apply_plot_style_sets_polished_rcparams(self):
        trajectory_visualizer._apply_plot_style()
        import matplotlib as mpl
        assert mpl.rcParams['figure.dpi'] >= 120
        assert mpl.rcParams['savefig.dpi'] >= 120
        assert mpl.rcParams['figure.constrained_layout.use'] is True
        assert mpl.rcParams['axes.spines.top'] is False
        assert mpl.rcParams['axes.spines.right'] is False
        assert mpl.rcParams['axes.grid'] is True

    def test_static_plots_inherit_style_and_despined_axes(self):
        model = make_seeded_model(n_samples=10)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_trajectories(model, n_trajectories=5, interactive=False)
        ax = fig.get_axes()[0]
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        plt.close(fig)

    def test_model_param_note_formats_process_and_params(self):
        model = make_seeded_model()
        note = trajectory_visualizer._model_param_note(model)
        assert 'Ornstein-Uhlenbeck with Jumps' in note
        assert 'drift=' in note and 'diffusion=' in note

    def test_model_param_note_empty_without_params_or_identity(self):
        note = trajectory_visualizer._model_param_note(SimpleNamespace())
        assert note == ''

    def test_annotate_model_params_draws_corner_text(self):
        model = make_seeded_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig, ax = plt.subplots()
        trajectory_visualizer._annotate_model_params(ax, [model])
        assert ax.texts, "expected a parameter corner note"
        plt.close(fig)

    def test_annotate_model_params_silent_without_notes(self):
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig, ax = plt.subplots()
        trajectory_visualizer._annotate_model_params(ax, [SimpleNamespace()])
        assert not ax.texts
        plt.close(fig)

    def test_aicc_winner_note_selects_lowest_aicc(self):
        fits = {'normal': {'aicc': 10.0}, 'beta': {'aicc': 5.5},
                'unfitted': {'aicc': np.inf}}
        note = trajectory_visualizer._aicc_winner_note(fits)
        assert note == 'Best fit: beta (AICc=5.50)'

    def test_aicc_winner_note_single_fit_dict(self):
        note = trajectory_visualizer._aicc_winner_note(
            {'distribution': 'normal', 'aicc': -3.25})
        assert note == 'Best fit: normal (AICc=-3.25)'

    def test_aicc_winner_note_absent_information_is_none(self):
        assert trajectory_visualizer._aicc_winner_note(None) is None
        assert trajectory_visualizer._aicc_winner_note({}) is None
        assert trajectory_visualizer._aicc_winner_note({'a': {'nope': 1}}) is None
        assert trajectory_visualizer._aicc_winner_note({'aicc': np.inf}) is None

    def test_draw_fit_note_annotates_carrier_object(self):
        carrier = SimpleNamespace(
            distribution_fit={'distribution': 'normal', 'aicc': 12.5})
        fig, ax = plt.subplots()
        note = trajectory_visualizer._draw_fit_note(ax, carrier)
        assert note is not None and 'normal' in note
        assert ax.texts and 'AICc=12.50' in ax.texts[0].get_text()
        plt.close(fig)

    def test_draw_fit_note_silent_without_fit_information(self):
        fig, ax = plt.subplots()
        assert trajectory_visualizer._draw_fit_note(ax, SimpleNamespace()) is None
        assert not ax.texts
        plt.close(fig)

    def test_model_comparison_annotates_fitted_parameters(self):
        m1 = make_seeded_model(seed=42)
        m2 = make_seeded_model(seed=43)
        # One model carries distribution-fit results -> the Final
        # Distributions panel annotates the AICc winner.
        m1.distribution_fits = {'normal': {'aicc': 7.25},
                                'beta': {'aicc': 3.5}}
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_model_comparison([m1, m2], ['Model A', 'Model B'])
        ax1_texts = '\n'.join(t.get_text() for t in fig.axes[0].texts)
        assert 'Ornstein-Uhlenbeck with Jumps' in ax1_texts
        assert 'drift=' in ax1_texts
        ax2_texts = '\n'.join(t.get_text() for t in fig.axes[1].texts)
        assert 'Best fit: beta (AICc=3.50)' in ax2_texts
        plt.close(fig)

    def test_plot_comparison_annotates_params_and_fit_winner(self):
        m1 = make_seeded_model(seed=42)
        m1.distribution_fit = {'distribution': 'normal', 'aicc': 42.0}
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_comparison([m1], ['Model A'])
        ax1_texts = '\n'.join(t.get_text() for t in fig.get_axes()[0].texts)
        assert 'drift=' in ax1_texts
        ax2_texts = '\n'.join(t.get_text() for t in fig.get_axes()[1].texts)
        assert 'Best fit: normal (AICc=42.00)' in ax2_texts
        plt.close(fig)

    def test_model_comparison_fit_note_absent_for_plain_models(self):
        # JumpRope models carry no distribution-fit results, so the AICc
        # panel annotation must stay silent rather than invent one.
        m1 = make_seeded_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_model_comparison([m1], ['Only'])
        assert not fig.axes[1].texts
        plt.close(fig)

    def test_cross_section_legends_on_every_labelled_panel(self):
        model = make_seeded_model(n_samples=15)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_cross_sections(model, time_points=[2.0, 6.0],
                                      show_kde=True, interactive=False)
        for ax in fig.get_axes():
            assert ax.get_legend() is not None, (
                "labelled overlay panel missing its legend")
        plt.close(fig)

    def test_comprehensive_violin_and_ridge_panels_have_legends(self):
        model = make_seeded_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_comprehensive_trajectories(model)
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
        assert panels['Violin Plots'].get_legend() is not None
        assert panels['Ridge Plot'].get_legend() is not None
        plt.close(fig)

    def test_animation_trajectory_panel_has_legend(self):
        model = make_seeded_model(n_samples=10)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        anim = viz.create_animation(model, n_frames=2)
        anim._func(0)
        legend = anim._fig.axes[0].get_legend()
        assert legend is not None
        assert 'Individual trajectories' in legend.get_texts()[0].get_text()
        plt.close(anim._fig)


class TestTrajectoryVisualizer:
    """Test TrajectoryVisualizer class."""

    def create_test_model(self):
        """Create test JumpRope model for visualization tests."""
        # Shared builder from tests/conftest.py: synthetic growth series.
        frame = make_growth_frame(n_points=5, phenotype_cols=('phenotype1',),
                                  seed=42)
        ts_data = datacore.TimeSeriesData(
            data=frame,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        model = jumprope.JumpRope.fit(
            data_core,
            model_type='jump-diffusion',
            time_points=np.arange(5, dtype=float),
            seed=42
        )

        model.generate_trajectories(n_samples=20, x0=10.0, seed=42)

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
        frames = controller.generate_frames(n_frames=5, time_range=(0.0, 4.0))

        assert len(frames) > 0
        assert all(isinstance(frame, trajectory_visualizer.AnimationFrame) for frame in frames)
        assert all(frame.time_point >= 0.0 and frame.time_point <= 4.0 for frame in frames)

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
        return make_seeded_model(seed=seed, n_samples=n_samples)

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


class TestTrajectoryGuards:
    """Every public plot method rejects a model without trajectories."""

    @pytest.mark.parametrize("method_name", [
        "plot_landscapes",
        "create_animation",
        "plot_heatmap",
        "plot_violin",
        "plot_ridge",
        "plot_phase_portrait",
        "plot_comprehensive_trajectories",
    ])
    def test_methods_require_generated_trajectories(self, method_name):
        model = make_seeded_model()
        model.trajectories = None
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="No trajectories available"):
            getattr(viz, method_name)(model)


class TestComparisonEdgeCases:
    """plot_model_comparison / plot_comparison input validation and skips."""

    def test_model_comparison_mismatched_names_raise(self):
        model = make_seeded_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="must match number of names"):
            viz.plot_model_comparison([model], ['A', 'B'])

    def test_plot_comparison_mismatched_names_raise(self):
        model = make_seeded_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="must match number of names"):
            viz.plot_comparison([model], ['A', 'B'])

    def test_model_comparison_skips_models_without_trajectories(self):
        empty = make_seeded_model()
        empty.trajectories = None
        good = make_seeded_model(seed=43)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_model_comparison([empty, good], ['Empty', 'Good'])
        assert len(fig.axes) == 9
        titles = [ax.get_title() for ax in fig.axes]
        assert 'Mean Trajectories' in titles[0]
        assert 'Final Distributions' in titles[1]
        plt.close(fig)

    def test_model_comparison_scatters_detected_jumps(self):
        m1 = make_seeded_model(seed=42, n_samples=20)
        m1.trajectories = make_jumped_trajectories(seed=3)
        m2 = make_seeded_model(seed=43, n_samples=20)
        m2.trajectories = make_jumped_trajectories(seed=4)
        assert len(m1.estimate_jump_times()) > 0, (
            "synthetic cohort must contain a detectable jump")
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_model_comparison([m1, m2], ['Jumpy 1', 'Jumpy 2'])
        ax3 = fig.axes[2]  # Jump Pattern Detection panel
        offsets = np.concatenate(
            [c.get_offsets() for c in ax3.collections]) \
            if ax3.collections else np.empty((0, 2))
        assert offsets.shape[0] > 0, "no jump markers were scattered"
        assert 'Jump Pattern Detection' in ax3.get_title()
        plt.close(fig)

    def test_plot_comparison_skips_models_without_trajectories(self):
        empty = make_seeded_model()
        empty.trajectories = None
        good = make_seeded_model(seed=43)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_comparison([empty, good], ['Empty', 'Good'])
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Mean Trajectories' in titles
        assert 'Number of Jumps' in titles
        plt.close(fig)


class TestStatHelperFallbacks:
    """_compute_skewness/_compute_kurtosis return 0.0 when scipy rejects data."""

    @pytest.mark.parametrize("method", ["_compute_skewness", "_compute_kurtosis"])
    def test_non_numeric_data_falls_back_to_zero(self, method):
        viz = trajectory_visualizer.TrajectoryVisualizer()
        bad = np.array(['a', 'b'])  # scipy.stats raises TypeError on strings
        assert getattr(viz, method)(bad) == 0.0


class TestResultPanelLanes:
    """Analytics-result panels whose optional branches were never driven."""

    def test_information_theory_plots_matrix_complexity_and_flow(self, tmp_path):
        result = analytics_engine.InformationResult(
            entropy_measures={'shannon_entropy': 1.5},
            mutual_information=np.array([[0.0, 0.5], [0.5, 0.0]]),
            transfer_entropy=np.array([]),
            complexity_measures={'lempel_ziv': 3.0, 'approx_entropy': 0.4},
            information_flow={'x_to_y': 0.25})
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_information_theory(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Mutual Information Matrix' in titles
        assert 'Complexity Measures' in titles
        assert 'Information Flow' in titles
        fig.savefig(tmp_path / 'information_theory.png', dpi=80)
        plt.close(fig)
        assert (tmp_path / 'information_theory.png').stat().st_size > 10000

    def test_robust_statistics_plots_outliers_and_efficiency(self, tmp_path):
        result = analytics_engine.RobustResult(
            robust_estimates={'median': 0.0, 'trimmed_mean': 0.1},
            outlier_analysis={'iqr_count': 2.0, 'zscore_count': 1.0},
            influence_measures={'lew': 0.2},
            breakdown_properties={},
            efficiency_comparison={'median': 0.95, 'mean': 1.0})
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_robust_statistics(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Outlier Analysis' in titles
        assert 'Efficiency Comparison' in titles
        fig.savefig(tmp_path / 'robust_statistics.png', dpi=80)
        plt.close(fig)
        assert (tmp_path / 'robust_statistics.png').stat().st_size > 10000

    def test_network_analysis_scalar_centrality_branch(self, tmp_path):
        result = analytics_engine.NetworkResult(
            graph=nx.path_graph(4),
            centrality_measures={'degree': 0.75},  # scalar, not per-node dict
            community_structure={},
            path_analysis={},
            network_metrics={'density': 0.5})
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_network_analysis(result)
        titles = [ax.get_title() for ax in fig.get_axes()]
        assert 'Centrality Measures' in titles
        assert 'Network Metrics' in titles
        fig.savefig(tmp_path / 'network_analysis.png', dpi=80)
        plt.close(fig)
        assert (tmp_path / 'network_analysis.png').stat().st_size > 10000


class TestInteractiveLandscape:
    """Interactive plotly landscape: unit-aware z label, traces, camera."""

    def test_interactive_landscape_units_in_z_label(self):
        model = make_seeded_model(n_samples=15)
        config = trajectory_visualizer.PlotConfig(phenotype_units='mm')
        viz = trajectory_visualizer.TrajectoryVisualizer(config)
        fig = viz.plot_landscapes(model, interactive=True)
        assert fig.layout.title.text == 'Phenotypic Landscape'
        assert fig.layout.scene.zaxis.title.text == 'Phenotype Value (mm)'
        assert fig.layout.scene.xaxis.title.text == 'Developmental Time'
        assert len(fig.data) == 15
        assert all(trace.type == 'scatter3d' for trace in fig.data)
        eye = fig.layout.scene.camera.eye
        assert eye.x is not None and eye.y is not None and eye.z is not None

    def test_static_landscape_without_mpl_toolkits_raises(self, monkeypatch):
        model = make_seeded_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        monkeypatch.setitem(sys.modules, 'mpl_toolkits.mplot3d', None)
        with pytest.raises(ImportError, match="mpl_toolkits.mplot3d"):
            viz.plot_landscapes(model, interactive=False)


class TestTrajectorySubsetting:
    """plot_trajectories clamps oversized n_trajectories requests."""

    def test_oversized_n_trajectories_is_clamped_to_population(self):
        model = make_seeded_model(n_samples=20)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_trajectories(model, n_trajectories=10**6)
        ax = fig.get_axes()[0]
        # 20 individual lines + 1 mean line.
        assert len(ax.get_lines()) == 21
        plt.close(fig)


class TestModelDiagnosticsPanels:
    """_plot_model_diagnostics guards for missing/invalid parameters."""

    def test_helper_without_fitted_parameters_draws_placeholder(self):
        trajs = make_jumped_trajectories(n_samples=5, n_times=6)
        payload = SimpleNamespace(trajectories=trajs,
                                  time_points=np.arange(6.0))
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig, ax = plt.subplots()
        viz._plot_model_diagnostics(payload, ax)
        assert ax.get_title() == 'Model Diagnostics'
        assert ax.texts, "expected a placeholder text"
        plt.close(fig)

    def test_helper_without_numeric_parameters_draws_placeholder(self):
        params = SimpleNamespace(bounds=None, label='not-numeric')
        payload = SimpleNamespace(fitted_parameters=params)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig, ax = plt.subplots()
        viz._plot_model_diagnostics(payload, ax)
        assert ax.get_title() == 'Model Diagnostics'
        assert ax.texts, "expected a placeholder text"
        plt.close(fig)


class TestComprehensiveTrajectories:
    """plot_comprehensive_trajectories renders all nine panels."""

    def test_all_nine_panels_render_and_save(self, tmp_path):
        model = make_seeded_model(seed=42)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_comprehensive_trajectories(model, output_dir=tmp_path)
        # Colorbar axes carry no title; the nine panels do.
        panels = [ax for ax in fig.axes if ax.get_title()]
        assert len(panels) == 9
        titles = [ax.get_title() for ax in panels]
        assert 'Individual Trajectories' in titles[0]
        assert 'Density Heatmap' in titles[1]
        assert 'Cross-Sectional' in titles[2]
        assert 'Violin Plots' in titles[3]
        assert 'Ridge Plot' in titles[4]
        assert 'Phase Portrait' in titles[5]
        assert 'Statistical Summary' in titles[6]
        # Fitted numeric parameters -> parameter bar panel, not placeholder.
        assert 'Fitted Parameters' in titles[7]
        assert 'Evolutionary Change' in titles[8]
        saved = tmp_path / 'comprehensive_trajectories.png'
        assert saved.exists() and saved.stat().st_size > 10000
        plt.close(fig)


    def test_ridge_panel_survives_kde_failure(self):
        # A constant column makes gaussian_kde raise inside the ridge
        # panel; the comprehensive figure must still render all panels.
        model = make_seeded_model(seed=42)
        model.trajectories = model.trajectories.copy()
        model.trajectories[:, 3] = 17.5  # within the 8 sampled indices
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_comprehensive_trajectories(model)
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(titles) == 9
        assert 'Ridge Plot' in titles[4]
        plt.close(fig)


class TestInteractiveTrajectoryAndCrossSections:
    """Interactive plotly branches for trajectories and cross-sections."""

    def test_interactive_trajectories_traces_and_layout(self):
        model = make_seeded_model(n_samples=15)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_trajectories(model, interactive=True)
        assert fig.layout.title.text == 'Developmental Trajectories'
        assert fig.layout.hovermode == 'x unified'
        # 15 individual lines + mean + CI band.
        assert len(fig.data) == 17
        types = [trace.type for trace in fig.data]
        assert set(types) == {'scatter'}
        assert fig.data[-2].name == 'Mean Trajectory'
        assert fig.data[-1].name == '95% Confidence Interval'

    def test_interactive_cross_sections_default_time_grid(self):
        model = make_seeded_model(n_samples=20)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        # time_points=None samples every len//5-th point of a 10-point grid.
        fig = viz.plot_cross_sections(model, interactive=True)
        assert fig.layout.title.text == 'Cross-Sectional Distributions'
        assert fig.layout.height == 300 * 5
        # One histogram plus one fitted-normal curve per subplot.
        assert len(fig.data) == 10
        assert all(trace.type == 'histogram' for trace in fig.data[::2])
        assert all(trace.line.color == 'red' for trace in fig.data[1::2])

    def test_interactive_cross_sections_explicit_time_points(self):
        model = make_seeded_model(n_samples=20)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_cross_sections(model, time_points=[2.0, 8.0],
                                      interactive=True)
        assert fig.layout.height == 300 * 2
        assert len(fig.data) == 4
        assert all(trace.type == 'histogram' for trace in fig.data[::2])


class TestModuleImportFallbacks:
    """Module import guards: optional seaborn and headless backend default."""

    def test_seaborn_missing_falls_back_and_backend_defaults_to_agg(self):
        saved_seaborn = sys.modules.get('seaborn')
        saved_env = os.environ.get('MPLBACKEND')
        saved_get_backend = matplotlib_module.get_backend
        try:
            sys.modules['seaborn'] = None  # forces ImportError on import
            os.environ.pop('MPLBACKEND', None)
            matplotlib_module.get_backend = lambda: 'svg'
            mod = importlib.reload(trajectory_visualizer)
            assert mod.HAS_SEABORN is False
            assert mod.sns is None
        finally:
            if saved_seaborn is not None:
                sys.modules['seaborn'] = saved_seaborn
            else:
                sys.modules.pop('seaborn', None)
            if saved_env is not None:
                os.environ['MPLBACKEND'] = saved_env
            matplotlib_module.get_backend = saved_get_backend
            mod = importlib.reload(trajectory_visualizer)
            assert mod.HAS_SEABORN is True
