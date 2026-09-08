"""
Lane tests: heatmap row sorting, axis labels, and landscape colorbar.

Renders real matplotlib figures (Agg) to PNG in tmp_path and asserts
non-trivial file sizes plus deterministic axis properties.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope, trajectory_visualizer


def make_model(n_samples: int = 40, seed: int = 23):
    """Build a fitted JumpRope model with generated trajectories."""
    rng = np.random.default_rng(seed)
    n_times = 8
    base = np.linspace(10, 20, n_times)
    rows = []
    for i in range(30):
        for t_idx, t in enumerate(range(1, n_times + 1)):
            rows.append((float(t), base[t_idx] + 0.4 * i + rng.normal(0, 0.5)))
    data = pd.DataFrame(rows, columns=['time', 'phenotype1'])
    ts_data = datacore.TimeSeriesData(
        data=data, time_column='time', phenotype_columns=['phenotype1'])
    data_core = datacore.DataCore([ts_data])
    model = jumprope.JumpRope.fit(
        data_core, model_type='jump-diffusion',
        time_points=np.arange(1, n_times + 1, dtype=float), seed=seed)
    model.generate_trajectories(n_samples=n_samples, x0=10.0, seed=seed)
    return model


def _save_and_check(fig, out_path: Path, min_bytes: int = 10000):
    fig.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)
    assert out_path.exists()
    size = out_path.stat().st_size
    assert size > min_bytes, f"PNG too small: {size} bytes"


class TestHeatmapRowSorting:
    """plot_heatmap: documented row statistic, sortable, real labels."""

    def test_default_sorts_by_final_value(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=15,
                               phenotype_resolution=15, interactive=False)
        ax = fig.get_axes()[0]
        assert 'final value' in ax.get_title()
        _save_and_check(fig, tmp_path / "heatmap_sorted.png")

    def test_sort_disabled(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=15,
                               phenotype_resolution=15, interactive=False,
                               sort_rows=False)
        ax = fig.get_axes()[0]
        assert 'sorted' not in ax.get_title()
        _save_and_check(fig, tmp_path / "heatmap_unsorted.png")

    @pytest.mark.parametrize("stat", [
        'final_value', 'mean_value', 'max_value', 'min_value'])
    def test_sort_statistics_supported(self, tmp_path, stat):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=10,
                               phenotype_resolution=10, interactive=False,
                               row_sort_statistic=stat)
        assert stat.replace('_', ' ') in fig.get_axes()[0].get_title()
        _save_and_check(fig, tmp_path / f"heatmap_{stat}.png")

    def test_invalid_statistic_rejected(self):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="row_sort_statistic"):
            viz.plot_heatmap(model, row_sort_statistic='bogus_stat')

    def test_real_axis_labels_override(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=10,
                               phenotype_resolution=10, interactive=False,
                               x_label='age (weeks)', y_label='trait score')
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'age (weeks)'
        assert ax.get_ylabel() == 'trait score'
        plt.close(fig)

    def test_sort_changes_row_order_deterministically(self):
        # Same input twice -> same permutation; sorting actually reorders
        # an unsorted cohort into ascending final-value order.
        model = make_model()
        trajs = model.trajectories.copy()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        final = trajs[:, -1]
        assert not np.all(np.diff(final) >= 0), (
            "test data itself must be unsorted for this assertion to mean anything")
        order1 = viz._trajectory_sort_order(trajs, 'final_value')
        order2 = viz._trajectory_sort_order(trajs, 'final_value')
        assert np.array_equal(order1, order2), "sort must be deterministic"
        assert np.array_equal(trajs[order1][:, -1], np.sort(final)), (
            "applying the order must sort trajectories by final value")
        assert not np.array_equal(order1, np.arange(len(final))), (
            "order must actually reorder an unsorted cohort")


class TestLandscapeDeterminism:
    """plot_landscapes: colorbar with units, deterministic camera."""

    def test_static_landscape_render(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_landscapes(model, interactive=False)
        _save_and_check(fig, tmp_path / "landscape.png")
        ax = fig.axes[0]
        assert ax.get_zlabel() == 'Phenotype Value'

    def test_config_camera_defaults(self):
        config = trajectory_visualizer.PlotConfig()
        assert config.landscape_elevation == 30.0
        assert config.landscape_azimuth == -60.0
        assert isinstance(config.phenotype_units, str)

    def test_custom_units_label(self, tmp_path):
        # config.phenotype_units must propagate to the landscape z-axis
        # label (which the static path draws alongside its axes).
        model = make_model()
        config = trajectory_visualizer.PlotConfig(phenotype_units='mm')
        viz = trajectory_visualizer.TrajectoryVisualizer(config)
        fig = viz.plot_landscapes(model, interactive=False)
        zlabel = fig.axes[0].get_zlabel()
        assert 'mm' in zlabel, f"units not propagated to z-axis label: {zlabel!r}"
        _save_and_check(fig, tmp_path / "landscape_units.png")

    def test_deterministic_camera_angles(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig1 = viz.plot_landscapes(model, interactive=False)
        ax1 = fig1.axes[0]
        _save_and_check(fig1, tmp_path / "landscape_run1.png")
        if hasattr(ax1, 'elev'):
            assert abs(ax1.elev - 30.0) < 1e-6
            assert abs(ax1.azim - (-60.0)) < 1e-6


class TestHeatmapEdges:
    """plot_heatmap: html output, label discovery, degenerate inputs."""

    def test_interactive_heatmap_writes_html(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=10,
                               phenotype_resolution=10, interactive=True,
                               output_dir=tmp_path)
        html = tmp_path / 'density_heatmap.html'
        assert html.exists() and html.stat().st_size > 0
        assert fig.layout.title.text == 'Trajectory Density Heatmap'
        assert fig.layout.xaxis.title.text is not None
        assert fig.layout.yaxis.title.text is not None

    def test_time_label_discovered_from_source_time_series(self):
        model = make_model()
        model._source_time_series = [SimpleNamespace(time_column='age_weeks')]
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=10,
                               phenotype_resolution=10, interactive=False)
        assert fig.get_axes()[0].get_xlabel() == 'age_weeks'
        plt.close(fig)

    def test_all_nan_trajectories_raise(self):
        model = make_model()
        model.trajectories = np.full((3, model.trajectories.shape[1]), np.nan)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="No finite phenotype values"):
            viz.plot_heatmap(model, interactive=False)

    def test_constant_phenotype_expands_extent(self, tmp_path):
        # All-equal values must widen the phenotype range instead of
        # producing a degenerate zero-width histogram axis.
        model = make_model()
        model.trajectories = np.full(model.trajectories.shape, 12.0)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_heatmap(model, time_resolution=8,
                               phenotype_resolution=8, interactive=False,
                               sort_rows=False)
        extent = fig.get_axes()[0].images[0].get_extent()
        assert extent[2] == pytest.approx(11.0)
        assert extent[3] == pytest.approx(13.0)
        _save_and_check(fig, tmp_path / "heatmap_constant.png")

    def test_plot_heatmap_requires_trajectories(self):
        model = make_model()
        model.trajectories = None
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="No trajectories available"):
            viz.plot_heatmap(model, interactive=False)

    def test_sort_order_helper_rejects_unknown_statistic(self):
        trajs = make_model().trajectories
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="Unknown row_sort_statistic"):
            viz._trajectory_sort_order(trajs, 'median')


class TestLandscapeEdges:
    """plot_landscapes guard branches."""

    def test_plot_landscapes_requires_trajectories(self):
        model = make_model()
        model.trajectories = None
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="No trajectories available"):
            viz.plot_landscapes(model, interactive=False)

    def test_static_landscape_without_mpl_toolkits_raises(self, monkeypatch):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        monkeypatch.setitem(sys.modules, 'mpl_toolkits.mplot3d', None)
        with pytest.raises(ImportError, match="mpl_toolkits.mplot3d"):
            viz.plot_landscapes(model, interactive=False)
