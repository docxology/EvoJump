"""
Lane tests: kernel-density cross-sections in TrajectoryVisualizer.

Renders real matplotlib figures (Agg backend) to PNG and checks the KDE
overlay is present and labelled.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evojump import datacore, jumprope, trajectory_visualizer


def make_model(n_samples: int = 40, seed: int = 11):
    """Build a fitted JumpRope model with generated trajectories."""
    data = pd.DataFrame({
        'time': [1, 2, 3, 4, 5] * 3,
        'phenotype1': [10, 12, 14, 16, 18,
                       11, 13, 15, 17, 19,
                       9, 11, 13, 15, 17],
    })
    ts_data = datacore.TimeSeriesData(
        data=data, time_column='time', phenotype_columns=['phenotype1'])
    data_core = datacore.DataCore([ts_data])
    model = jumprope.JumpRope.fit(
        data_core, model_type='jump-diffusion',
        time_points=np.array([1, 2, 3, 4, 5]), seed=seed)
    model.generate_trajectories(n_samples=n_samples, x0=10.0, seed=seed)
    return model


def _save_and_check(fig, out_path: Path):
    """Render figure to PNG and assert a non-trivial file was produced."""
    fig.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)
    assert out_path.exists(), "PNG file was not written"
    assert out_path.stat().st_size > 5000, (
        f"PNG suspiciously small: {out_path.stat().st_size} bytes")


class TestCrossSectionKDE:
    """KDE overlay option for plot_cross_sections."""

    def test_kde_off_by_default(self):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_cross_sections(model, time_points=[3.0], interactive=False)
        ax = fig.get_axes()[0]
        labels = [ln.get_label() for ln in ax.get_lines()]
        assert not any("KDE" in lbl for lbl in labels)
        plt.close(fig)

    def test_kde_overlay_rendered_and_labelled(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_cross_sections(
            model, time_points=[3.0], interactive=False, show_kde=True)
        ax = fig.get_axes()[0]
        labels = [ln.get_label() for ln in ax.get_lines()]
        assert "KDE (Scott's rule)" in labels, labels
        _save_and_check(fig, tmp_path / "cross_sections_kde.png")

    def test_kde_with_fitted_normal_both_curves(self, tmp_path):
        # Fitted model adds the red 'Fitted Normal' line; both should coexist.
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_cross_sections(
            model, time_points=[3.0, 5.0], interactive=False, show_kde=True)
        all_labels = [ln.get_label() for ax in fig.get_axes()
                      for ln in ax.get_lines()]
        assert "Fitted Normal" in all_labels
        assert "KDE (Scott's rule)" in all_labels
        _save_and_check(fig, tmp_path / "cross_sections_kde_and_fit.png")

    def test_kde_degenerate_constant_data(self):
        # Constant cross-section must not crash (std == 0 skips KDE).
        model = make_model()
        model.trajectories = np.tile(
            model.trajectories[:, :1], (1, model.trajectories.shape[1]))
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_cross_sections(
            model, time_points=[2.0], interactive=False, show_kde=True)
        assert fig is not None
        plt.close(fig)


class TestRidgeAndViolinEdges:
    """plot_ridge / plot_violin axis wrapping, fallbacks, snapping."""

    def test_ridge_single_distribution_wraps_axes(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_ridge(model, n_distributions=1)
        assert len(fig.get_axes()) == 1
        _save_and_check(fig, tmp_path / "ridge_single.png")

    def test_ridge_kde_failure_falls_back_to_histogram(self):
        # gaussian_kde raises on a constant cross-section (singular
        # covariance); the ridge panel must fall back to a histogram.
        model = make_model()
        model.trajectories = model.trajectories.copy()
        model.trajectories[:, -1] = 15.0  # last sampled column is constant
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_ridge(model, n_distributions=2)
        axes = fig.get_axes()
        assert len(axes) == 2
        assert axes[-1].patches, "expected histogram fallback bars"
        plt.close(fig)

    def test_violin_explicit_time_points_snap_to_closest(self):
        model = make_model()  # time grid 1..5
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_violin(model, time_points=[2.4, 4.6])
        labels = [t.get_text() for t in fig.get_axes()[0].get_xticklabels()]
        assert labels == ['t=2.00', 't=5.00'], labels
        plt.close(fig)


class TestPhasePortraitDerivativeMethods:
    """plot_phase_portrait spline path, its fallback, and validation."""

    def test_spline_method_renders(self, tmp_path):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_phase_portrait(model, derivative_method='spline',
                                      interactive=False)
        ax = fig.get_axes()[0]
        assert ax.collections, "expected scatter points"
        _save_and_check(fig, tmp_path / "phase_spline.png")

    def test_spline_falls_back_to_finite_differences(self, tmp_path):
        # 3 time points cannot support a cubic UnivariateSpline; every
        # trajectory must fall back to finite differences without error.
        model = make_model()
        model.time_points = np.array([1.0, 2.0, 3.0])
        model.trajectories = model.trajectories[:, :3]
        viz = trajectory_visualizer.TrajectoryVisualizer()
        fig = viz.plot_phase_portrait(model, derivative_method='spline',
                                      interactive=False)
        assert fig.get_axes()[0].collections
        _save_and_check(fig, tmp_path / "phase_spline_fallback.png")

    def test_unknown_derivative_method_raises(self):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        with pytest.raises(ValueError, match="Unknown derivative method"):
            viz.plot_phase_portrait(model, derivative_method='wronskian')
