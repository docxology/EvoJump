"""
Lane tests: trailing-window animation mode in create_animation.

Checks frame content, fixed axes, and rendered GIF output via the
real matplotlib animation machinery (Agg backend, pillow writer).
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


def make_model(n_samples: int = 12, n_times: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(15):
        for t in range(1, n_times + 1):
            rows.append((float(t), 10.0 + 0.5 * t + 0.3 * i + rng.normal(0, 0.4)))
    data = pd.DataFrame(rows, columns=['time', 'phenotype1'])
    ts_data = datacore.TimeSeriesData(
        data=data, time_column='time', phenotype_columns=['phenotype1'])
    data_core = datacore.DataCore([ts_data])
    model = jumprope.JumpRope.fit(
        data_core, model_type='jump-diffusion',
        time_points=np.arange(1, n_times + 1, dtype=float), seed=seed)
    model.generate_trajectories(n_samples=n_samples, x0=10.0, seed=seed)
    return model


class TestTrailingWindowAnimation:
    """create_animation(trailing_window=k) shows only the last k time points."""

    def test_trailing_window_accepted_and_animates(self):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        anim = viz.create_animation(model, n_frames=3, trailing_window=3)
        assert anim is not None
        plt.close(anim._fig)

    def test_trailing_window_limits_drawn_points(self):
        # Drive the animate callback directly and inspect the plotted data.
        model = make_model(n_times=10)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        anim = viz.create_animation(model, n_frames=4, trailing_window=3)
        animate = anim._func
        # Last frame: full history available, window should clip to 3 points.
        animate(3)
        ax1, ax2 = anim._fig.axes[0], anim._fig.axes[1]
        segments = [ln for ln in ax1.get_lines()]
        assert segments, "no trajectory lines drawn"
        max_pts = max(len(ln.get_xdata()) for ln in segments)
        assert max_pts <= 3, f"trailing window exceeded: {max_pts} points"
        # Fixed axes preserved against ax1.clear() per frame.
        assert ax1.get_xlim() == (float(model.time_points[0]),
                                  float(model.time_points[-1]))
        plt.close(anim._fig)

    def test_no_trailing_window_draws_full_history(self):
        model = make_model(n_times=10)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        anim = viz.create_animation(model, n_frames=4)
        animate = anim._func
        animate(3)
        ax1 = anim._fig.axes[0]
        segments = [ln for ln in ax1.get_lines()]
        assert segments
        max_pts = max(len(ln.get_xdata()) for ln in segments)
        assert max_pts == 10, f"expected full history, got {max_pts}"
        plt.close(anim._fig)

    def test_title_reports_window(self):
        model = make_model()
        viz = trajectory_visualizer.TrajectoryVisualizer()
        anim = viz.create_animation(model, n_frames=2, trailing_window=4)
        anim._func(1)
        title = anim._fig.axes[0].get_title()
        assert 'trailing window: 4' in title
        plt.close(anim._fig)

    def test_saved_gif_nontrivial(self, tmp_path):
        model = make_model(n_samples=6, n_times=6)
        viz = trajectory_visualizer.TrajectoryVisualizer()
        anim = viz.create_animation(model, n_frames=3, trailing_window=3,
                                    output_dir=tmp_path)
        gif = tmp_path / 'animation.gif'
        assert gif.exists(), "GIF not saved"
        assert gif.stat().st_size > 5000, gif.stat().st_size
        plt.close(anim._fig)
