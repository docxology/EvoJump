"""Shared pytest fixtures for the EvoJump test suite.

House rules (mirroring tests/README.md and the repo AGENTS.md):
- Real or synthetic data only — never mocks.
- Every stochastic fixture derives from an explicit seed so runs are
  deterministic, including under ``pytest-xdist`` (``-n auto``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_growth_frame(
    n_points: int = 20,
    phenotype_cols: tuple[str, ...] = ("x",),
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic developmental time series: smooth growth + small noise.

    The canonical long-format frame used across module tests: a ``time``
    column of ``0..n_points-1`` and one column per requested phenotype with
    logistic-shaped growth and seeded jitter.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    frame = {"time": t}
    for i, col in enumerate(phenotype_cols):
        growth = 10.0 / (1.0 + np.exp(-(t - n_points / 2) / 2.0)) + i
        frame[col] = growth + rng.normal(0, 0.05, n_points)
    return pd.DataFrame(frame)


def make_population_frame(
    n_individuals: int = 40,
    n_time_points: int = 3,
    trait_cols: tuple[str, ...] = ("trait",),
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic population table for EvolutionSampler/AnalyticsEngine tests.

    Columns: ``time`` (generation), one column per trait (population-mean
    drift + individual noise), and an ``individual_id`` object column.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_time_points):
        for ind in range(n_individuals):
            row: dict[str, float | str] = {"time": float(g)}
            for i, col in enumerate(trait_cols):
                base = 10.0 + 0.5 * g + i
                row[col] = base + rng.normal(0, 1.0)
            row["individual_id"] = f"ind_{ind:03d}"
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def ts_frame() -> pd.DataFrame:
    """Minimal 2-column frame used by sampler/analytics regression tests."""
    return make_growth_frame(n_points=12, phenotype_cols=("x",))


@pytest.fixture
def population_frame() -> pd.DataFrame:
    """Default population table for sampler tests."""
    return make_population_frame()
