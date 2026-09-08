"""Property-based invariants over the real EvoJump APIs (hypothesis).

Every property below exercises a public (or documented-behaviour) API on
synthetic, seeded data — no mocks. Global hypothesis settings keep the suite
fast and deterministic: at most 40 examples per property, no deadline,
``derandomize`` for reproducibility, and no example database so runs never
write a ``.hypothesis`` cache into the repo.

Invariants covered
------------------
DataCore
    * ``interpolate_missing_data`` preserves row count and row order, never
      touches the time column, leaves originally-finite values unchanged, and
      is idempotent. Missing time values are rejected with ``ValueError``.
    * Outlier removal (``_remove_outliers`` via ``preprocess_data``'s helper)
      never deletes rows whose phenotype values are missing, deletes rows
      only, and is independent of row order.
AnalyticsEngine
    * Kaplan-Meier ``survival_analysis``: the survival curve is
      non-increasing and confined to [0, 1], pointwise confidence intervals
      bracket the curve, the cumulative hazard is non-decreasing, and the
      reported median is the first time the curve reaches 0.5.
    * Robust estimators (Huber location, Rousseeuw-Croux Sn scale) stay
      within sane bounds on contaminated normal samples.
JumpRope
    * fBM increments: empirical standard deviation scales as
      ``diffusion * dt**hurst``.
    * OU and geometric jump-diffusion log-likelihoods are finite across a
      parameter box and strictly larger at the generating parameters than at
      grossly misspecified ones; the OU likelihood is maximized at the true
      diffusion on a one-dimensional grid.
EvolutionSampler
    * ``PopulationModel.compute_selection_gradient`` equals the standardized
      ``scipy.stats.linregress`` slope (the Pearson correlation) on the
      finite paired observations, and is NaN for degenerate inputs.
LaserPlane
    * ``MomentAnalyzer.compute_confidence_intervals``: the order-statistic
      median interval brackets the sample median, the mean interval brackets
      the sample mean, the std interval brackets the sample std; all-NaN
      input yields NaN triplets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats

from conftest import make_growth_frame, make_population_frame

from evojump import AnalyticsEngine, DataCore, ModelParameters, TimeSeriesData
from evojump.jumprope import (
    FractionalBrownianMotion,
    GeometricJumpDiffusion,
    OrnsteinUhlenbeckJump,
)
from evojump.laserplane import MomentAnalyzer
from evojump.evolution_sampler import PopulationModel

# Shared deterministic, fast profile. ``derandomize`` pins the example
# generation; the stochastic APIs themselves receive explicit seeded
# generators drawn as examples.
_COMMON = dict(
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
P_FAST = settings(max_examples=40, **_COMMON)
P_MODEL = settings(max_examples=20, **_COMMON)  # likelihood grids / simulation
P_SLOW = settings(max_examples=12, **_COMMON)  # python-loop likelihoods

PHENOTYPE_COLS = ("x", "y")
PHENOTYPE_LIST = list(PHENOTYPE_COLS)  # pandas needs a list, not a tuple, to select columns


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _growth_frame_with_gaps(
    n: int,
    seed: int,
    nan_cells: list[tuple[int, str]],
) -> pd.DataFrame:
    """Growth frame with selected (row, phenotype-column) cells set to NaN."""
    frame = make_growth_frame(n_points=n, phenotype_cols=PHENOTYPE_COLS, seed=seed)
    for row, col in nan_cells:
        frame.loc[row, col] = np.nan
    return frame


def _contaminated_growth_frame(
    n: int,
    seed: int,
    all_nan_rows: list[int],
    outlier_rows: list[int],
) -> pd.DataFrame:
    """Growth frame with all-phenotype-NaN rows and planted extreme outliers.

    Outlier magnitude (+1000 above the observed maximum) is far outside any
    IQR fence for these smooth growth series, so the IQR method provably
    drops the planted rows while the NaN-only rows must survive.
    """
    frame = make_growth_frame(n_points=n, phenotype_cols=PHENOTYPE_COLS, seed=seed)
    for row in set(all_nan_rows):
        frame.loc[row, PHENOTYPE_COLS] = np.nan
    for row in set(outlier_rows):
        frame.loc[row, "x"] = float(frame["x"].max()) + 1000.0
    return frame


def _datacore(frame: pd.DataFrame) -> DataCore:
    ts = TimeSeriesData(
        data=frame,
        time_column="time",
        phenotype_columns=list(PHENOTYPE_COLS),
    )
    return DataCore(ts)


# --------------------------------------------------------------------------
# DataCore.interpolate_missing_data
# --------------------------------------------------------------------------
@given(
    n=st.integers(min_value=8, max_value=40),
    seed=st.integers(0, 10_000),
    shuffle_seed=st.integers(0, 10_000),
    data=st.data(),
)
@P_FAST
def test_interpolate_missing_data_preserves_rows_order_and_finite_values(
    n: int, seed: int, shuffle_seed: int, data
) -> None:
    """Interpolation keeps the frame shape and row order, fills only NaNs."""
    cells = data.draw(
        st.lists(
            st.tuples(st.integers(0, n - 1), st.sampled_from(PHENOTYPE_COLS)),
            max_size=n // 2,
        )
    )
    # Every phenotype column must keep at least one finite value, otherwise
    # no interpolation target exists for that column at all.
    per_col = {col: sum(1 for _, c in cells if c == col) for col in PHENOTYPE_COLS}
    assume_all_columns_keep_finite = all(cnt < n for cnt in per_col.values())
    if not assume_all_columns_keep_finite:
        return

    frame = _growth_frame_with_gaps(n, seed, cells)
    rng = np.random.default_rng(shuffle_seed)
    frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)

    core = _datacore(frame)
    original = core.time_series_data[0].data.copy(deep=True)
    finite_mask = original[PHENOTYPE_LIST].notna().to_numpy()

    core.time_series_data[0].interpolate_missing_data()
    after = core.time_series_data[0].data

    # Row count and row order preserved: the time column is untouched,
    # positionally.
    assert len(after) == len(original)
    np.testing.assert_array_equal(after["time"].to_numpy(), original["time"].to_numpy())

    # Originally-finite phenotype values are never altered.
    after_values = after[PHENOTYPE_LIST].to_numpy()
    original_values = original[PHENOTYPE_LIST].to_numpy()
    np.testing.assert_allclose(
        after_values[finite_mask], original_values[finite_mask], rtol=0, atol=0
    )
    # Every previously-missing cell is filled.
    assert after[PHENOTYPE_LIST].notna().all().all()


@given(
    n=st.integers(min_value=8, max_value=30),
    seed=st.integers(0, 10_000),
    gap_row=st.integers(1, 28),
)
@P_FAST
def test_interpolate_missing_data_fills_single_interior_gap_linearly(
    n: int, seed: int, gap_row: int
) -> None:
    gap_row = min(gap_row, n - 2)  # keep strictly interior
    frame = make_growth_frame(n_points=n, phenotype_cols=PHENOTYPE_COLS, seed=seed)
    core = _datacore(frame)
    ts = core.time_series_data[0]
    left, right = float(ts.data["x"].iloc[gap_row - 1]), float(ts.data["x"].iloc[gap_row + 1])
    ts.data.loc[gap_row, "x"] = np.nan

    ts.interpolate_missing_data()

    assert ts.data.loc[gap_row, "x"] == pytest.approx(0.5 * (left + right), rel=1e-9)


@given(
    n=st.integers(min_value=8, max_value=30),
    seed=st.integers(0, 10_000),
    missing_time_row=st.integers(0, 29),
)
@P_FAST
def test_interpolate_missing_data_rejects_missing_time_values(
    n: int, seed: int, missing_time_row: int
) -> None:
    frame = make_growth_frame(n_points=n, phenotype_cols=PHENOTYPE_COLS, seed=seed)
    core = _datacore(frame)
    ts = core.time_series_data[0]
    ts.data.loc[missing_time_row % n, "time"] = np.nan

    with pytest.raises(ValueError, match="(?i)time column"):
        ts.interpolate_missing_data()


@given(
    n=st.integers(min_value=8, max_value=30),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
@P_FAST
def test_interpolate_missing_data_is_idempotent(n: int, seed: int, data) -> None:
    """Interpolating an already-complete frame changes nothing (fixed point)."""
    frame = make_growth_frame(n_points=n, phenotype_cols=PHENOTYPE_COLS, seed=seed)
    core = _datacore(frame)
    ts = core.time_series_data[0]
    ts.interpolate_missing_data()
    once = ts.data.copy(deep=True)

    ts.interpolate_missing_data()

    pd.testing.assert_frame_equal(ts.data, once)


# --------------------------------------------------------------------------
# DataCore outlier removal
# --------------------------------------------------------------------------
@given(
    n=st.integers(min_value=10, max_value=40),
    seed=st.integers(0, 10_000),
    method=st.sampled_from(["iqr", "zscore"]),
    threshold=st.floats(1.0, 3.0),
    data=st.data(),
)
@P_FAST
def test_outlier_removal_never_drops_rows_with_missing_values(
    n: int, seed: int, method: str, threshold: float, data
) -> None:
    """Rows whose phenotype entries are NaN are never classified as outliers."""
    all_nan_rows = data.draw(
        st.lists(st.integers(0, n - 1), max_size=max(1, n // 3), unique=True)
    )
    outlier_rows = data.draw(
        st.lists(
            st.integers(0, n - 1),
            min_size=1,
            max_size=2,
            unique=True,
        )
    ).copy()
    # Planted outliers must live on rows that still have finite values.
    outlier_rows = [r for r in outlier_rows if r not in set(all_nan_rows)]
    if not outlier_rows:
        outlier_rows = [next(r for r in range(n) if r not in set(all_nan_rows))]

    frame = _contaminated_growth_frame(n, seed, all_nan_rows, outlier_rows)
    core = _datacore(frame)
    ts = core.time_series_data[0]

    core._remove_outliers(ts, method=method, threshold=threshold)
    kept_times = set(ts.data["time"])

    # Every all-NaN row survived, in original relative order.
    for row in sorted(set(all_nan_rows)):
        assert frame.loc[row, "time"] in kept_times
    surviving = ts.data["time"].to_numpy()
    assert np.all(np.diff(surviving) > 0), "surviving rows must keep temporal order"

    # The IQR fence is outlier-proof here: planted extremes are always dropped.
    if method == "iqr":
        for row in outlier_rows:
            assert frame.loc[row, "time"] not in kept_times


@given(
    n=st.integers(min_value=10, max_value=40),
    seed=st.integers(0, 10_000),
    method=st.sampled_from(["iqr", "zscore"]),
    threshold=st.floats(1.0, 3.0),
    perm_seed=st.integers(0, 10_000),
    data=st.data(),
)
@P_FAST
def test_outlier_removal_is_row_order_independent(
    n: int, seed: int, method: str, threshold: float, perm_seed: int, data
) -> None:
    """Permuting rows before removal yields the identical surviving row set."""
    all_nan_rows = data.draw(
        st.lists(st.integers(0, n - 1), max_size=max(1, n // 3), unique=True)
    )
    outlier_row = next(r for r in range(n) if r not in set(all_nan_rows))

    frame = _contaminated_growth_frame(n, seed, all_nan_rows, [outlier_row])
    permuted = frame.iloc[np.random.default_rng(perm_seed).permutation(n)]
    ts_a = TimeSeriesData(frame.copy(), "time", PHENOTYPE_LIST)
    ts_b = TimeSeriesData(permuted.copy(), "time", PHENOTYPE_LIST)
    core_a, core_b = DataCore(ts_a), DataCore(ts_b)
    core_a._remove_outliers(ts_a, method=method, threshold=threshold)
    core_b._remove_outliers(ts_b, method=method, threshold=threshold)
    kept_a = set(ts_a.data["time"])
    kept_b = set(ts_b.data["time"])
    assert kept_a == kept_b
    values_a = ts_a.data.set_index("time")[PHENOTYPE_LIST].sort_index()
    values_b = ts_b.data.set_index("time")[PHENOTYPE_LIST].sort_index()
    np.testing.assert_allclose(values_a.to_numpy(), values_b.to_numpy(), rtol=0, atol=0)


# --------------------------------------------------------------------------
# AnalyticsEngine.survival_analysis — Kaplan-Meier
# --------------------------------------------------------------------------
@given(
    times=st.lists(
        st.floats(min_value=0.5, max_value=200.0, allow_nan=False),
        min_size=4,
        max_size=60,
    ),
    events=st.lists(st.sampled_from([0, 1]), min_size=4, max_size=60),
)
@P_FAST
def test_kaplan_meier_curve_is_non_increasing_and_bounded(times, events) -> None:
    """KM survival curve: non-increasing, inside [0, 1], CI-bracketed, median-consistent."""
    n = min(len(times), len(events))
    frame = pd.DataFrame({"time": times[:n], "event": events[:n]})
    result = AnalyticsEngine(frame).survival_analysis("time", "event")

    survival = result.survival_function
    unique_times = np.sort(frame["time"].unique())

    assert len(survival) == len(unique_times)
    assert np.all(np.diff(survival) <= 1e-12), "survival must be non-increasing"
    assert np.all(survival >= 0.0) and np.all(survival <= 1.0)

    lower = result.confidence_intervals["lower"]
    upper = result.confidence_intervals["upper"]
    assert np.all(lower <= survival + 1e-12)
    assert np.all(survival <= upper + 1e-12)
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)

    cum_hazard = result.cumulative_hazard
    assert np.all(cum_hazard >= 0.0) and np.all(np.diff(cum_hazard) >= -1e-12)

    below = np.where(survival <= 0.5)[0]
    if below.size:
        assert np.isclose(result.median_survival_time, unique_times[below[0]])
    else:
        assert np.isnan(result.median_survival_time)


# --------------------------------------------------------------------------
# JumpRope — fractional Brownian motion variance scaling
# --------------------------------------------------------------------------
@given(
    hurst=st.floats(0.15, 0.85),
    dt=st.floats(0.05, 2.0),
    diffusion=st.floats(0.1, 2.0),
    seed=st.integers(0, 10_000),
)
@P_MODEL
def test_fbm_increment_std_scales_as_dt_pow_hurst(
    hurst: float, dt: float, diffusion: float, seed: int
) -> None:
    """Empirical fBM increment sd matches diffusion * dt**H (the dt**(2H) variance law)."""
    model = FractionalBrownianMotion(
        ModelParameters(diffusion=diffusion),
        hurst=hurst,
        rng=np.random.default_rng(seed),
    )
    t = np.array([0.0, dt])
    n_draws = 2000
    increments = np.array(
        [model._generate_fbm_increments(1, t)[0] for _ in range(n_draws)]
    )

    empirical_sd = float(np.std(increments, ddof=1))
    expected_sd = diffusion * dt ** hurst
    # sd of the sample-sd estimator ~ 1/sqrt(2n) ~ 1.6%; 0.08 is ~5 sigma.
    assert empirical_sd == pytest.approx(expected_sd, rel=0.08)

    # Drift is zero, so increments are centered.
    assert abs(float(np.mean(increments))) < 5.0 * empirical_sd / np.sqrt(n_draws)


# --------------------------------------------------------------------------
# JumpRope — Ornstein-Uhlenbeck log-likelihood
# --------------------------------------------------------------------------
OU_TRUE = ModelParameters(
    equilibrium=0.0,
    reversion_speed=1.0,
    diffusion=0.5,
    jump_intensity=0.0,
)
OU_GRID = list(
    dict(
        equilibrium=eq,
        reversion_speed=rev,
        diffusion=diff,
        jump_intensity=ji,
        jump_mean=jm,
        jump_std=js,
    )
    for eq in (-1.0, 0.0, 1.0)
    for rev in (0.2, 1.0, 3.0)
    for diff in (0.1, 0.5, 2.0)
    for ji in (0.0, 0.3)
    for jm in (-0.2, 0.2)
    for js in (0.5, 1.0)
)
OU_DIFFUSION_GRID = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0)
OU_GROSSLY_WRONG = [
    ModelParameters(equilibrium=0.0, reversion_speed=1.0, diffusion=3.0),
    ModelParameters(equilibrium=0.0, reversion_speed=1.0, diffusion=0.05),
    ModelParameters(equilibrium=2.0, reversion_speed=1.0, diffusion=0.5),
    ModelParameters(equilibrium=-2.0, reversion_speed=1.0, diffusion=0.5),
    ModelParameters(equilibrium=0.0, reversion_speed=5.0, diffusion=0.5),
    ModelParameters(equilibrium=0.0, reversion_speed=1.0, diffusion=0.5, jump_intensity=2.0),
]


@given(seed=st.integers(0, 10_000))
@P_MODEL
def test_ou_log_likelihood_finite_over_box_and_maximized_near_truth(seed: int) -> None:
    """OU log-likelihood: finite everywhere on a parameter box, maximized at truth."""
    process = OrnsteinUhlenbeckJump(OU_TRUE, rng=np.random.default_rng(seed))
    dt = 0.1
    t = np.arange(0.0, 20.0 + dt, dt)
    path = process.simulate(x0=0.0, t=t, n_paths=1)[0]

    def ll(params: ModelParameters) -> float:
        return OrnsteinUhlenbeckJump(params).log_likelihood(path, dt)

    grid_lls = np.array([ll(ModelParameters(**kw)) for kw in OU_GRID])
    assert np.all(np.isfinite(grid_lls)), "log-likelihood must be finite over the box"

    true_ll = ll(OU_TRUE)
    for wrong in OU_GROSSLY_WRONG:
        assert true_ll > ll(wrong), f"truth must beat gross misspecification {wrong}"

    # Diffusion is the strongly identified coordinate: one-dimensional grid
    # over it (all other coordinates at truth) must peak at the true value.
    fixed = dict(
        equilibrium=OU_TRUE.equilibrium,
        reversion_speed=OU_TRUE.reversion_speed,
        jump_intensity=OU_TRUE.jump_intensity,
        jump_mean=0.0,
        jump_std=1.0,
    )
    diffusion_lls = [
        ll(ModelParameters(**fixed, diffusion=d)) for d in OU_DIFFUSION_GRID
    ]
    assert int(np.argmax(diffusion_lls)) == OU_DIFFUSION_GRID.index(OU_TRUE.diffusion)


# --------------------------------------------------------------------------
# JumpRope — geometric jump-diffusion log-likelihood
# --------------------------------------------------------------------------
GJD_TRUE = ModelParameters(
    drift=0.1,
    diffusion=0.2,
    jump_intensity=0.1,
    jump_mean=0.05,
    jump_std=0.1,
)
GJD_GRID = list(
    dict(
        drift=dr,
        diffusion=diff,
        jump_intensity=ji,
        jump_mean=jm,
        jump_std=js,
    )
    for dr in (-0.4, 0.1, 0.6)
    for diff in (0.05, 0.2, 0.8)
    for ji in (0.0, 0.1, 0.6)
    for jm in (0.0, 0.05)
    for js in (0.1,)
)
GJD_GROSSLY_WRONG = [
    ModelParameters(drift=0.6, diffusion=0.2, jump_intensity=0.1,
                    jump_mean=0.05, jump_std=0.1),
    ModelParameters(drift=0.1, diffusion=0.8, jump_intensity=0.1,
                    jump_mean=0.05, jump_std=0.1),
    ModelParameters(drift=0.1, diffusion=0.2, jump_intensity=0.9,
                    jump_mean=1.0, jump_std=0.1),
]


@given(seed=st.integers(0, 10_000))
@P_SLOW
def test_gjd_log_likelihood_finite_over_box_and_beats_misspecification(seed: int) -> None:
    """GJD log-likelihood: finite on a parameter box, larger at truth than at wrong params."""
    process = GeometricJumpDiffusion(GJD_TRUE, rng=np.random.default_rng(seed))
    dt = 0.1
    t = np.arange(0.0, 10.0 + dt, dt)
    path = process.simulate(x0=10.0, t=t, n_paths=1)[0]
    assert np.all(path > 0), "simulate() must keep the geometric process positive"

    def ll(params: ModelParameters) -> float:
        return GeometricJumpDiffusion(params).log_likelihood(path, dt)

    grid_lls = np.array([ll(ModelParameters(**kw)) for kw in GJD_GRID])
    assert np.all(np.isfinite(grid_lls)), "log-likelihood must be finite over the box"

    true_ll = ll(GJD_TRUE)
    for wrong in GJD_GROSSLY_WRONG:
        assert true_ll > ll(wrong), f"truth must beat gross misspecification {wrong}"


# --------------------------------------------------------------------------
# EvolutionSampler.compute_selection_gradient
# --------------------------------------------------------------------------
@given(
    n=st.integers(min_value=10, max_value=80),
    seed=st.integers(0, 10_000),
    slope=st.floats(-2.0, 2.0),
    data=st.data(),
)
@P_FAST
def test_selection_gradient_equals_standardized_linregress_slope(
    n: int, seed: int, slope: float, data
) -> None:
    """The gradient is exactly the standardized regression slope (= Pearson r)."""
    missing_pheno = data.draw(st.lists(st.integers(0, n - 1), max_size=n // 10, unique=True))
    missing_fit = data.draw(st.lists(st.integers(0, n - 1), max_size=n // 10, unique=True))

    frame = make_population_frame(
        n_individuals=n, n_time_points=1, trait_cols=("trait",), seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    pheno = rng.normal(0.0, 1.0, n)
    fitness = slope * pheno + rng.normal(0.0, 1.0, n)
    pheno[list(missing_pheno)] = np.nan
    fitness[list(missing_fit)] = np.nan
    frame["trait"] = pheno
    frame["fitness"] = fitness

    gradient = PopulationModel(frame).compute_selection_gradient("trait", "fitness")

    valid = np.isfinite(pheno) & np.isfinite(fitness)
    assert valid.sum() >= 2
    p, f = pheno[valid], fitness[valid]
    z = (p - p.mean()) / p.std()
    w = (f - f.mean()) / f.std()
    expected = stats.linregress(z, w).slope

    assert gradient == pytest.approx(expected, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize(
    "scenario",
    ["constant_phenotype", "constant_fitness", "missing_column", "single_pair"],
)
def test_selection_gradient_is_nan_for_degenerate_inputs(scenario: str) -> None:
    """Zero variance, absent columns, or a single pair yield NaN, not a number."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.0, 0.0],
            "trait": [5.0, 5.0, 5.0, 5.0],
            "fitness": rng.normal(size=4),
        }
    )
    model = PopulationModel(frame)
    if scenario == "constant_phenotype":
        assert np.isnan(model.compute_selection_gradient("trait", "fitness"))
    elif scenario == "constant_fitness":
        frame["fitness"] = 2.0
        assert np.isnan(model.compute_selection_gradient("trait", "fitness"))
    elif scenario == "missing_column":
        assert np.isnan(model.compute_selection_gradient("nope", "fitness"))
    else:
        single = frame.iloc[:1]
        assert np.isnan(PopulationModel(single).compute_selection_gradient("trait", "fitness"))


# --------------------------------------------------------------------------
# AnalyticsEngine robust statistics — Huber location, Sn scale
# --------------------------------------------------------------------------
@given(
    n=st.integers(min_value=40, max_value=140),
    contamination=st.floats(0.05, 0.25),
    outlier_magnitude=st.floats(5.0, 50.0),
    seed=st.integers(0, 10_000),
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # k/|u| inside np.where is guarded by the where
@P_FAST
def test_robust_estimators_stay_bounded_under_contaminated_normals(
    n: int, contamination: float, outlier_magnitude: float, seed: int
) -> None:
    """Huber location and Sn scale remain sane when up to 25% of a N(0,1) sample is blown up."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, n)
    n_outliers = max(1, int(contamination * n))
    values[:n_outliers] = outlier_magnitude

    result = AnalyticsEngine(pd.DataFrame({"v": values})).robust_statistical_analysis("v")

    huber = result["location_estimates"]["huber_estimator"]
    median = result["location_estimates"]["median"]
    sn = result["scale_estimates"]["sn_scale"]
    mad_normalized = result["scale_estimates"]["mad_normalized"]

    assert np.isfinite(huber) and np.isfinite(sn)
    # Contamination moves the arithmetic mean by contamination * magnitude
    # (>= 0.25), but the Huber M-estimator stays within one core sd of zero.
    assert abs(huber) <= 1.0
    assert abs(median) <= 1.0
    # Sn is scaled to be sigma-consistent on Gaussians; even 25% gross
    # contamination cannot inflate it past a factor of 3 or collapse it.
    assert 0.3 <= sn <= 3.0
    assert 0.3 <= mad_normalized <= 3.0


# --------------------------------------------------------------------------
# LaserPlane — order-statistic median confidence interval
# --------------------------------------------------------------------------
@given(
    n=st.integers(min_value=5, max_value=200),
    loc=st.floats(-10.0, 10.0),
    scale=st.floats(0.1, 5.0),
    confidence=st.floats(0.5, 0.999),
    n_missing=st.integers(0, 20),
    seed=st.integers(0, 10_000),
)
@P_FAST
def test_median_confidence_interval_brackets_sample_median(
    n: int, loc: float, scale: float, confidence: float, n_missing: int, seed: int
) -> None:
    """The distribution-free median CI contains the sample median of the clean data."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc, scale, n)
    n_missing = min(n_missing, max(0, n - 3))  # keep enough finite observations for CIs
    values[:n_missing] = np.nan

    intervals = MomentAnalyzer().compute_confidence_intervals(
        values, confidence_level=confidence
    )

    clean = values[~np.isnan(values)]
    median = float(np.median(clean))
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))

    med_lo, med_hi = intervals["median_ci"]
    assert med_lo <= median <= med_hi, "median CI must bracket the sample median"
    assert med_lo <= med_hi

    mean_lo, mean_hi = intervals["mean_ci"]
    assert mean_lo <= mean <= mean_hi

    std_lo, std_hi = intervals["std_ci"]
    assert std_lo <= std <= std_hi


@pytest.mark.parametrize("n", [1, 6])
def test_confidence_intervals_all_nan_input_returns_nan_triplets(n: int) -> None:
    """An all-NaN sample has no estimable parameters: NaN triplets, not zeros."""
    intervals = MomentAnalyzer().compute_confidence_intervals(np.full(n, np.nan))
    for key in ("mean_ci", "median_ci", "std_ci"):
        lo, hi = intervals[key]
        assert np.isnan(lo) and np.isnan(hi)
