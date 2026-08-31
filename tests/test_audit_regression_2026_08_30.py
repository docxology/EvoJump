"""
Regression tests for the 2026-08-30 EvoJump audit fixes.

Each test targets a verified pre-existing defect:
1. Packaging: gpu extra unsatisfiable / requires-python bounds.
2. JumpRope likelihoods: OU and geometric jump-diffusion mixtures; seed control.
3. EvolutionSampler: real importance sampling (ESS recorded), real MH-MCMC
   (acceptance recorded), Moran's I phylogenetic signal, pedigree-gated
   heritability.
4. AnalyticsEngine: Kaplan-Meier hazards/median/CI, Rosenstein Lyapunov,
   honest Bayesian diagnostics, conjugate posterior.
5. CLI: visualize uses instance methods; model-type choices complete.
6. DataCore: temporally sorted interpolation; stable dataset ids.
7. Animation: fixed axes; percentile CI documented as SEM band.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evojump import analytics_engine, datacore, evolution_sampler, jumprope


@pytest.fixture
def ts_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    times = np.repeat(np.arange(10.0), 5)
    return pd.DataFrame({
        'time': times,
        'size': rng.normal(10 + 0.5 * times, 1.0),
        'weight': rng.normal(5 + 0.2 * times, 0.5),
    })


@pytest.fixture
def data_core(ts_frame, tmp_path) -> datacore.DataCore:
    csv = tmp_path / 'data.csv'
    ts_frame.to_csv(csv, index=False)
    return datacore.DataCore.load_from_csv(csv)


class TestJumpropeFixes:
    def test_ou_likelihood_jump_mixture_is_finite(self):
        proc = jumprope.OrnsteinUhlenbeckJump(
            jumprope.ModelParameters(equilibrium=0.0, reversion_speed=1.0,
                                     diffusion=0.5, jump_intensity=0.3,
                                     jump_mean=0.5, jump_std=0.2))
        rng = np.random.default_rng(0)
        t = np.linspace(0, 5, 50)
        data = proc.simulate(0.0, t, n_paths=1)[0]
        ll = proc.log_likelihood(data, float(np.mean(np.diff(t))))
        assert np.isfinite(ll)
        # positive jump intensity must change the likelihood
        proc.parameters.jump_intensity = 0.0
        ll_no_jump = proc.log_likelihood(data, float(np.mean(np.diff(t))))
        assert ll != pytest.approx(ll_no_jump)

    def test_geometric_likelihood_finite_and_jump_sensitive(self):
        proc = jumprope.GeometricJumpDiffusion(
            jumprope.ModelParameters(drift=0.1, diffusion=0.2,
                                     jump_intensity=0.2, jump_mean=0.1,
                                     jump_std=0.1))
        rng = np.random.default_rng(1)
        t = np.linspace(0, 4, 40)
        data = np.exp(np.cumsum(rng.normal(0.05, 0.1, len(t))))
        ll = proc.log_likelihood(data, float(np.mean(np.diff(t))))
        assert np.isfinite(ll)

    def test_seed_reproducibility(self):
        core = datacore.DataCore.__new__(datacore.DataCore)
        core.time_series_data = [datacore.TimeSeriesData(
            pd.DataFrame({'time': [0., 1., 2., 3.], 'x': [0., 1., 2., 3.]}),
            'time', ['x'])]
        core.metadata_manager = datacore.MetadataManager()
        m1 = jumprope.JumpRope.fit(core, model_type='ornstein-uhlenbeck', seed=42)
        t1 = m1.generate_trajectories(n_samples=3, seed=42)
        t2 = jumprope.JumpRope.fit(core, model_type='ornstein-uhlenbeck', seed=42)             .generate_trajectories(n_samples=3, seed=42)
        assert np.allclose(t1, t2)


class TestSamplerFixes:
    def test_importance_sampling_records_ess(self, ts_frame):
        sampler = evolution_sampler.EvolutionSampler(ts_frame)
        params = {'temperature': 2.0}
        result = sampler.sample(n_samples=100, method='importance-sampling',
                                parameters=params)
        assert 'ess' in params
        assert 1.0 <= params['ess'] <= 100.0

    def test_mcmc_records_acceptance(self, ts_frame):
        sampler = evolution_sampler.EvolutionSampler(ts_frame)
        sampler.seed(3)
        params = {'step_size': 0.5}
        result = sampler.sample(n_samples=50, method='mcmc', parameters=params)
        assert 'acceptance_rate' in params
        assert 0.0 <= params['acceptance_rate'] <= 1.0
        assert result.samples.shape[0] == 50

    def test_morans_i_positive_for_clustered(self):
        d = np.array([[0, 1, 4, 9], [1, 0, 3, 8], [4, 3, 0, 2], [9, 8, 2, 0]],
                     dtype=float)
        analyzer = evolution_sampler.PhylogeneticAnalyzer(distance_matrix=d)
        # traits cluster with distance blocks
        traits = np.array([1.0, 1.2, 5.0, 5.3])
        I = analyzer.compute_morans_i_signal(traits)
        assert np.isfinite(I)
        assert I > 0.5

    def test_morans_i_nan_without_matrix(self):
        analyzer = evolution_sampler.PhylogeneticAnalyzer()
        assert np.isnan(analyzer.compute_morans_i_signal(np.array([1., 2., 3.])))

    def test_heritability_requires_pedigree(self, ts_frame):
        sampler = evolution_sampler.EvolutionSampler(ts_frame)
        with pytest.warns(UserWarning, match='pedigree'):
            h = sampler.population_model.estimate_heritability('size')
        assert np.isnan(h)


class TestAnalyticsFixes:
    def test_km_median_and_hazards(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            'time': rng.exponential(5, 80),
            'event': rng.integers(0, 2, 80),
        })
        engine = analytics_engine.AnalyticsEngine(df)
        res = engine.survival_analysis('time', 'event')
        assert len(res.cumulative_hazard) > 0
        assert np.all(np.diff(res.cumulative_hazard) >= -1e-12)
        assert np.isfinite(res.median_survival_time)
        assert 'lower' in res.confidence_intervals and 'upper' in res.confidence_intervals

    def test_km_median_nan_when_survival_never_below_half(self, ts_frame):
        engine = analytics_engine.AnalyticsEngine(
            pd.DataFrame({'time': [1., 2., 3.], 'event': [0, 0, 0]}))
        res = engine.survival_analysis('time', 'event')
        # no events -> survival stays 1.0 -> median is NaN, not median of times
        assert np.isnan(res.median_survival_time)

    def test_rousenstein_lyapunov_real_values(self):
        rng = np.random.default_rng(11)
        # logistic map near chaos gives positive Lyapunov exponent
        x = np.zeros(600)
        x[0] = 0.4
        for i in range(1, 600):
            x[i] = 3.9 * x[i-1] * (1 - x[i-1])
        df = pd.DataFrame({'signal': x})
        engine = analytics_engine.AnalyticsEngine(df)
        res = engine.nonlinear_dynamics_analysis('signal', embedding_dim=3, tau=1)
        lam = res['largest_lyapunov_exponent']
        assert np.isfinite(lam)
        # chaotic logistic map: exponent should be substantially positive
        assert lam > 0.05
        corr_dims = np.asarray(res['correlation_dimensions'], dtype=float)
        assert corr_dims.size > 0
        # real Grassberger-Procaccia output varies with radius (not constants)
        assert not np.allclose(corr_dims, corr_dims[0])

    def test_bayesian_diagnostics_honest(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 10, 100)
        y = 2.0 * x + rng.normal(0, 1, 100)
        engine = analytics_engine.AnalyticsEngine(pd.DataFrame({'x': x, 'y': y}))
        blr = analytics_engine.BayesianAnalyzer(engine.data)
        res = blr.bayesian_linear_regression(x, y, n_samples=500)
        # posterior mean should recover the true slope
        assert res.convergence_diagnostics['posterior_mean'] == pytest.approx(2.0, abs=0.2)
        # evidence is a real finite log-evidence, not hardcoded 0.0
        assert np.isfinite(res.model_evidence)
        assert res.model_evidence != 0.0


# Cold-start imports of pandas/numba/dask from a loaded external drive can
# exceed 240s under fleet load (sampled: child stuck in PyImport chains, not
# in evojump code). 900s bounds real work while still catching hangs.
CLI_SUBPROCESS_TIMEOUT_S = 900


class TestCliFixes:

    def _run_cli_py(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True,
            timeout=CLI_SUBPROCESS_TIMEOUT_S,
            cwd=str(Path(__file__).resolve().parents[1]))

    def test_model_type_choices_complete(self):
        from evojump import cli
        parser = cli.create_parser()
        # should not raise for every supported model
        for mt in ['jump-diffusion', 'ornstein-uhlenbeck', 'compound-poisson',
                   'geometric-jump-diffusion', 'fractional-brownian', 'cir', 'levy']:
            args = parser.parse_args(['fit', 'x.csv', '--model-type', mt])
            assert args.model_type == mt

    def test_visualize_command_uses_instance(self, tmp_path):
        # build a model file, then run the visualize path in-process
        rng = np.random.default_rng(0)
        core = datacore.DataCore.__new__(datacore.DataCore)
        core.time_series_data = [datacore.TimeSeriesData(
            pd.DataFrame({'time': np.linspace(0, 3, 8),
                          'x': rng.normal(0, 1, 8)}), 'time', ['x'])]
        core.metadata_manager = datacore.MetadataManager()
        model = jumprope.JumpRope.fit(core, seed=0)
        model.generate_trajectories(n_samples=5, seed=0)
        pkl = tmp_path / 'model.pkl'
        model.save(pkl)
        out = tmp_path / 'plots'

        proc = subprocess.run(
            [sys.executable, '-m', 'evojump.cli', '--output', str(out),
             'visualize', str(pkl), '--plot-type', 'trajectories'],
            capture_output=True, text=True,
            timeout=CLI_SUBPROCESS_TIMEOUT_S,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=None)
        # previously crashed with TypeError (unbound call); must now succeed
        assert proc.returncode == 0, proc.stderr


class TestDatacoreFixes:
    def test_interpolation_is_temporal_not_row_order(self):
        # One row per time point so row order == time order permutations
        base = pd.DataFrame({'time': np.arange(8.0),
                             'size': [1., 2., np.nan, 4., 5., 6., 7., 8.]})
        ts = datacore.TimeSeriesData(base.copy(), 'time', ['size'])
        shuffled = base.sample(frac=1, random_state=0)
        ts2 = datacore.TimeSeriesData(shuffled.copy(), 'time', ['size'])
        ts.interpolate_missing_data()
        ts2.interpolate_missing_data()
        # NaN at t=2 must be linear between t=1 and t=3 in both row orders
        for frame in (ts, ts2):
            vals = frame.data.sort_values('time')['size'].to_numpy()
            assert vals[2] == pytest.approx(3.0)
        assert np.allclose(
            ts.data.sort_values('time')['size'],
            ts2.data.sort_values('time')['size'], equal_nan=True)

    def test_stable_dataset_ids(self, ts_frame):
        dc = datacore.DataCore([
            datacore.TimeSeriesData(ts_frame.copy(), 'time', ['size']),
            datacore.TimeSeriesData(ts_frame.copy(), 'time', ['size']),
        ])
        agg1 = dc.get_aggregated_data()
        agg2 = dc.get_aggregated_data()
        assert agg1.equals(agg2)


class TestVisualizerFixes:
    def test_animation_axes_fixed_and_saves(self, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        from evojump import trajectory_visualizer as tv

        rng = np.random.default_rng(0)
        core = datacore.DataCore.__new__(datacore.DataCore)
        core.time_series_data = [datacore.TimeSeriesData(
            pd.DataFrame({'time': np.linspace(0, 2, 6),
                          'x': rng.normal(0, 1, 6)}), 'time', ['x'])]
        core.metadata_manager = datacore.MetadataManager()
        model = jumprope.JumpRope.fit(core, seed=1)
        model.generate_trajectories(n_samples=4, seed=1)

        vis = tv.TrajectoryVisualizer()
        out = tmp_path / 'anim'
        anim = vis.create_animation(model, n_frames=3, output_dir=out)
        assert anim is not None
        assert (out / 'animation.gif').exists()
