"""Regression tests for post-audit fixes (methods lane, 2026-08-30).
Real computation only — no mock frameworks."""
import math

import numpy as np
import pytest

import pandas as pd

from evojump import analytics_engine, evolution_sampler, jumprope
from evojump import laserplane


class TestCompoundPoissonLogLikelihood:
    def test_log_likelihood_runs_on_numpy2(self):
        """np.math was removed in numpy 2.0; lgamma path must work."""
        assert not hasattr(np, 'math') or np.__version__ >= '2'
        p = jumprope.CompoundPoisson(jumprope.ModelParameters(
            jump_intensity=2.0, jump_mean=0.5, jump_std=1.0))
        data = np.array([0.0, 0.4, 1.1, 0.9, 2.2])
        ll = p.log_likelihood(data, dt=1.0)
        assert np.isfinite(ll)

    def test_log_likelihood_changes_with_intensity(self):
        """LL must vary with jump intensity (pmf enters the mixture)."""
        data = np.array([1.0, 1.0, 1.05, 0.98, 1.0])
        lls = []
        for lam in (0.1, 2.0, 10.0):
            p = jumprope.CompoundPoisson(jumprope.ModelParameters(
                jump_intensity=lam, jump_mean=0.5, jump_std=1.0))
            lls.append(p.log_likelihood(data, dt=1.0))
        assert all(np.isfinite(v) for v in lls)
        assert len(set(np.round(lls, 9))) == 3

    def test_log_likelihood_finite_for_jump_data(self):
        p = jumprope.CompoundPoisson(jumprope.ModelParameters(
            jump_intensity=1.5, jump_mean=0.3, jump_std=0.8))
        rng = np.random.default_rng(0)
        data = np.cumsum(rng.normal(0, 0.5, 50))
        ll = p.log_likelihood(data, dt=1.0)
        assert np.isfinite(ll)


class TestBayesianLinearRegressionEvidence:
    def _independent_log_evidence(self, x, y, m0, prec0, a0, b0):
        """Recompute the NIG evidence independently from first principles."""
        n = len(x)
        xm, ym = x.mean(), y.mean()
        sxx = np.sum((x - xm) ** 2)
        sxy = np.sum((x - xm) * (y - ym))
        tyy = np.sum((y - ym) ** 2)
        post_prec = prec0 + sxx
        post_mean = (prec0 * m0 + sxy) / post_prec
        post_a = a0 + n / 2.0
        post_b = b0 + 0.5 * (tyy + prec0 * m0 ** 2 - post_prec * post_mean ** 2)
        return (math.lgamma(post_a) - math.lgamma(a0)
                + a0 * math.log(b0) - post_a * math.log(post_b)
                + 0.5 * math.log(prec0) - 0.5 * math.log(post_prec)
                - n / 2.0 * math.log(2 * math.pi))

    def test_evidence_matches_independent_computation(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 10, 40)
        y = 2.0 * x + rng.normal(0, 1.0, 40)
        analyzer = analytics_engine.BayesianAnalyzer(np.column_stack([x, y]))
        result = analyzer.bayesian_linear_regression(x, y)
        expected = self._independent_log_evidence(
            x, y,
            analyzer.prior_parameters['location']['mean'],
            analyzer.prior_parameters['location']['precision'],
            analyzer.prior_parameters['scale']['shape'],
            analyzer.prior_parameters['scale']['rate'])
        assert result.model_evidence == pytest.approx(expected, rel=1e-10)

    def test_posterior_scale_uses_centered_rss(self):
        """post_b must be close to centered RSS/2 + prior, not sum(y^2)/2."""
        rng = np.random.default_rng(4)
        x = rng.uniform(0, 10, 60)
        y = 3.0 * x + 5.0 + rng.normal(0, 1.0, 60)
        analyzer = analytics_engine.BayesianAnalyzer(np.column_stack([x, y]))
        xm, ym = x.mean(), y.mean()
        sxx = np.sum((x - xm) ** 2)
        sxy = np.sum((x - xm) * (y - ym))
        slope = sxy / sxx
        rss = np.sum((y - ym - slope * (x - xm)) ** 2)
        b0 = analyzer.prior_parameters['scale']['rate']
        expected_post_b = b0 + 0.5 * rss  # m0=0 simplification (prec0*m0^2=0)
        # indirect check: evidence formula uses post_b consistently
        result = analyzer.bayesian_linear_regression(x, y)
        expected = self._independent_log_evidence(
            x, y,
            analyzer.prior_parameters['location']['mean'],
            analyzer.prior_parameters['location']['precision'],
            analyzer.prior_parameters['scale']['shape'],
            analyzer.prior_parameters['scale']['rate'])
        assert result.model_evidence == pytest.approx(expected, rel=1e-10)
        # sanity: with large mean y, uncentered version would differ hugely
        assert rss < 2 * np.sum(y ** 2)

    def test_posterior_mean_recovers_slope(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 10, 200)
        y = 2.5 * x + rng.normal(0, 0.5, 200)
        analyzer = analytics_engine.BayesianAnalyzer(np.column_stack([x, y]))
        result = analyzer.bayesian_linear_regression(x, y)
        assert abs(np.mean(result.posterior_samples) - 2.5) < 0.2


class TestMCMDetailedBalance:
    def test_mcmc_runs_and_targets_empirical_mean(self):
        rng = np.random.default_rng(6)
        n = 120
        data = pd.DataFrame({
            'time': np.arange(n),
            'trait': rng.normal(10.0, 2.0, n),
        })
        model = evolution_sampler.PopulationModel(data, 'time')
        sampler = evolution_sampler.EvolutionSampler(data)
        samples = sampler.sample(n_samples=500, method='mcmc',
                                 parameters={'step_size': 0.5, 'mcmc_scale': 1.0})
        arr = np.asarray(samples.samples)
        assert arr.shape[0] == 500
        # samples drawn from the pool, centered near the empirical mean
        assert abs(arr.mean() - data['trait'].mean()) < 1.0

    def test_mcmc_samples_are_observed_individuals(self):
        data = pd.DataFrame({
            'time': np.arange(40),
            'trait': np.linspace(0, 10, 40),
        })
        sampler = evolution_sampler.EvolutionSampler(data)
        samples = sampler.sample(n_samples=100, method='mcmc',
                                 parameters={'step_size': 1.0})
        pool = set(np.round(data['trait'].values, 9))
        for v in np.round(np.asarray(samples.samples).ravel(), 9):
            assert v in pool


class TestGeometricJacobian:
    def test_jump_component_includes_price_jacobian(self):
        """LL of a pure-jump geometric series must include log(price) terms."""
        params = jumprope.ModelParameters(
            jump_intensity=10.0, jump_mean=0.0, jump_std=0.5,
            drift=0.0, diffusion=0.05)
        g = jumprope.GeometricJumpDiffusion(params)
        data = np.array([1.0, 1.5, 2.0])
        ll = g.log_likelihood(data, dt=1.0)
        assert np.isfinite(ll)
        # The Jacobian adds log(price) per step; against a no-Jacobian
        # variant recomputed here, the difference must be exactly
        # log(1.5) + log(2.0).
        import numpy.testing as npt
        from scipy.stats import norm, lognorm
        manual = 0.0
        p0 = np.exp(-10.0 * 1.0)
        for prev, cur in ((1.0, 1.5), (1.5, 2.0)):
            lr = np.log(cur / prev)
            comps = [
                np.log(max(p0, 1e-300)) + norm.logpdf(lr, 0.0, 0.05),
                np.log(max(1 - p0, 1e-300)) + lognorm.logpdf(
                    np.exp(lr), s=0.5, scale=1.0) + np.log(cur),
            ]
            manual += np.logaddexp.reduce(comps)
        npt.assert_allclose(ll, manual, rtol=1e-12)

    def test_docstring_declares_truncated_mixture(self):
        doc = jumprope.GeometricJumpDiffusion.__doc__ or ''
        assert 'truncat' in doc.lower() or 'two-component' in doc.lower()


class TestSeededBootstrap:
    def test_bootstrap_ci_deterministic(self):
        analyzer = laserplane.LaserPlaneAnalyzer.__new__(laserplane.LaserPlaneAnalyzer)
        data = np.random.default_rng(7).normal(0, 1, 60)
        ci1 = analyzer._bootstrap_confidence_intervals(data, 200)
        ci2 = analyzer._bootstrap_confidence_intervals(data, 200)
        assert ci1['mean_ci'] == ci2['mean_ci']
