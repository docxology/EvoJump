"""Methods-lane tests: BOCPD change-point detection and evolution_sampler
Lande response prediction / honest temporal Ne. Real computation only."""
import warnings

import numpy as np
import pandas as pd
import pytest

from evojump import analytics_engine, evolution_sampler


def _make_trajectory(with_jump: bool, n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = rng.normal(0, 0.3, n)
    if with_jump:
        y[n // 2:] += 3.0
    return pd.DataFrame({'time': t, 'signal': y})


class TestBOCPD:
    def test_bocpd_finds_real_changepoint(self):
        df = _make_trajectory(with_jump=True, seed=1)
        detector = analytics_engine.ChangePointDetector(df, 'time')
        cps = detector.detect_changes(method='bayesian')
        assert len(cps) > 0
        hit = [c for c in cps
               if c['variable'] == 'signal'
               and abs(c['time_index'] - 30) <= 3]
        assert hit, f"expected a change point near index 30, got {cps}"
        for c in cps:
            assert c['method'] == 'bocpd'
            assert 0.0 <= c['changepoint_probability'] <= 1.0

    def test_bocpd_quiet_series_few_changepoints(self):
        df = _make_trajectory(with_jump=False, seed=2)
        detector = analytics_engine.ChangePointDetector(df, 'time')
        cps = detector.detect_changes(method='bayesian', threshold=0.5)
        assert len(cps) <= 2

    def test_cusum_path_untouched(self):
        df = _make_trajectory(with_jump=True, seed=3)
        detector = analytics_engine.ChangePointDetector(df, 'time')
        cps = detector.detect_changes(method='statistical')
        for c in cps:
            assert c['method'] == 'statistical'

    def test_bocpd_short_series_returns_empty(self):
        df = pd.DataFrame({'time': np.arange(5.0), 'signal': [1, 2, 1, 2, 1]})
        detector = analytics_engine.ChangePointDetector(df, 'time')
        assert detector.detect_changes(method='bayesian') == []


class TestTemporalNeHonesty:
    def test_temporal_ne_without_markers_warns_nan(self):
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10.0, 12.0, 14.0, 16.0, 18.0],
        })
        model = evolution_sampler.PopulationModel(data, 'time')
        with pytest.warns(UserWarning, match="allele-frequency"):
            ne = model.estimate_effective_population_size(method='temporal')
        assert np.isnan(ne)

    def test_temporal_ne_with_marker_frequencies(self):
        rng = np.random.default_rng(4)
        n_loci = 10
        rows = []
        for t, shift in ((1, 0.0), (2, 0.0)):
            freqs = np.clip(rng.uniform(0.2, 0.8, n_loci) + shift, 0.01, 0.99)
            row = {'time': t}
            row.update({f'freq_{i}': p for i, p in enumerate(freqs)})
            rows.append(row)
        data = pd.DataFrame(rows)
        model = evolution_sampler.PopulationModel(data, 'time')
        ne = model.estimate_effective_population_size(method='temporal')
        assert np.isfinite(ne) and ne > 0

    def test_unsupported_method_raises(self):
        data = pd.DataFrame({'time': [1, 2], 'phenotype1': [1.0, 2.0]})
        model = evolution_sampler.PopulationModel(data, 'time')
        with pytest.raises(ValueError):
            model.estimate_effective_population_size(method='bogus')


class TestLandeResponse:
    def test_selection_differential_and_response(self):
        data = pd.DataFrame({
            'time': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'phenotype1': [10.0, 10.5, 9.5, 10.2, 9.8,
                           12.0, 12.5, 11.5, 12.2, 11.8],
        })
        model = evolution_sampler.PopulationModel(data, 'time')
        s = model.compute_selection_differential('phenotype1')
        assert s == pytest.approx(2.0, abs=1e-9)
        r = model.predict_phenotypic_response('phenotype1', h2=0.5)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_response_requires_h2_in_range(self):
        data = pd.DataFrame({'time': [1, 2], 'phenotype1': [1.0, 2.0]})
        model = evolution_sampler.PopulationModel(data, 'time')
        with pytest.raises(ValueError):
            model.predict_phenotypic_response('phenotype1', h2=1.5)
        with pytest.raises(ValueError):
            model.predict_phenotypic_response('phenotype1', h2=-0.1)

    def test_response_nan_without_time_series(self):
        data = pd.DataFrame({'phenotype1': [1.0, 2.0, 3.0]})
        model = evolution_sampler.PopulationModel(data)
        assert np.isnan(model.predict_phenotypic_response('phenotype1', h2=0.5))
