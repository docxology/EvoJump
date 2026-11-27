"""Replay the finite stochastic and deterministic trajectory controls."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evojump.jumprope import ModelParameters, OrnsteinUhlenbeckJump
from evojump.laserplane import MomentAnalyzer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    times = np.linspace(0.0, 2.0, 21)
    initial = 2.0
    equilibrium = 0.5
    reversion = 0.8
    deterministic = ModelParameters(diffusion=0, equilibrium=equilibrium, reversion_speed=reversion)
    control = OrnsteinUhlenbeckJump(deterministic, rng=np.random.default_rng(27)).simulate(initial, times)[0]
    expected = equilibrium + (initial - equilibrium) * (1 - reversion * (times[1] - times[0])) ** np.arange(len(times))
    tolerance = 1e-12
    if not np.allclose(control, expected, atol=tolerance, rtol=0):
        raise ValueError("The deterministic solver differs from the discrete recurrence.")

    settings = ModelParameters(
        diffusion=0.15, jump_intensity=0.5, jump_mean=0.1, jump_std=0.05,
        equilibrium=equilibrium, reversion_speed=reversion,
    )
    paths, repeat = (
        OrnsteinUhlenbeckJump(settings, rng=np.random.default_rng(27)).simulate(initial, times, n_paths=8)
        for _ in range(2)
    )
    if not np.isfinite(paths).all() or not np.array_equal(paths, repeat):
        raise ValueError("Seeded trajectory arrays differ or contain nonfinite values.")
    terminal = paths[:, -1]
    summary = MomentAnalyzer().compute_moments(terminal)
    if summary["variance"] <= 0 or not np.allclose(
        (summary["mean"], summary["variance"]),
        (terminal.mean(), terminal.var(ddof=1)), atol=tolerance, rtol=0,
    ):
        raise ValueError("The moment analyzer did not reproduce the terminal summary.")

    output = ROOT / "output"
    data = output / "data"
    figures = output / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    receipt = {
        "time_point_count": int(times.size), "path_count": len(paths), "seed": 27,
        "dependency_versions": {name: version(name) for name in ("numpy", "scipy", "evojump")},
        "times": times.tolist(), "paths": paths.tolist(), "settings": asdict(settings),
        "tolerance": tolerance, "maximum_control_error": float(np.max(abs(control - expected))),
        "terminal_mean": float(summary["mean"]), "terminal_sample_variance": float(summary["variance"]),
    }
    (data / "trajectory-evidence.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lower, median, upper = np.quantile(paths, (0.25, 0.5, 0.75), axis=0)
    plot, axes = plt.subplots(figsize=(7.6, 4), layout="constrained")
    axes.plot(times, control, color="#2b3b56", label="Deterministic control")
    axes.plot(times, median, color="#16695e", label="Seeded median")
    axes.fill_between(times, lower, upper, color="#57a79a", alpha=0.35, label="Seeded quartiles")
    axes.set_xlabel("Simulation time")
    axes.set_ylabel("State")
    axes.legend()
    plot.savefig(figures / "trajectory-cross-sections.png", dpi=170)
    plt.close(plot)
    registry = {"figures": [{
        "label": "fig:trajectory-cross-sections", "filename": "trajectory-cross-sections.png",
        "caption": "Deterministic and seeded mean-reverting trajectories.",
        "alt_text": "The deterministic trace descends. The seeded paths show dispersion around their median.",
        "generated_by": "python -m docs.reproduce_trajectories",
    }]}
    (figures / "figure_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print("Verified 21 time points, eight seeded paths and the terminal sample moments.")


if __name__ == "__main__":
    main()
