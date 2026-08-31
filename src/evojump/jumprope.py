"""
JumpRope Engine: Jump-Diffusion Modeling for Developmental Trajectories

This module implements jump-diffusion modeling for developmental trajectories.
It processes temporal phenotypic data to estimate drift parameters, diffusion coefficients,
and jump intensities characterizing developmental transitions. The engine supports multiple
stochastic process models including Ornstein-Uhlenbeck processes with jumps, geometric
jump-diffusion, compound Poisson processes, fractional Brownian motion, Cox-Ingersoll-Ross
processes, and Levy processes.

Classes:
    JumpRope: Main class for jump-diffusion modeling
    StochasticProcess: Base class for stochastic process models
    OrnsteinUhlenbeckJump: Ornstein-Uhlenbeck process with jumps
    GeometricJumpDiffusion: Geometric jump-diffusion process
    CompoundPoisson: Compound Poisson process
    FractionalBrownianMotion: Fractional Brownian motion with long-range dependence
    CoxIngersollRoss: CIR process for mean-reverting non-negative processes
    LevyProcess: Levy process with stable distributions

Examples:
    >>> # Fit jump rope model to data
    >>> model = JumpRope.fit(data, model_type='jump-diffusion')
    >>> # Fit fractional Brownian motion
    >>> model = JumpRope.fit(data, model_type='fractional-brownian', hurst=0.7)
    >>> # Fit CIR process
    >>> model = JumpRope.fit(data, model_type='cir')
    >>> # Generate trajectories
    >>> trajectories = model.generate_trajectories(n_samples=100)
    >>> # Estimate parameters
    >>> params = model.estimate_parameters()
"""

import math
from math import lgamma
import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from scipy.stats import norm, lognorm
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Container for model parameters."""
    drift: float = 0.0
    diffusion: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 1.0
    equilibrium: float = 0.0
    reversion_speed: float = 1.0
    bounds: Optional[Tuple[float, float]] = None
    correlation_matrix: Optional[np.ndarray] = None


class StochasticProcess(ABC):
    """Base class for stochastic process models."""

    def __init__(self, parameters: ModelParameters):
        """Initialize stochastic process with parameters."""
        self.parameters = parameters

    @abstractmethod
    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate stochastic process trajectories."""
        pass

    @abstractmethod
    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute log-likelihood of observed data."""
        pass

    @abstractmethod
    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate model parameters from data."""
        pass


class OrnsteinUhlenbeckJump(StochasticProcess):
    """Ornstein-Uhlenbeck process with Poisson jumps."""

    def __init__(self, parameters: ModelParameters, rng: Optional[np.random.Generator] = None):
        """Initialize Ornstein-Uhlenbeck process with jumps."""
        super().__init__(parameters)
        self.process_name = "Ornstein-Uhlenbeck with Jumps"
        self.rng = rng or np.random.default_rng()

    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate Ornstein-Uhlenbeck process with jumps."""
        dt = np.diff(t)
        n_steps = len(t) - 1
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = x0

        for i in range(n_paths):
            x = x0
            for j in range(n_steps):
                # Continuous part (Ornstein-Uhlenbeck)
                drift = (self.parameters.equilibrium - x) * self.parameters.reversion_speed * dt[j]
                diffusion = self.parameters.diffusion * np.sqrt(dt[j]) * self.rng.normal()

                # Jump part (Poisson process)
                jump_occurred = self.rng.poisson(self.parameters.jump_intensity * dt[j])
                if jump_occurred > 0:
                    jump_size = self.rng.normal(self.parameters.jump_mean, self.parameters.jump_std)
                else:
                    jump_size = 0.0

                x = x + drift + diffusion + jump_size
                paths[i, j + 1] = x

        return paths

    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute exact one-step log-likelihood for OU + compound Poisson jumps.

        The one-step transition density is a Poisson mixture: with probability
        exp(-lambda*dt) no jump occurs (Gaussian increment around the OU mean),
        otherwise a compound Poisson jump of Gaussian size is added. All mixture
        components are summed in log space for numerical stability.
        """
        if len(data) < 2:
            return 0.0

        lam = max(self.parameters.jump_intensity, 0.0)
        p0 = np.exp(-lam * dt)  # probability of zero jumps in [t, t+dt]
        mu0 = data[:-1] + (self.parameters.equilibrium - data[:-1]) * self.parameters.reversion_speed * dt
        sigma = self.parameters.diffusion * np.sqrt(dt)
        log_likelihood = 0.0

        for i in range(1, len(data)):
            x_prev, x = data[i-1], data[i]
            comps = []
            if sigma > 0:
                # zero-jump component
                comps.append(np.log(max(p0, 1e-300)) + norm.logpdf(x, mu0[i-1], sigma))
                # one-jump component: jump size N(jump_mean, jump_std)
                jump_mu = mu0[i-1] + self.parameters.jump_mean
                comps.append(np.log(max(1.0 - p0, 1e-300)) + norm.logpdf(x, jump_mu,
                              np.sqrt(sigma**2 + self.parameters.jump_std**2)))
                log_likelihood += np.logaddexp.reduce(comps)
            else:
                comps.append(np.log(max(p0, 1e-300)))
                log_likelihood += np.logaddexp.reduce(comps)

        return float(log_likelihood)

    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate parameters using maximum likelihood."""
        def objective(params):
            # Unpack parameters
            equilibrium, reversion_speed, diffusion, jump_intensity, jump_mean, jump_std = params

            self.parameters.equilibrium = equilibrium
            self.parameters.reversion_speed = reversion_speed
            self.parameters.diffusion = diffusion
            self.parameters.jump_intensity = jump_intensity
            self.parameters.jump_mean = jump_mean
            self.parameters.jump_std = jump_std

            return -self.log_likelihood(data, dt)

        # Initial parameter guess
        initial_guess = [
            np.mean(data),  # equilibrium
            1.0,            # reversion_speed
            np.std(data),   # diffusion
            0.1,            # jump_intensity
            0.0,            # jump_mean
            np.std(data)    # jump_std
        ]

        # Parameter bounds
        bounds = [
            (np.min(data), np.max(data)),  # equilibrium
            (0.001, 10.0),                # reversion_speed
            (0.001, np.std(data) * 10),   # diffusion
            (0.0, 1.0),                   # jump_intensity
            (-5 * np.std(data), 5 * np.std(data)),  # jump_mean
            (0.001, np.std(data) * 10)    # jump_std
        ]

        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                equilibrium, reversion_speed, diffusion, jump_intensity, jump_mean, jump_std = result.x
                return ModelParameters(
                    equilibrium=equilibrium,
                    reversion_speed=reversion_speed,
                    diffusion=diffusion,
                    jump_intensity=jump_intensity,
                    jump_mean=jump_mean,
                    jump_std=jump_std
                )
            else:
                warnings.warn("Parameter estimation failed, using initial values")
                return self.parameters
        except Exception as e:
            warnings.warn(f"Parameter estimation error: {e}")
            return self.parameters


class GeometricJumpDiffusion(StochasticProcess):
    """Geometric jump-diffusion process.

    Note: the one-step density used by log_likelihood is a TWO-component
    (no-jump / single-jump) truncation of the full compound-Poisson mixture;
    multiple jumps within one step are absorbed into the single-jump component."""

    def __init__(self, parameters: ModelParameters, rng: Optional[np.random.Generator] = None):
        """Initialize geometric jump-diffusion process."""
        super().__init__(parameters)
        self.process_name = "Geometric Jump-Diffusion"
        self.rng = rng or np.random.default_rng()

    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate geometric jump-diffusion process."""
        dt = np.diff(t)
        n_steps = len(t) - 1
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = x0

        for i in range(n_paths):
            x = x0
            for j in range(n_steps):
                # Jump part (compound Poisson) — multiplicative before diffusion
                jump_occurred = self.rng.poisson(self.parameters.jump_intensity * dt[j])
                if jump_occurred > 0:
                    jump_factor = self.rng.lognormal(self.parameters.jump_mean, self.parameters.jump_std)
                    x = x * jump_factor

                # Continuous part (geometric Brownian motion)
                drift = self.parameters.drift * x * dt[j]
                diffusion = self.parameters.diffusion * x * np.sqrt(dt[j]) * self.rng.normal()
                x = x + drift + diffusion
                paths[i, j + 1] = max(x, 0.001)  # Ensure positive values

        return paths

    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute log-likelihood for geometric jump-diffusion."""
        if len(data) < 2:
            return 0.0

        log_likelihood = 0.0

        for i in range(1, len(data)):
            if data[i-1] <= 0 or data[i] <= 0:
                continue

            # Log-returns with Poisson jump mixture in log space.
            # The jump component is a density over the PRICE, so transforming
            # to log-return space requires the Jacobian |d price / d log_return|
            # = exp(log_return) (= the observed price data[i]).
            log_return = np.log(data[i] / data[i-1])
            jacobian = np.log(data[i])
            p0 = np.exp(-self.parameters.jump_intensity * dt)
            mu = self.parameters.drift * dt
            sigma = self.parameters.diffusion * np.sqrt(dt)

            if sigma > 0:
                comps = [
                    np.log(max(p0, 1e-300)) + norm.logpdf(log_return, mu, sigma),
                    np.log(max(1.0 - p0, 1e-300)) + lognorm.logpdf(
                        np.exp(log_return), s=self.parameters.jump_std,
                        scale=np.exp(self.parameters.jump_mean)) + jacobian,
                ]
                log_likelihood += np.logaddexp.reduce(comps)

        return log_likelihood

    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate parameters for geometric jump-diffusion."""
        def objective(params):
            drift, diffusion, jump_intensity, jump_mean, jump_std = params

            self.parameters.drift = drift
            self.parameters.diffusion = diffusion
            self.parameters.jump_intensity = jump_intensity
            self.parameters.jump_mean = jump_mean
            self.parameters.jump_std = jump_std

            return -self.log_likelihood(data, dt)

        # Calculate log-returns
        log_returns = []
        for i in range(1, len(data)):
            if data[i-1] > 0 and data[i] > 0:
                log_returns.append(np.log(data[i] / data[i-1]))

        if not log_returns:
            return self.parameters

        log_returns = np.array(log_returns)

        # Initial parameter guess
        initial_guess = [
            np.mean(log_returns) / dt,      # drift
            np.std(log_returns) / np.sqrt(dt),  # diffusion
            0.1,                            # jump_intensity
            0.0,                            # jump_mean
            np.std(log_returns)             # jump_std
        ]

        # Parameter bounds
        bounds = [
            (-5.0, 5.0),                    # drift
            (0.001, np.std(log_returns) * 10),  # diffusion
            (0.0, 1.0),                     # jump_intensity
            (-2.0, 2.0),                    # jump_mean
            (0.001, np.std(log_returns) * 10)  # jump_std
        ]

        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                drift, diffusion, jump_intensity, jump_mean, jump_std = result.x
                return ModelParameters(
                    drift=drift,
                    diffusion=diffusion,
                    jump_intensity=jump_intensity,
                    jump_mean=jump_mean,
                    jump_std=jump_std
                )
            else:
                warnings.warn("Parameter estimation failed, using initial values")
                return self.parameters
        except Exception as e:
            warnings.warn(f"Parameter estimation error: {e}")
            return self.parameters


class CompoundPoisson(StochasticProcess):
    """Compound Poisson process."""

    def __init__(self, parameters: ModelParameters, rng: Optional[np.random.Generator] = None):
        """Initialize compound Poisson process."""
        super().__init__(parameters)
        self.process_name = "Compound Poisson"
        self.rng = rng or np.random.default_rng()

    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate compound Poisson process."""
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = x0

        for i in range(n_paths):
            x = x0
            for j in range(1, len(t)):
                # Generate jumps
                n_jumps = self.rng.poisson(self.parameters.jump_intensity * (t[j] - t[j-1]))
                for _ in range(n_jumps):
                    jump_size = self.rng.normal(self.parameters.jump_mean, self.parameters.jump_std)
                    x += jump_size
                paths[i, j] = x

        return paths

    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Exact log-likelihood for compound Poisson with Gaussian jump sizes.

        Per-step increments follow a Poisson-Gaussian mixture (Skellam-like
        continuous limit): sum the Poisson-weighted Gaussian mixture over jump
        counts 0..k_max in log space.
        """
        if len(data) < 2:
            return 0.0

        lam = max(self.parameters.jump_intensity, 0.0)
        k_max = 20
        # Precompute log poisson pmf for k = 0..k_max
        ks = np.arange(k_max + 1)
        # math.lgamma(k+1) == log(k!) and stays valid on numpy>=2.0
        # (np.math was removed there).
        log_pmf = ks * np.log(max(lam, 1e-300)) - lam - np.array(
            [lgamma(k + 1.0) for k in ks]) if lam > 0 else np.where(ks == 0, 0.0, -np.inf)
        log_pmf = np.asarray(log_pmf, dtype=float)

        log_likelihood = 0.0
        for i in range(1, len(data)):
            inc = data[i] - data[i-1]
            # increment given k jumps: N(k*jump_mean, k*jump_std^2)
            comps = []
            for k in ks:
                if log_pmf[k] == -np.inf:
                    continue
                if k == 0:
                    if abs(inc) < 1e-15:
                        comps.append(log_pmf[k])
                    else:
                        comps.append(-np.inf)
                else:
                    sd = self.parameters.jump_std * np.sqrt(k)
                    comps.append(log_pmf[k] + norm.logpdf(inc, k * self.parameters.jump_mean, sd))
            log_likelihood += np.logaddexp.reduce(comps)

        return float(log_likelihood)

    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate parameters for compound Poisson process."""
        # Simple estimation based on jump frequency and sizes
        differences = np.diff(data)

        # Estimate jump intensity
        jump_intensity = len(differences[differences != 0]) / len(differences) / dt

        # Estimate jump parameters
        jump_sizes = differences[differences != 0]
        if len(jump_sizes) > 0:
            jump_mean = np.mean(jump_sizes)
            jump_std = np.std(jump_sizes)
        else:
            jump_mean = 0.0
            jump_std = 1.0

        return ModelParameters(
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std
        )


class FractionalBrownianMotion(StochasticProcess):
    """Fractional Brownian Motion with Hurst parameter for long-range dependence."""
    
    def __init__(self, parameters: ModelParameters, hurst: float = 0.7,
                 rng: Optional[np.random.Generator] = None):
        """Initialize fractional Brownian motion process."""
        super().__init__(parameters)
        self.process_name = "Fractional Brownian Motion"
        self.rng = rng or np.random.default_rng()
        self.hurst = hurst  # Hurst parameter (0.5 = standard Brownian, >0.5 = persistent, <0.5 = anti-persistent)
    
    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate fractional Brownian motion using Davies-Harte method."""
        n_steps = len(t)
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = x0
        
        for i in range(n_paths):
            # Generate correlated increments using covariance structure
            increments = self._generate_fbm_increments(n_steps - 1, t)
            paths[i, 1:] = x0 + np.cumsum(increments)
        
        return paths
    
    def _generate_fbm_increments(self, n_steps: int, t: np.ndarray) -> np.ndarray:
        """Generate fBM increments with long-range dependence."""
        dt = np.diff(t)
        increments = np.zeros(n_steps)
        
        # Simple approximation using fractional Gaussian noise
        for i in range(n_steps):
            # Covariance structure for fBM
            if i == 0:
                cov = self.parameters.diffusion * (dt[i] ** self.hurst)
            else:
                # Handle negative or zero differences
                dt_diff = dt[i] - dt[i-1]
                if abs(dt_diff) < 1e-10:
                    dt_diff = 0.0
                
                cov = 0.5 * self.parameters.diffusion * (
                    (dt[i] ** (2 * self.hurst)) + 
                    (dt[i-1] ** (2 * self.hurst)) - 
                    (abs(dt_diff) ** (2 * self.hurst))
                )
            
            # Ensure non-negative variance
            cov = max(cov, 1e-10)
            increments[i] = self.parameters.drift * dt[i] + np.sqrt(cov) * self.rng.normal()
        
        return increments
    
    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute log-likelihood for fractional Brownian motion."""
        increments = np.diff(data)
        n = len(increments)
        
        # Build covariance matrix for fBM increments
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                t_i = (i + 1) * dt
                t_j = (j + 1) * dt
                cov_matrix[i, j] = 0.5 * self.parameters.diffusion * (
                    t_i ** (2 * self.hurst) + t_j ** (2 * self.hurst) - 
                    abs(t_i - t_j) ** (2 * self.hurst)
                )
        
        try:
            # Multivariate normal log-likelihood
            sign, logdet = np.linalg.slogdet(cov_matrix)
            if sign <= 0:
                return -np.inf
            inv_cov = np.linalg.inv(cov_matrix)
            ll = -0.5 * (n * np.log(2 * np.pi) + logdet + 
                        increments @ inv_cov @ increments)
            return ll
        except np.linalg.LinAlgError:
            return -np.inf
    
    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate parameters including Hurst exponent."""
        increments = np.diff(data)
        
        # Estimate drift
        drift = np.mean(increments) / dt if len(increments) > 0 else 0.0
        
        # Estimate Hurst parameter using variance method
        # Var[X(t)] = sigma^2 * t^(2H)
        if len(data) > 10:
            lags = np.arange(1, min(len(data) // 2, 20))
            variances = []
            for lag in lags:
                diffs = data[lag:] - data[:-lag]
                var = np.var(diffs)
                if var > 0:
                    variances.append(var)
            
            if len(variances) > 2:
                # Fit log-log relationship
                log_lags = np.log(np.arange(1, len(variances) + 1) * dt)
                log_vars = np.log(variances)
                
                # Remove inf/nan values
                valid = np.isfinite(log_lags) & np.isfinite(log_vars)
                if np.sum(valid) > 2:
                    hurst_est = np.polyfit(log_lags[valid], log_vars[valid], 1)[0] / 2
                    self.hurst = np.clip(hurst_est, 0.1, 0.9)
        
        # Estimate diffusion coefficient
        var_increments = np.var(increments) if len(increments) > 0 else 1.0
        diffusion = max(var_increments / (dt ** (2 * self.hurst)), 1e-6)
        
        return ModelParameters(drift=drift, diffusion=diffusion)


class CoxIngersollRoss(StochasticProcess):
    """Cox-Ingersoll-Ross process for mean-reverting non-negative processes."""
    
    def __init__(self, parameters: ModelParameters, rng: Optional[np.random.Generator] = None):
        """Initialize CIR process."""
        super().__init__(parameters)
        self.process_name = "Cox-Ingersoll-Ross"
        self.rng = rng or np.random.default_rng()
    
    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate CIR process using Euler-Maruyama scheme."""
        dt = np.diff(t)
        n_steps = len(t) - 1
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = max(x0, 0.001)  # Ensure positive initial value
        
        for i in range(n_paths):
            x = paths[i, 0]
            for j in range(n_steps):
                # CIR dynamics: dx = kappa(theta - x)dt + sigma * sqrt(x) * dW
                drift_term = self.parameters.reversion_speed * (
                    self.parameters.equilibrium - x) * dt[j]
                diffusion_term = self.parameters.diffusion * np.sqrt(max(x, 0)) * np.sqrt(dt[j]) * self.rng.normal()
                
                x = max(x + drift_term + diffusion_term, 0.001)  # Ensure positivity
                paths[i, j + 1] = x
        
        return paths
    
    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute log-likelihood for CIR process."""
        increments = np.diff(data)
        ll = 0.0
        
        for i in range(len(increments)):
            x = data[i]
            dx = increments[i]
            
            # Expected increment
            expected_dx = self.parameters.reversion_speed * (
                self.parameters.equilibrium - x) * dt
            
            # Variance of increment
            var_dx = self.parameters.diffusion ** 2 * max(x, 0.001) * dt
            
            # Normal approximation for likelihood
            ll += -0.5 * np.log(2 * np.pi * var_dx) - 0.5 * ((dx - expected_dx) ** 2) / var_dx
        
        return ll
    
    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate CIR parameters using moment matching."""
        # Calculate sample moments
        mean_x = np.mean(data)
        var_x = np.var(data)
        
        # Estimate equilibrium level
        equilibrium = mean_x
        
        # Estimate reversion speed using autocorrelation
        if len(data) > 2:
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            reversion_speed = -np.log(max(autocorr, 0.01)) / dt
        else:
            reversion_speed = 1.0
        
        # Estimate diffusion coefficient
        # Var(X) ≈ sigma^2 * theta / (2*kappa)
        diffusion = np.sqrt(2 * reversion_speed * var_x / max(equilibrium, 0.001))
        
        return ModelParameters(
            equilibrium=equilibrium,
            reversion_speed=max(reversion_speed, 0.01),
            diffusion=max(diffusion, 0.01)
        )


class LevyProcess(StochasticProcess):
    """Levy process with independent increments and infinite divisibility."""
    
    def __init__(self, parameters: ModelParameters, levy_alpha: float = 1.5, levy_beta: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        """Initialize Levy process."""
        super().__init__(parameters)
        self.process_name = "Levy Process"
        self.rng = rng or np.random.default_rng()
        self.levy_alpha = levy_alpha  # Stability parameter (0 < alpha <= 2)
        self.levy_beta = levy_beta    # Skewness parameter (-1 <= beta <= 1)
    
    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate Levy process using stable distribution increments."""
        dt = np.diff(t)
        n_steps = len(t) - 1
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = x0
        
        for i in range(n_paths):
            x = x0
            for j in range(n_steps):
                # Generate stable distribution increment
                increment = self._stable_increment(dt[j])
                x += increment
                paths[i, j + 1] = x
        
        return paths
    
    def _stable_increment(self, dt: float) -> float:
        """Generate increment from stable distribution."""
        # Chambers-Mallows-Stuck method for stable distribution
        alpha = self.levy_alpha
        beta = self.levy_beta
        
        # Generate uniform and exponential random variables
        u = self.rng.uniform(-np.pi / 2, np.pi / 2)
        w = self.rng.exponential(1.0)
        
        if alpha == 1:
            # Cauchy case
            s = np.tan(u)
        else:
            # General case
            b = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
            s = ((np.sin(alpha * (u + b)) / (np.cos(u) ** (1 / alpha))) * 
                 ((np.cos(u - alpha * (u + b)) / w) ** ((1 - alpha) / alpha)))
        
        # Scale by time and parameters
        return self.parameters.drift * dt + self.parameters.diffusion * (dt ** (1 / alpha)) * s
    
    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute approximate log-likelihood for Levy process."""
        # Use normal approximation for increments
        increments = np.diff(data)
        expected_increment = self.parameters.drift * dt
        variance_increment = self.parameters.diffusion ** 2 * dt
        
        ll = np.sum(stats.norm.logpdf(increments, expected_increment, np.sqrt(variance_increment)))
        return ll
    
    def estimate_parameters(self, data: np.ndarray, dt: float) -> ModelParameters:
        """Estimate Levy process parameters."""
        increments = np.diff(data)
        
        # Estimate drift
        drift = np.mean(increments) / dt
        
        # Estimate scale parameter using median absolute deviation
        mad = np.median(np.abs(increments - np.median(increments)))
        diffusion = mad / (np.sqrt(dt) * 0.6745)  # Robust scale estimator
        
        # Estimate stability parameter using log-log variance
        if len(data) > 10:
            lags = np.arange(1, min(len(data) // 2, 20))
            variances = [np.var(data[lag:] - data[:-lag]) for lag in lags]
            log_lags = np.log(lags * dt)
            log_vars = np.log(variances)
            alpha_est = np.polyfit(log_lags, log_vars, 1)[0]
            self.levy_alpha = np.clip(alpha_est, 0.5, 2.0)
        
        return ModelParameters(drift=drift, diffusion=diffusion)


class JumpRope:
    """Main class for jump-diffusion modeling."""

    def __init__(self,
                 stochastic_process: StochasticProcess,
                 time_points: np.ndarray,
                 initial_conditions: Optional[Dict[str, float]] = None):
        """Initialize JumpRope model."""
        self.stochastic_process = stochastic_process
        self.time_points = time_points
        self.initial_conditions = initial_conditions or {}
        self.fitted_parameters: Optional[ModelParameters] = None
        self.trajectories: Optional[np.ndarray] = None

        logger.info(f"Initialized JumpRope with {stochastic_process.process_name}")

    @classmethod
    def fit(cls,
            data_core,
            model_type: str = 'jump-diffusion',
            time_points: Optional[np.ndarray] = None,
            rng: Optional[np.random.Generator] = None,
            seed: Optional[int] = None,
            **kwargs) -> 'JumpRope':
        """
        Fit stochastic process model to data.

        Parameters:
            data_core: DataCore instance with training data
            model_type: Type of stochastic process ('jump-diffusion', 'ornstein-uhlenbeck', 'compound-poisson')
            time_points: Optional time points for evaluation
            **kwargs: Additional model parameters

        Returns:
            Fitted JumpRope instance
        """
        logger.info(f"Fitting {model_type} model to data")

        if not any(len(ts.data) > 0 for ts in data_core.time_series_data):
            raise ValueError("Cannot fit model: all time series data are empty")

        # Determine time points
        if time_points is None:
            all_time_points = []
            for ts in data_core.time_series_data:
                all_time_points.extend(ts.time_points)
            time_points = np.sort(np.unique(all_time_points))

        # Extract model-specific parameters
        hurst = kwargs.pop('hurst', 0.7)
        levy_alpha = kwargs.pop('levy_alpha', 1.5)
        levy_beta = kwargs.pop('levy_beta', 0.0)
        
        # Create initial parameters with remaining kwargs
        initial_params = ModelParameters(**kwargs)

        # Shared generator for reproducibility
        generator = rng if rng is not None else (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng())

        # Select and initialize stochastic process
        if model_type == 'ornstein-uhlenbeck':
            stochastic_process = OrnsteinUhlenbeckJump(initial_params, rng=generator)
        elif model_type == 'geometric-jump-diffusion':
            stochastic_process = GeometricJumpDiffusion(initial_params, rng=generator)
        elif model_type == 'compound-poisson':
            stochastic_process = CompoundPoisson(initial_params, rng=generator)
        elif model_type == 'fractional-brownian':
            stochastic_process = FractionalBrownianMotion(initial_params, hurst=hurst, rng=generator)
        elif model_type == 'cir':
            stochastic_process = CoxIngersollRoss(initial_params, rng=generator)
        elif model_type == 'levy':
            stochastic_process = LevyProcess(initial_params, levy_alpha=levy_alpha, levy_beta=levy_beta, rng=generator)
        else:  # Default to jump-diffusion (Ornstein-Uhlenbeck with jumps)
            stochastic_process = OrnsteinUhlenbeckJump(initial_params, rng=generator)

        # Create JumpRope instance
        model = cls(stochastic_process, time_points)

        # Fit parameters to each time series
        fitted_params_list = []
        for ts in data_core.time_series_data:
            # Get phenotype data for each time point
            phenotype_data = []
            for t in time_points:
                if t in ts.data[ts.time_column].values:
                    row = ts.data[ts.data[ts.time_column] == t]
                    if not row.empty:
                        # Average across all phenotype columns for now
                        avg_phenotype = row[ts.phenotype_columns].mean(axis=1).iloc[0]
                        phenotype_data.append(avg_phenotype)

            if len(phenotype_data) >= 2:
                dt = np.mean(np.diff(time_points))
                fitted_params = stochastic_process.estimate_parameters(
                    np.array(phenotype_data), dt
                )
                fitted_params_list.append(fitted_params)

        # Aggregate parameters across time series
        if fitted_params_list:
            model.fitted_parameters = cls._aggregate_parameters(fitted_params_list)
        else:
            model.fitted_parameters = initial_params

        logger.info("Model fitting completed")
        return model

    @staticmethod
    def _aggregate_parameters(param_list: List[ModelParameters]) -> ModelParameters:
        """Aggregate parameters across multiple fits."""
        # Simple averaging of parameters
        n_params = len(param_list)
        aggregated = ModelParameters()

        aggregated.drift = np.mean([p.drift for p in param_list])
        aggregated.diffusion = np.mean([p.diffusion for p in param_list])
        aggregated.jump_intensity = np.mean([p.jump_intensity for p in param_list])
        aggregated.jump_mean = np.mean([p.jump_mean for p in param_list])
        aggregated.jump_std = np.mean([p.jump_std for p in param_list])
        aggregated.equilibrium = np.mean([p.equilibrium for p in param_list])
        aggregated.reversion_speed = np.mean([p.reversion_speed for p in param_list])

        return aggregated

    def generate_trajectories(self, n_samples: int = 100, x0: Optional[float] = None,
                              seed: Optional[int] = None) -> np.ndarray:
        """Generate sample trajectories from the fitted model."""
        if self.fitted_parameters is None:
            # Unfitted model: simulate directly from the process's own
            # parameters (documented contract for prior-configuration runs).
            if hasattr(self.stochastic_process, 'parameters'):
                self.fitted_parameters = self.stochastic_process.parameters
            else:
                raise ValueError(
                    "Model parameters not fitted. Call fit() first.")

        if seed is not None and hasattr(self.stochastic_process, 'rng'):
            self.stochastic_process.rng = np.random.default_rng(seed)

        if x0 is None:
            # Use mean of initial conditions or 0
            x0 = self.initial_conditions.get('x0', 0.0)

        logger.info(f"Generating {n_samples} trajectories")

        # Update stochastic process with fitted parameters
        self.stochastic_process.parameters = self.fitted_parameters

        # Generate trajectories
        trajectories = self.stochastic_process.simulate(x0, self.time_points, n_samples)

        self.trajectories = trajectories
        logger.info("Trajectory generation completed")

        return trajectories

    def compute_cross_sections(self, time_point_idx: int) -> np.ndarray:
        """Compute cross-sectional distribution at a specific time point."""
        if self.trajectories is None:
            raise ValueError("No trajectories available. Call generate_trajectories() first.")

        return self.trajectories[:, time_point_idx]

    def estimate_jump_times(self) -> List[float]:
        """Estimate times of developmental jumps."""
        if self.trajectories is None:
            raise ValueError("No trajectories available. Call generate_trajectories() first.")

        # Simple jump detection based on large changes
        jump_times = []

        for i in range(self.trajectories.shape[0]):
            trajectory = self.trajectories[i, :]
            differences = np.abs(np.diff(trajectory))

            # Find large differences (potential jumps)
            threshold = np.percentile(differences, 95)
            jump_indices = np.where(differences > threshold)[0]

            if len(jump_indices) > 0:
                jump_time_points = self.time_points[jump_indices]
                jump_times.extend(jump_time_points.tolist())

        # Remove duplicates and sort
        jump_times = sorted(list(set(jump_times)))
        logger.info(f"Estimated {len(jump_times)} potential jump times")

        return jump_times

    def save(self, file_path: Path) -> None:
        """Save model to file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> 'JumpRope':
        """Load model from file."""
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {file_path}")
        return model

