"""
JumpRope Engine: Jump-Diffusion Modeling for Developmental Trajectories

This module implements jump-diffusion modeling for developmental trajectories.
It processes temporal phenotypic data to estimate drift parameters, diffusion coefficients,
and jump intensities characterizing developmental transitions. The engine supports multiple
stochastic process models including Ornstein-Uhlenbeck processes with jumps, geometric
jump-diffusion, and compound Poisson processes.

Classes:
    JumpRope: Main class for jump-diffusion modeling
    StochasticProcess: Base class for stochastic process models
    OrnsteinUhlenbeckJump: Ornstein-Uhlenbeck process with jumps
    GeometricJumpDiffusion: Geometric jump-diffusion process
    CompoundPoisson: Compound Poisson process

Examples:
    >>> # Fit jump rope model to data
    >>> model = JumpRope.fit(data, model_type='jump-diffusion')
    >>> # Generate trajectories
    >>> trajectories = model.generate_trajectories(n_samples=100)
    >>> # Estimate parameters
    >>> params = model.estimate_parameters()
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from scipy.stats import norm, lognorm
from numba import jit, cuda
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

    def __init__(self, parameters: ModelParameters):
        """Initialize Ornstein-Uhlenbeck process with jumps."""
        super().__init__(parameters)
        self.process_name = "Ornstein-Uhlenbeck with Jumps"

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
                diffusion = self.parameters.diffusion * np.sqrt(dt[j]) * np.random.normal()

                # Jump part (Poisson process)
                jump_occurred = np.random.poisson(self.parameters.jump_intensity * dt[j])
                if jump_occurred > 0:
                    jump_size = np.random.normal(self.parameters.jump_mean, self.parameters.jump_std)
                else:
                    jump_size = 0.0

                x = x + drift + diffusion + jump_size
                paths[i, j + 1] = x

        return paths

    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute log-likelihood for Ornstein-Uhlenbeck process with jumps."""
        if len(data) < 2:
            return 0.0

        # Separate continuous and jump components
        log_likelihood = 0.0

        for i in range(1, len(data)):
            # Continuous component (Ornstein-Uhlenbeck)
            mu = data[i-1] + (self.parameters.equilibrium - data[i-1]) * self.parameters.reversion_speed * dt
            sigma = self.parameters.diffusion * np.sqrt(dt)

            if sigma > 0:
                continuous_ll = norm.logpdf(data[i], mu, sigma)
                log_likelihood += continuous_ll

            # Jump component (Poisson)
            jump_ll = -self.parameters.jump_intensity * dt  # Log-likelihood for no jump
            log_likelihood += jump_ll

        return log_likelihood

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
    """Geometric jump-diffusion process."""

    def __init__(self, parameters: ModelParameters):
        """Initialize geometric jump-diffusion process."""
        super().__init__(parameters)
        self.process_name = "Geometric Jump-Diffusion"

    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate geometric jump-diffusion process."""
        dt = np.diff(t)
        n_steps = len(t) - 1
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = x0

        for i in range(n_paths):
            x = x0
            for j in range(n_steps):
                # Continuous part (geometric Brownian motion)
                drift = self.parameters.drift * x * dt[j]
                diffusion = self.parameters.diffusion * x * np.sqrt(dt[j]) * np.random.normal()

                # Jump part (compound Poisson)
                jump_occurred = np.random.poisson(self.parameters.jump_intensity * dt[j])
                if jump_occurred > 0:
                    jump_factor = np.random.lognormal(self.parameters.jump_mean, self.parameters.jump_std)
                    x = x * jump_factor
                else:
                    jump_factor = 1.0

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

            # Log-returns
            log_return = np.log(data[i] / data[i-1])

            # Continuous component
            mu = self.parameters.drift * dt
            sigma = self.parameters.diffusion * np.sqrt(dt)

            if sigma > 0:
                continuous_ll = norm.logpdf(log_return, mu, sigma)
                log_likelihood += continuous_ll

            # Jump component
            jump_ll = -self.parameters.jump_intensity * dt
            log_likelihood += jump_ll

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

    def __init__(self, parameters: ModelParameters):
        """Initialize compound Poisson process."""
        super().__init__(parameters)
        self.process_name = "Compound Poisson"

    def simulate(self, x0: float, t: np.ndarray, n_paths: int = 1) -> np.ndarray:
        """Simulate compound Poisson process."""
        paths = np.zeros((n_paths, len(t)))
        paths[:, 0] = x0

        for i in range(n_paths):
            x = x0
            for j in range(1, len(t)):
                # Generate jumps
                n_jumps = np.random.poisson(self.parameters.jump_intensity * (t[j] - t[j-1]))
                for _ in range(n_jumps):
                    jump_size = np.random.normal(self.parameters.jump_mean, self.parameters.jump_std)
                    x += jump_size
                paths[i, j] = x

        return paths

    def log_likelihood(self, data: np.ndarray, dt: float) -> float:
        """Compute log-likelihood for compound Poisson process."""
        if len(data) < 2:
            return 0.0

        log_likelihood = 0.0

        for i in range(1, len(data)):
            # Jump component
            jump_ll = -self.parameters.jump_intensity * dt
            log_likelihood += jump_ll

        return log_likelihood

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

        # Determine time points
        if time_points is None:
            all_time_points = []
            for ts in data_core.time_series_data:
                all_time_points.extend(ts.time_points)
            time_points = np.sort(np.unique(all_time_points))

        # Create initial parameters
        initial_params = ModelParameters(**kwargs)

        # Select and initialize stochastic process
        if model_type == 'ornstein-uhlenbeck':
            stochastic_process = OrnsteinUhlenbeckJump(initial_params)
        elif model_type == 'geometric-jump-diffusion':
            stochastic_process = GeometricJumpDiffusion(initial_params)
        elif model_type == 'compound-poisson':
            stochastic_process = CompoundPoisson(initial_params)
        else:  # Default to jump-diffusion (Ornstein-Uhlenbeck with jumps)
            stochastic_process = OrnsteinUhlenbeckJump(initial_params)

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

    def generate_trajectories(self, n_samples: int = 100, x0: Optional[float] = None) -> np.ndarray:
        """Generate sample trajectories from the fitted model."""
        if self.fitted_parameters is None:
            raise ValueError("Model parameters not fitted. Call fit() first.")

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
