"""
Evolution Sampler: Population-Level Analysis

This module handles population-level analysis by sampling multiple developmental trajectories
and performing comparative evolutionary analysis. Implements phylogenetic comparative methods,
quantitative genetics approaches, and population dynamics modeling.
Quantities that cannot be identified from the available data (e.g. dominance
variance, broad-sense heritability without family structure) are reported as
``np.nan`` with an ``'available'`` marker rather than placeholder values.

Classes:
    EvolutionSampler: Main class for evolutionary sampling
    PopulationModel: Models population-level dynamics
    PhylogeneticAnalyzer: Performs phylogenetic comparative analysis
    QuantitativeGenetics: Analyzes genetic contributions to traits

Examples:
    >>> # Create sampler
    >>> sampler = EvolutionSampler(population_data)
    >>> # Sample from population
    >>> samples = sampler.sample(n_samples=1000, method='monte-carlo')
    >>> # Analyze evolutionary patterns
    >>> patterns = sampler.analyze_evolutionary_patterns()
"""

import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Container for sampling results."""
    samples: np.ndarray
    sample_ids: List[str]
    sampling_method: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationStatistics:
    """Container for population-level statistics."""
    mean_trajectory: np.ndarray
    variance_trajectory: np.ndarray
    covariance_matrix: np.ndarray
    heritability_estimates: Dict[str, float]
    selection_gradients: Dict[str, float]
    effective_population_size: float
    genetic_correlations: np.ndarray


class PopulationModel:
    """Models population-level dynamics and evolution."""

    def __init__(self, population_data: pd.DataFrame, time_column: str = 'time'):
        """Initialize population model."""
        self.population_data = population_data
        self.time_column = time_column
        self.individuals = population_data.columns.drop(time_column) if time_column in population_data.columns else population_data.columns
        self.time_points = population_data[time_column].unique() if time_column in population_data.columns else None

    def estimate_heritability(self, phenotype: str, method: str = 'parent-offspring') -> float:
        """
        Estimate heritability of a phenotype.

        Parameters:
            phenotype: Name of phenotype column
            method: Estimation method

        Returns:
            Heritability estimate
        """
        if method == 'parent-offspring':
            # Parent-offspring regression requires explicit generational pairing.
            # Row order in a time-series table is NOT a pedigree; naive
            # next-row pairing conflates time with relatedness, so this method
            # only runs when the frame carries explicit 'parent'/'offspring'
            # phenotype columns.
            if 'parent' in self.population_data.columns and 'offspring' in self.population_data.columns:
                parent_values = self.population_data['parent'].dropna()
                offspring_values = self.population_data['offspring'].dropna()
                n = min(len(parent_values), len(offspring_values))
                if n < 4:
                    return np.nan
                slope, _, _, _, _ = stats.linregress(
                    parent_values.iloc[:n], offspring_values.iloc[:n])
                heritability = 2 * slope if slope > 0 else 0.0
                return min(heritability, 1.0)  # Cap at 1.0

            warnings.warn(
                "parent-offspring heritability requires explicit "
                "'parent'/'offspring' columns (a pedigree); returning NaN "
                "instead of a spurious estimate.")
            return np.nan

        else:
            raise ValueError(f"Unsupported heritability method: {method}")

    def compute_selection_gradient(self, phenotype: str, fitness_measure: str) -> float:
        """
        Compute the univariate directional selection gradient for a phenotype.

        The gradient is the standardized regression slope of relative fitness
        on the phenotype,

            beta_std = cov(z, w) / var(z),

        where z and w are the standardized phenotype and fitness. Expressed
        in standard-deviation units this equals the Pearson correlation
        r(z, w): +1 (or -1) means fitness increases (decreases) perfectly
        linearly with the phenotype.

        Parameters:
            phenotype: Name of phenotype column
            fitness_measure: Name of fitness column

        Returns:
            Standardized selection gradient in units of SD(fitness) per
            SD(phenotype); np.nan when either column is missing, either
            variable has zero variance, or fewer than two finite paired
            observations are available.
        """
        if phenotype not in self.population_data.columns or fitness_measure not in self.population_data.columns:
            return np.nan

        pheno = self.population_data[phenotype].to_numpy(dtype=float)
        fitness = self.population_data[fitness_measure].to_numpy(dtype=float)
        valid = np.isfinite(pheno) & np.isfinite(fitness)
        if valid.sum() < 2:
            return np.nan
        pheno = pheno[valid]
        fitness = fitness[valid]

        pheno_sd = float(pheno.std())
        fitness_sd = float(fitness.std())
        if not (pheno_sd > 0 and fitness_sd > 0):
            return np.nan

        z = (pheno - pheno.mean()) / pheno_sd
        w = (fitness - fitness.mean()) / fitness_sd

        # Standardized regression slope: cov(z, w) / var(z). var(z) is 1 by
        # construction, but the division is kept explicit so the code states
        # the quantity it documents (the plain covariance would coincide with
        # it only by accident of the z-scoring).
        return float(np.mean(z * w) / np.mean(z ** 2))

    def estimate_effective_population_size(self, method: str = 'temporal') -> float:
        """
        Estimate effective population size.

        Parameters:
            method: Estimation method

        Returns:
            Effective population size estimate
        """
        if method == 'temporal':
            # The temporal method REQUIRES genetic markers (allele-frequency
            # changes across time points); phenotypes alone cannot identify Ne.
            # This method no longer fabricates an estimate from phenotypic
            # variance. Use `predict_phenotypic_response` for the honest
            # phenotype-only analogue (Lande equation R = h2 * S).
            freq_cols = [c for c in self.population_data.columns
                         if str(c).startswith('freq_')]
            if len(freq_cols) == 0 or self.time_points is None or len(self.time_points) < 2:
                warnings.warn(
                    "temporal Ne estimation requires allele-frequency columns "
                    "(prefixed 'freq_') at >= 2 time points; returning NaN. "
                    "Phenotypes alone cannot identify Ne - see "
                    "predict_phenotypic_response for the Lande-equation analogue.")
                return np.nan

            # Waples (1989) plan-II temporal method:
            # F = mean over alleles of Var(p_t2 - p_t1) / (p_bar (1 - p_bar));
            # Ne = 1 / (2F) per generation.
            f_values = []
            times = sorted(self.time_points)
            t1_data = self.population_data[self.population_data[self.time_column] == times[0]]
            t2_data = self.population_data[self.population_data[self.time_column] == times[-1]]
            for col in freq_cols:
                p1 = float(np.nanmean(t1_data[col].to_numpy(dtype=float)))
                p2 = float(np.nanmean(t2_data[col].to_numpy(dtype=float)))
                p_bar = 0.5 * (p1 + p2)
                if 0.0 < p_bar < 1.0:
                    f_values.append((p2 - p1) ** 2 / (p_bar * (1.0 - p_bar)))
            if not f_values:
                return np.nan
            f_stat = float(np.mean(f_values))
            if f_stat <= 0:
                return np.inf
            return 1.0 / (2.0 * f_stat)

        else:
            raise ValueError(f"Unsupported method: {method}")

    def compute_selection_differential(self, phenotype: str) -> float:
        """Selection differential S = mean(after selection) - mean(before).

        With time-series phenotype data (no explicit fitness column), S is
        the change in mean phenotype between the first and last time points.
        """
        if phenotype not in self.population_data.columns:
            return np.nan
        if self.time_column not in self.population_data.columns or self.time_points is None or len(self.time_points) < 2:
            return np.nan
        times = sorted(self.time_points)
        first = self.population_data[self.population_data[self.time_column] == times[0]][phenotype].dropna()
        last = self.population_data[self.population_data[self.time_column] == times[-1]][phenotype].dropna()
        if len(first) < 1 or len(last) < 1:
            return np.nan
        return float(last.mean() - first.mean())

    def predict_phenotypic_response(self, phenotype: str, h2: float) -> float:
        """Lande (1979) response to selection R = h2 * S.

        Honest phenotype-only analogue of a full quantitative-genetic
        projection: uses the measured selection differential S (first vs last
        time point) and a SUPPLIED narrow-sense heritability h2 in [0, 1].
        """
        if h2 is None or not (0.0 <= h2 <= 1.0):
            raise ValueError("h2 must be supplied and lie in [0, 1]")
        s = self.compute_selection_differential(phenotype)
        if s is None or np.isnan(s):
            return np.nan
        return float(h2 * s)


class PhylogeneticAnalyzer:
    """Performs phylogenetic comparative analysis."""

    def __init__(self, distance_matrix: Optional[np.ndarray] = None):
        """Initialize phylogenetic analyzer."""
        self.distance_matrix = distance_matrix
        self.phylogeny: Optional[Any] = None

    def compute_morans_i_signal(self, traits: np.ndarray) -> float:
        """Compute Moran's I phylogenetic signal from the distance matrix.

        Uses an inverse-squared-distance spatial weights matrix; returns I in
        [-1, 1]. Values near +1 indicate strong phylogenetic clustering of
        trait values. Requires a distance matrix; returns np.nan otherwise.
        """
        if self.distance_matrix is None:
            return np.nan
        traits = np.asarray(traits, dtype=float)
        traits = traits[~np.isnan(traits)]
        n = len(traits)
        if n < 3 or self.distance_matrix.shape[0] != n:
            return np.nan

        d = self.distance_matrix[:n, :n]
        W = 1.0 / np.maximum(d, 1e-12) ** 2
        np.fill_diagonal(W, 0.0)
        W_sum = W.sum()
        if W_sum <= 0:
            return np.nan

        centered = traits - traits.mean()
        denom = float(np.sum(centered ** 2))
        if denom <= 0:
            return np.nan
        I = (n / W_sum) * (centered @ W @ centered) / denom
        return float(I)

    def compute_phylogenetic_signal(self,
                                  traits: np.ndarray,
                                  method: str = 'lambda') -> float:
        """
        Compute phylogenetic signal in traits.

        Parameters:
            traits: Trait values for each species/individual
            method: Method for computing phylogenetic signal

        Returns:
            Phylogenetic signal estimate
        """
        if method == 'lambda':
            # Pagel's lambda estimation
            # This is a simplified implementation
            if self.distance_matrix is None:
                # If no phylogeny, assume no signal
                return 0.0

            n = len(traits)
            if n < 3:
                return np.nan

            # Brownian-motion phylogenetic covariance from the double-centered
            # squared-distance kernel (validated below), then optimize lambda.
            phylo_matrix = self._compute_brownian_covariance(self.distance_matrix)

            # The likelihood below is only meaningful for a positive
            # semidefinite covariance structure. The kernel is PSD whenever
            # the distance matrix is Euclidean (as cophenetic distances of
            # any tree are); for non-Euclidean input, project to the nearest
            # PSD matrix rather than let the optimizer chase a degenerate
            # (-inf) likelihood.
            eigvals = np.linalg.eigvalsh(phylo_matrix)
            if eigvals.min() < -1e-8 * max(1.0, float(np.abs(eigvals).max())):
                warnings.warn(
                    "distance matrix is not Euclidean: the Brownian-motion "
                    "kernel is not positive semidefinite; projecting to the "
                    "nearest PSD matrix before optimizing Pagel's lambda.")
                phylo_matrix = self._project_psd(phylo_matrix)

            # Optimize lambda
            def objective(lambda_val):
                if lambda_val < 0 or lambda_val > 1:
                    return np.inf

                # Transform covariance matrix
                transformed_matrix = lambda_val * phylo_matrix + (1 - lambda_val) * np.eye(n)

                # Compute log-likelihood
                try:
                    log_likelihood = self._compute_gaussian_loglikelihood(traits, transformed_matrix)
                    return -log_likelihood
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logger.debug("phylogenetic signal objective failed: %s", exc)
                    return np.inf

            # Optimize lambda
            result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
            return result.x if result.success else 0.0

        else:
            raise ValueError(f"Unsupported method: {method}")

    def _compute_brownian_covariance(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Compute a Brownian-motion phylogenetic covariance from a distance matrix.

        Uses the double-centered squared-distance (Gower/classical-MDS) kernel

            G = -0.5 * J D^2 J,      J = I - (1/n) 11^T,

        which equals the covariance of tip states under a Brownian motion on
        a tree whose cophenetic distances are D, and is positive semidefinite
        whenever D is Euclidean. The distance matrix itself is NOT a
        covariance (it has zero diagonal and need not be PSD).
        """
        d = np.asarray(distance_matrix, dtype=float)
        d_squared = d ** 2
        n = d.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        kernel = -0.5 * centering @ d_squared @ centering
        # Symmetrize away floating-point asymmetry.
        return 0.5 * (kernel + kernel.T)

    @staticmethod
    def _project_psd(matrix: np.ndarray) -> np.ndarray:
        """Project a symmetric matrix to the nearest positive semidefinite matrix."""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        projected = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
        return 0.5 * (projected + projected.T)

    def _compute_gaussian_loglikelihood(self,
                                       traits: np.ndarray,
                                       covariance_matrix: np.ndarray) -> float:
        """Compute log-likelihood under multivariate Gaussian."""
        n = len(traits)

        try:
            # Add small regularization
            reg_matrix = covariance_matrix + np.eye(n) * 1e-6

            # Compute log determinant
            sign, logdet = np.linalg.slogdet(reg_matrix)

            if sign <= 0:
                return -np.inf

            # Compute quadratic form
            traits_centered = traits - np.mean(traits)
            inv_matrix = np.linalg.inv(reg_matrix)
            quad_form = traits_centered.T @ inv_matrix @ traits_centered

            log_likelihood = -0.5 * (n * np.log(2 * np.pi) + logdet + quad_form)

            return log_likelihood

        except (ValueError, FloatingPointError) as exc:
            logger.debug("gaussian log-likelihood failed: %s", exc)
            return -np.inf


class QuantitativeGenetics:
    """Analyzes genetic contributions to developmental traits."""

    def __init__(self, genotype_data: Optional[pd.DataFrame] = None):
        """Initialize quantitative genetics analyzer."""
        self.genotype_data = genotype_data
        self.loci: List[str] = genotype_data.columns.tolist() if genotype_data is not None else []

    def estimate_breeding_values(self,
                                phenotype_data: pd.DataFrame,
                                method: str = 'blup') -> pd.DataFrame:
        """
        Estimate breeding values for individuals.

        Parameters:
            phenotype_data: Phenotypic measurements
            method: Estimation method

        Returns:
            DataFrame with breeding values
        """
        if method == 'blup':
            # Best Linear Unbiased Prediction
            # This is a simplified implementation
            n_individuals = len(phenotype_data)
            n_traits = phenotype_data.shape[1]

            breeding_values = pd.DataFrame(
                index=phenotype_data.index,
                columns=phenotype_data.columns
            )

            for trait in phenotype_data.columns:
                trait_data = phenotype_data[trait].dropna()

                if len(trait_data) < 3:
                    breeding_values[trait] = np.nan
                    continue

                # Simple BLUP approximation
                # In practice, this would use mixed models
                mean_value = trait_data.mean()
                breeding_values[trait] = trait_data - mean_value

            return breeding_values

        else:
            raise ValueError(f"Unsupported method: {method}")

    def compute_genetic_correlations(self,
                                   phenotype_data: pd.DataFrame,
                                   time_points: List[float]) -> np.ndarray:
        """
        Compute genetic correlations between traits.

        Parameters:
            phenotype_data: Phenotypic data
            time_points: Time points for correlation analysis

        Returns:
            Genetic correlation matrix
        """
        n_traits = len(phenotype_data.columns)
        correlation_matrix = np.zeros((n_traits, n_traits))

        for i, trait1 in enumerate(phenotype_data.columns):
            for j, trait2 in enumerate(phenotype_data.columns):
                if i <= j:
                    # Compute correlation at each time point
                    correlations = []

                    for time_point in time_points:
                        if time_point in phenotype_data.index:
                            trait1_data = phenotype_data.loc[time_point, trait1]
                            trait2_data = phenotype_data.loc[time_point, trait2]

                            # Handle both scalar and array data
                            if pd.isna(trait1_data) or pd.isna(trait2_data):
                                continue

                            try:
                                corr = np.corrcoef(trait1_data, trait2_data)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                            except (ValueError, FloatingPointError):
                                continue

                    if correlations:
                        correlation_matrix[i, j] = np.mean(correlations)
                        correlation_matrix[j, i] = np.mean(correlations)
                    else:
                        correlation_matrix[i, j] = np.nan
                        correlation_matrix[j, i] = np.nan
                else:
                    correlation_matrix[i, j] = correlation_matrix[j, i]

        return correlation_matrix


class EvolutionSampler:
    """Main class for evolutionary sampling and analysis."""

    def __init__(self,
                 population_data: Union[pd.DataFrame, 'datacore.DataCore'],
                 time_column: str = 'time'):
        """Initialize evolution sampler."""
        from . import datacore
        if isinstance(population_data, datacore.DataCore):
            # Extract data from DataCore
            combined_data = []
            for ts in population_data.time_series_data:
                combined_data.append(ts.data)
            self.population_data = pd.concat(combined_data, ignore_index=True)
            self.time_column = time_column
        else:
            self.population_data = population_data
            self.time_column = time_column

        self.population_model = PopulationModel(self.population_data, self.time_column)
        self.phylogenetic_analyzer = PhylogeneticAnalyzer()
        self.quantitative_genetics = QuantitativeGenetics()
        self._rng = np.random.default_rng()

        logger.info("Initialized Evolution Sampler")

    def seed(self, seed: int) -> None:
        """Seed the sampler's random generator for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def sample(self,
              n_samples: int = 1000,
              method: str = 'monte-carlo',
              parameters: Optional[Dict[str, Any]] = None) -> SampleResult:
        """
        Sample from evolutionary population.

        Parameters:
            n_samples: Number of samples to generate
            method: Sampling method
            parameters: Sampling parameters

        Returns:
            SampleResult with generated samples
        """
        logger.info(f"Sampling {n_samples} individuals using {method} method")

        # Work on a copy: sampling diagnostics are attached to this dict and
        # the result must not alias (or mutate) the caller's dictionary.
        parameters = dict(parameters or {})

        if method == 'monte-carlo':
            samples = self._monte_carlo_sampling(n_samples, parameters)
        elif method == 'importance-sampling':
            samples = self._importance_sampling(n_samples, parameters)
        elif method == 'mcmc':
            samples = self._mcmc_sampling(n_samples, parameters)
        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        # Generate sample IDs
        sample_ids = [f"sample_{i:06d}" for i in range(n_samples)]

        result = SampleResult(
            samples=samples,
            sample_ids=sample_ids,
            sampling_method=method,
            parameters=parameters
        )

        logger.info(f"Generated {n_samples} samples")
        return result

    def _monte_carlo_sampling(self,
                             n_samples: int,
                             parameters: Dict[str, Any]) -> np.ndarray:
        """Perform Monte Carlo sampling."""
        # Simple random sampling from population data
        if self.time_column in self.population_data.columns:
            # Time series data
            numeric_columns = self.population_data.select_dtypes(include=[np.number]).columns
            phenotype_columns = [col for col in numeric_columns
                               if col != self.time_column]

            n_timepoints = len(self.population_data[self.time_column].unique())
            n_phenotypes = len(phenotype_columns)

            samples = np.zeros((n_samples, n_timepoints, n_phenotypes))

            for i in range(n_samples):
                # Sample with replacement from each time point
                for j, time_point in enumerate(self.population_data[self.time_column].unique()):
                    time_data = self.population_data[
                        self.population_data[self.time_column] == time_point
                    ][phenotype_columns]

                    if len(time_data) > 0:
                        sample_idx = int(self._rng.integers(0, len(time_data)))
                        samples[i, j, :] = time_data.iloc[sample_idx].values
        else:
            # Cross-sectional data
            numeric_columns = self.population_data.select_dtypes(include=[np.number]).columns
            n_variables = len(numeric_columns)
            samples = np.zeros((n_samples, n_variables))

            for i in range(n_samples):
                sample_idx = int(self._rng.integers(0, len(self.population_data)))
                samples[i, :] = self.population_data.iloc[sample_idx][numeric_columns].values

        return samples

    def _importance_sampling(self,
                           n_samples: int,
                           parameters: Dict[str, Any]) -> np.ndarray:
        """Importance sampling with resampling from a tilted proposal.

        Proposals are drawn from the empirical population; weights are computed
        from an exponential tilt on the mean phenotype (temperature given by
        ``parameters.get('temperature', 1.0)``), then normalized. Weighted
        resampling (systematic) yields the final sample set; the effective
        sample size is recorded in ``parameters`` under 'ess'.
        """
        base = self._monte_carlo_sampling(n_samples, parameters)
        temperature = float(parameters.get('temperature', 1.0))

        # Weight each drawn sample by exp(temperature * standardized mean phenotype)
        if base.ndim == 3:
            scores = base.mean(axis=(1, 2))
        else:
            scores = base.mean(axis=1)
        scores = (scores - scores.mean()) / (scores.std() + 1e-12)
        log_w = temperature * scores
        w = np.exp(log_w - log_w.max())
        w /= w.sum()

        # Systematic resampling according to normalized weights
        positions = (np.arange(n_samples) + self._rng.uniform()) / n_samples
        cumulative = np.cumsum(w)
        idx = np.searchsorted(cumulative, positions)
        idx = np.clip(idx, 0, n_samples - 1)
        resampled = base[idx]

        # Record effective sample size for diagnostics
        ess = 1.0 / np.sum(w ** 2)
        parameters['ess'] = float(ess)
        return resampled

    def _mcmc_sampling(self,
                      n_samples: int,
                      parameters: Dict[str, Any]) -> np.ndarray:
        """Metropolis-Hastings MCMC over the empirical population statistic.

        State: index into the population. Target: unnormalized density implied
        by the empirical mean phenotype (Gaussian around observed mean with
        scale ``parameters.get('mcmc_scale', 1.0)``). Proposal: symmetric
        random walk on the index lattice with step ``parameters.get(
        'step_size', 0.5)`` of the pool size; wrap-around keeps the kernel
        symmetric so detailed balance holds exactly.
        Burn-in is 10% of n_samples; the reported acceptance rate excludes
        burn-in.
        """
        if self.time_column in self.population_data.columns:
            numeric_columns = self.population_data.select_dtypes(include=[np.number]).columns
            phenotype_columns = [c for c in numeric_columns if c != self.time_column]
        else:
            phenotype_columns = list(self.population_data.select_dtypes(include=[np.number]).columns)

        pool = self.population_data[phenotype_columns].dropna().values
        if pool.ndim == 1:
            pool = pool.reshape(-1, 1)

        step_size = float(parameters.get('step_size', 0.5))
        mcmc_scale = float(parameters.get('mcmc_scale', 1.0))
        burn_in = max(1, n_samples // 10)

        observed_mean = pool.mean(axis=0)
        observed_std = pool.std(axis=0) + 1e-12

        def log_target_idx(idx: int) -> float:
            x = pool[idx]
            return -0.5 * np.sum(((x - observed_mean) / (observed_std * mcmc_scale)) ** 2)

        # Discrete index-to-index Metropolis-Hastings: symmetric random-walk
        # proposal on indices with a Gaussian tilt. Both directions have
        # identical proposal probability q(j|i) = q(i|j) (symmetric kernel on
        # the index lattice), so detailed balance holds exactly -- unlike the
        # previous phenotype-space proposal with nearest-individual snapping,
        # which broke reversibility.
        idx_sigma = max(step_size * len(pool), 1.0)
        current_idx = int(self._rng.integers(0, len(pool)))
        samples = np.zeros((n_samples, pool.shape[1]))
        accepted = 0
        for i in range(n_samples + burn_in):
            proposal_idx = int(current_idx + round(self._rng.normal(0, idx_sigma)))
            proposal_idx = proposal_idx % len(pool)  # wrap: keeps q symmetric
            log_alpha = log_target_idx(proposal_idx) - log_target_idx(current_idx)
            if np.log(self._rng.uniform()) < log_alpha:
                current_idx = proposal_idx
                if i >= burn_in:
                    accepted += 1
            if i >= burn_in:
                samples[i - burn_in] = pool[current_idx]

        parameters['acceptance_rate'] = accepted / n_samples if n_samples > 0 else 0.0
        return samples

    def analyze_evolutionary_patterns(self) -> Dict[str, Any]:
        """
        Analyze evolutionary patterns in the population.

        Returns:
            Dictionary with evolutionary analysis results. The
            'phylogenetic_signal' entry is populated only when
            ``phylogenetic_analyzer.distance_matrix`` is set AND each trait
            vector has one finite value per matrix row, ordered consistently
            with the matrix rows (Moran's I pairs the i-th trait value with
            the i-th row/column of the matrix); otherwise it is left empty
            because the statistic is undefined, not zero.
        """
        logger.info("Analyzing evolutionary patterns")

        results = {
            'population_statistics': self._compute_population_statistics(),
            'phylogenetic_signal': {},
            'genetic_parameters': {},
            'selection_analysis': {}
        }

        # Compute phylogenetic signal for each trait via Moran's I, but only
        # when a distance matrix has been supplied whose rows correspond, in
        # order, to the trait vector. Without a matrix the statistic is
        # undefined (the analyzer would return np.nan unconditionally), so
        # the entry is skipped entirely.
        distance_matrix = self.phylogenetic_analyzer.distance_matrix
        if distance_matrix is not None:
            n_rows = distance_matrix.shape[0]
            for col in self.population_data.columns:
                if col == self.time_column:
                    continue
                trait_data = self.population_data[col].dropna().values
                if len(trait_data) == n_rows and len(trait_data) >= 3:
                    results['phylogenetic_signal'][col] = (
                        self.phylogenetic_analyzer.compute_morans_i_signal(trait_data))

        # Estimate genetic parameters
        genetic_params = self._estimate_genetic_parameters()
        results['genetic_parameters'] = genetic_params

        # Analyze selection
        selection_results = self._analyze_selection()
        results['selection_analysis'] = selection_results

        logger.info("Evolutionary pattern analysis completed")
        return results

    def _compute_population_statistics(self) -> PopulationStatistics:
        """Compute population-level statistics."""
        if self.time_column in self.population_data.columns:
            # Time series data
            phenotype_columns = [col for col in self.population_data.columns
                               if col != self.time_column]

            time_points = sorted(self.population_data[self.time_column].unique())
            n_timepoints = len(time_points)
            n_phenotypes = len(phenotype_columns)

            # Mean trajectory
            mean_trajectory = np.zeros((n_timepoints, n_phenotypes))
            variance_trajectory = np.zeros((n_timepoints, n_phenotypes))

            for i, time_point in enumerate(time_points):
                time_data = self.population_data[
                    self.population_data[self.time_column] == time_point
                ][phenotype_columns]

                mean_trajectory[i, :] = time_data.mean().values
                variance_trajectory[i, :] = time_data.var().values

            # Covariance matrix (simplified)
            covariance_matrix = np.cov(mean_trajectory.T)

            # Heritability estimates
            heritability_estimates = {}
            for col in phenotype_columns:
                heritability_estimates[col] = self.population_model.estimate_heritability(col)

            # Selection gradients require an explicit fitness measure; a
            # phenotype-only time series has none, so report np.nan rather
            # than a degenerate phenotype-vs-itself value (which is always
            # exactly 1.0). Use population_model.compute_selection_gradient(
            # phenotype, fitness) directly when a fitness column exists.
            selection_gradients = {col: np.nan for col in phenotype_columns}

            # Effective population size
            ne = self.population_model.estimate_effective_population_size()

            # Genetic correlations (placeholder)
            genetic_correlations = np.eye(n_phenotypes)

            return PopulationStatistics(
                mean_trajectory=mean_trajectory,
                variance_trajectory=variance_trajectory,
                covariance_matrix=covariance_matrix,
                heritability_estimates=heritability_estimates,
                selection_gradients=selection_gradients,
                effective_population_size=ne,
                genetic_correlations=genetic_correlations
            )
        else:
            # Cross-sectional data
            n_variables = len(self.population_data.columns)
            mean_values = self.population_data.mean(numeric_only=True).values
            variance_values = self.population_data.var(numeric_only=True).values
            covariance_matrix = self.population_data.cov(numeric_only=True).values

            return PopulationStatistics(
                mean_trajectory=mean_values.reshape(1, -1),
                variance_trajectory=variance_values.reshape(1, -1),
                covariance_matrix=covariance_matrix,
                heritability_estimates={},
                selection_gradients={},
                effective_population_size=len(self.population_data),
                genetic_correlations=np.eye(n_variables)
            )

    def _estimate_genetic_parameters(self) -> Dict[str, Any]:
        """Estimate genetic parameters from the phenotype time series.

        Additive and environmental variances follow from the measured
        phenotypic variance (mean within-time-point variance across traits)
        and the narrow-sense heritability estimated by the population model:
        V_A = h2 * V_P and V_E = (1 - h2) * V_P.

        Dominance and epistatic variances and broad-sense heritability cannot
        be identified from a phenotypic time series without family or clonal
        structure; they are always reported as np.nan. The 'available' flag
        records whether the additive/environmental split could be estimated
        at all (it requires a pedigree and replicated observations within at
        least one time point). No quantity is ever reported as a fabricated
        placeholder such as 0.0.
        """
        pedigree_columns = {'parent', 'offspring'}
        phenotype_columns = [col for col in self.population_data.columns
                             if col != self.time_column
                             and col not in pedigree_columns]

        # Phenotypic variance: mean within-time-point variance per trait,
        # averaged over traits. Undefined (np.nan) when no time point has
        # replicated observations.
        phenotypic_variance = np.nan
        if phenotype_columns and self.time_column in self.population_data.columns:
            per_time = (self.population_data
                        .groupby(self.time_column)[phenotype_columns]
                        .var())
            if len(per_time) > 0:
                phenotypic_variance = float(per_time.mean(axis=0).mean())

        # Narrow-sense heritability from the population model (requires the
        # explicit 'parent'/'offspring' pedigree; estimate_heritability warns
        # and returns np.nan otherwise).
        h2 = np.nan
        if phenotype_columns:
            h2 = self.population_model.estimate_heritability(phenotype_columns[0])

        available = bool(np.isfinite(phenotypic_variance) and np.isfinite(h2))
        if available:
            additive_variance = float(h2 * phenotypic_variance)
            environmental_variance = float((1.0 - h2) * phenotypic_variance)
            narrow_sense = float(h2)
        else:
            additive_variance = environmental_variance = narrow_sense = np.nan

        return {
            'additive_variance': additive_variance,
            'dominance_variance': np.nan,      # not identifiable from phenotypes alone
            'epistatic_variance': np.nan,      # not identifiable from phenotypes alone
            'environmental_variance': environmental_variance,
            'narrow_sense_heritability': narrow_sense,
            'broad_sense_heritability': np.nan,  # requires clonal/family structure
            'available': available,
        }

    def _analyze_selection(self) -> Dict[str, Any]:
        """Analyze selection patterns from measured phenotypic change.

        All components are estimated from the phenotype time series; none is
        fabricated:

        - 'selection_differential': per-trait change in mean phenotype
          between the first and last time points
          (``compute_selection_differential``).
        - 'selection_response': per-trait Lande response R = h2 * S via
          ``predict_phenotypic_response``, using the heritability estimated
          by the population model (np.nan when h2 is unavailable).
        - 'directional_selection': mean over traits of the standardized
          differential S / SD(first time point).
        - 'stabilizing_selection' / 'disruptive_selection': mean proportional
          decrease / increase in phenotypic variance between the first and
          last time points.

        Entries are np.nan (or empty per-trait dicts) when the underlying
        quantity cannot be measured, e.g. fewer than two time points or zero
        phenotypic variance.
        """
        pedigree_columns = {'parent', 'offspring'}
        phenotype_columns = [col for col in self.population_data.columns
                             if col != self.time_column
                             and col not in pedigree_columns]

        not_measurable: Dict[str, Any] = {
            'directional_selection': np.nan,
            'stabilizing_selection': np.nan,
            'disruptive_selection': np.nan,
            'selection_differential': {},
            'selection_response': {},
        }
        if (not phenotype_columns
                or self.time_column not in self.population_data.columns):
            return not_measurable

        times = sorted(self.population_data[self.time_column].unique())
        if len(times) < 2:
            return not_measurable
        first = self.population_data[
            self.population_data[self.time_column] == times[0]][phenotype_columns]
        last = self.population_data[
            self.population_data[self.time_column] == times[-1]][phenotype_columns]

        h2 = self.population_model.estimate_heritability(phenotype_columns[0])

        differentials: Dict[str, float] = {}
        responses: Dict[str, float] = {}
        directional: List[float] = []
        for col in phenotype_columns:
            s = self.population_model.compute_selection_differential(col)
            differentials[col] = s
            if np.isfinite(h2):
                responses[col] = self.population_model.predict_phenotypic_response(col, h2)
            else:
                responses[col] = np.nan
            sd_first = float(first[col].std())
            if np.isfinite(s) and sd_first > 0:
                directional.append(s / sd_first)

        var_first = first.var()
        var_last = last.var()
        stabilizing: List[float] = []
        disruptive: List[float] = []
        for col in phenotype_columns:
            v0 = float(var_first[col])
            v1 = float(var_last[col])
            if np.isfinite(v0) and np.isfinite(v1) and v0 > 0:
                proportional_change = (v1 - v0) / v0
                if proportional_change < 0:
                    stabilizing.append(-proportional_change)
                elif proportional_change > 0:
                    disruptive.append(proportional_change)

        return {
            'directional_selection': float(np.mean(directional)) if directional else np.nan,
            'stabilizing_selection': float(np.mean(stabilizing)) if stabilizing else np.nan,
            'disruptive_selection': float(np.mean(disruptive)) if disruptive else np.nan,
            'selection_differential': differentials,
            'selection_response': responses,
        }

    def cluster_individuals(self, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Cluster individuals based on their developmental trajectories.

        Parameters:
            n_clusters: Number of clusters

        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Clustering individuals into {n_clusters} groups")

        if self.time_column not in self.population_data.columns:
            raise ValueError("Clustering requires time series data")

        phenotype_columns = [col for col in self.population_data.columns
                           if col != self.time_column]

        # Prepare data for clustering
        # Use final time point values for simplicity
        final_time = self.population_data[self.time_column].max()
        final_data = self.population_data[
            self.population_data[self.time_column] == final_time
        ][phenotype_columns]

        if len(final_data) < n_clusters:
            raise ValueError("Not enough data points for clustering")

        # Perform clustering
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(final_data)

        # Compute cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = final_data[cluster_labels == i]
            cluster_stats[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict(),
                'cov': cluster_data.cov().to_dict()
            }

        results = {
            'cluster_labels': cluster_labels,
            'cluster_statistics': cluster_stats,
            'gmm_parameters': {
                'means': gmm.means_,
                'covariances': gmm.covariances_,
                'weights': gmm.weights_
            }
        }

        logger.info(f"Clustering completed: {n_clusters} clusters identified")
        return results
