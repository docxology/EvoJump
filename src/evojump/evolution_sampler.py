"""
Evolution Sampler: Population-Level Analysis

This module handles population-level analysis by sampling multiple developmental trajectories
and performing comparative evolutionary analysis. Implements phylogenetic comparative methods,
quantitative genetics approaches, and population dynamics modeling.

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
            # Simple parent-offspring regression
            if len(self.individuals) < 4:
                return np.nan

            # This is a simplified implementation
            # In practice, would need pedigree information
            parent_values = self.population_data[phenotype].iloc[:-1]
            offspring_values = self.population_data[phenotype].iloc[1:]

            if len(parent_values) != len(offspring_values):
                parent_values = parent_values[:len(offspring_values)]

            slope, _, _, _, _ = stats.linregress(parent_values, offspring_values)
            heritability = 2 * slope if slope > 0 else 0.0

            return min(heritability, 1.0)  # Cap at 1.0

        else:
            raise ValueError(f"Unsupported heritability method: {method}")

    def compute_selection_gradient(self, phenotype: str, fitness_measure: str) -> float:
        """
        Compute selection gradient for a phenotype.

        Parameters:
            phenotype: Name of phenotype column
            fitness_measure: Name of fitness column

        Returns:
            Selection gradient
        """
        if phenotype not in self.population_data.columns or fitness_measure not in self.population_data.columns:
            return np.nan

        # Standardize variables
        pheno_std = (self.population_data[phenotype] - self.population_data[phenotype].mean()) / self.population_data[phenotype].std()
        fitness_std = (self.population_data[fitness_measure] - self.population_data[fitness_measure].mean()) / self.population_data[fitness_measure].std()

        # Compute covariance
        covariance = np.cov(pheno_std, fitness_std)[0, 1]

        return covariance

    def estimate_effective_population_size(self, method: str = 'temporal') -> float:
        """
        Estimate effective population size.

        Parameters:
            method: Estimation method

        Returns:
            Effective population size estimate
        """
        if method == 'temporal':
            # Temporal method using allele frequency changes
            # This is a simplified implementation
            n_individuals = len(self.individuals)

            # Assume we can estimate from phenotypic variance
            # This would typically use genetic markers
            phenotypic_variance = self.population_data.select_dtypes(include=[np.number]).var().mean()

            # Rough approximation
            ne_estimate = n_individuals * 0.5  # Conservative estimate

            return ne_estimate

        else:
            raise ValueError(f"Unsupported method: {method}")


class PhylogeneticAnalyzer:
    """Performs phylogenetic comparative analysis."""

    def __init__(self, distance_matrix: Optional[np.ndarray] = None):
        """Initialize phylogenetic analyzer."""
        self.distance_matrix = distance_matrix
        self.phylogeny: Optional[Any] = None

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

            # Compute phylogenetic variance-covariance matrix
            # For simplicity, use Brownian motion model
            phylo_matrix = self._compute_brownian_covariance(self.distance_matrix)

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
                except:
                    return np.inf

            # Optimize lambda
            result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
            return result.x if result.success else 0.0

        else:
            raise ValueError(f"Unsupported method: {method}")

    def _compute_brownian_covariance(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Compute Brownian motion covariance matrix from distance matrix."""
        n = distance_matrix.shape[0]
        covariance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                covariance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[j, i])

        return covariance_matrix

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

        except:
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
                            except:
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

        logger.info("Initialized Evolution Sampler")

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

        parameters = parameters or {}

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
                        sample_idx = np.random.randint(0, len(time_data))
                        samples[i, j, :] = time_data.iloc[sample_idx].values
        else:
            # Cross-sectional data
            n_variables = len(self.population_data.columns)
            samples = np.zeros((n_samples, n_variables))

            for i in range(n_samples):
                sample_idx = np.random.randint(0, len(self.population_data))
                samples[i, :] = self.population_data.iloc[sample_idx].values

        return samples

    def _importance_sampling(self,
                           n_samples: int,
                           parameters: Dict[str, Any]) -> np.ndarray:
        """Perform importance sampling."""
        # Simplified importance sampling
        # In practice, this would use importance weights
        return self._monte_carlo_sampling(n_samples, parameters)

    def _mcmc_sampling(self,
                      n_samples: int,
                      parameters: Dict[str, Any]) -> np.ndarray:
        """Perform Markov Chain Monte Carlo sampling."""
        # Simplified MCMC
        # In practice, this would implement proper MCMC chains
        return self._monte_carlo_sampling(n_samples, parameters)

    def analyze_evolutionary_patterns(self) -> Dict[str, Any]:
        """
        Analyze evolutionary patterns in the population.

        Returns:
            Dictionary with evolutionary analysis results
        """
        logger.info("Analyzing evolutionary patterns")

        results = {
            'population_statistics': self._compute_population_statistics(),
            'phylogenetic_signal': {},
            'genetic_parameters': {},
            'selection_analysis': {}
        }

        # Compute phylogenetic signal for each trait
        if self.time_column in self.population_data.columns:
            for col in self.population_data.columns:
                if col != self.time_column:
                    trait_data = self.population_data[col].dropna()
                    if len(trait_data) >= 3:
                        # This is a placeholder - would need actual phylogenetic data
                        signal = 0.0  # Placeholder
                        results['phylogenetic_signal'][col] = signal

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

            # Selection gradients
            selection_gradients = {}
            for col in phenotype_columns:
                # Placeholder fitness measure
                selection_gradients[col] = self.population_model.compute_selection_gradient(col, col)

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
            mean_values = self.population_data.mean().values
            variance_values = self.population_data.var().values
            covariance_matrix = self.population_data.cov().values

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
        """Estimate genetic parameters."""
        # Placeholder implementation
        return {
            'additive_variance': 0.0,
            'dominance_variance': 0.0,
            'epistatic_variance': 0.0,
            'environmental_variance': 0.0,
            'narrow_sense_heritability': 0.0,
            'broad_sense_heritability': 0.0
        }

    def _analyze_selection(self) -> Dict[str, Any]:
        """Analyze selection patterns."""
        # Placeholder implementation
        return {
            'directional_selection': 0.0,
            'stabilizing_selection': 0.0,
            'disruptive_selection': 0.0,
            'selection_differential': 0.0,
            'selection_response': 0.0
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
