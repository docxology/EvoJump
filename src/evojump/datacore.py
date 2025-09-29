"""
DataCore Module: Data Ingestion, Validation, and Preprocessing

This module handles ontogenetic time series data ingestion, validation, and preprocessing.
It supports multiple data formats including time-stamped phenotypic measurements,
gene expression profiles, and morphometric data. The module implements robust data
structures capable of managing longitudinal datasets with varying temporal resolutions
and missing data points.

Classes:
    DataCore: Main class for data management and preprocessing
    TimeSeriesData: Container for time series phenotypic data
    MetadataManager: Handles experimental metadata and provenance

Examples:
    >>> # Load data from CSV
    >>> data = DataCore.load_from_csv("data.csv", time_column="time")
    >>> # Preprocess data
    >>> data.preprocess_data()
    >>> # Validate data quality
    >>> data.validate_data_quality()
"""

import pandas as pd
import numpy as np
import h5py
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Container for time series phenotypic data."""
    data: pd.DataFrame
    time_column: str
    phenotype_columns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: Optional[str] = None
    temporal_resolution: Optional[float] = None

    def __post_init__(self):
        """Validate data structure after initialization."""
        if self.time_column not in self.data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")

        missing_cols = [col for col in self.phenotype_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Phenotype columns not found: {missing_cols}")

    @property
    def time_points(self) -> np.ndarray:
        """Get unique time points."""
        return np.sort(self.data[self.time_column].unique())

    @property
    def n_timepoints(self) -> int:
        """Get number of time points."""
        return len(self.time_points)

    @property
    def n_phenotypes(self) -> int:
        """Get number of phenotype measurements."""
        return len(self.phenotype_columns)

    def get_phenotype_at_time(self, time_point: float) -> pd.DataFrame:
        """Get phenotypic data at a specific time point."""
        mask = self.data[self.time_column] == time_point
        return self.data.loc[mask, self.phenotype_columns]

    def interpolate_missing_data(self, method: str = 'linear') -> None:
        """Interpolate missing data points."""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        # Interpolate only numeric columns
        for col in numeric_columns:
            if col != self.time_column:
                self.data[col] = self.data[col].interpolate(method=method)

        # Forward fill remaining missing values
        self.data[numeric_columns] = self.data[numeric_columns].fillna(method='ffill')

        logger.info(f"Interpolated missing data using {method} method")


class MetadataManager:
    """Handles experimental metadata and provenance tracking."""

    def __init__(self, metadata_file: Optional[Path] = None):
        """Initialize metadata manager."""
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'version': '0.1.0',
            'source': None,
            'experimental_conditions': {},
            'genotype_info': {},
            'measurement_protocols': {},
            'processing_history': []
        }

        if metadata_file:
            self.load_metadata(metadata_file)

    def add_processing_step(self, step: str, parameters: Dict[str, Any]) -> None:
        """Add a processing step to the history."""
        processing_step = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters
        }
        self.metadata['processing_history'].append(processing_step)
        logger.info(f"Added processing step: {step}")

    def load_metadata(self, metadata_file: Path) -> None:
        """Load metadata from file."""
        if metadata_file.suffix.lower() in ['.yaml', '.yml']:
            with open(metadata_file, 'r') as f:
                loaded_metadata = yaml.safe_load(f)
        elif metadata_file.suffix.lower() == '.json':
            with open(metadata_file, 'r') as f:
                loaded_metadata = json.load(f)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_file.suffix}")

        # Merge with existing metadata
        self.metadata.update(loaded_metadata)
        logger.info(f"Loaded metadata from {metadata_file}")

    def save_metadata(self, metadata_file: Path) -> None:
        """Save metadata to file."""
        if metadata_file.suffix.lower() in ['.yaml', '.yml']:
            with open(metadata_file, 'w') as f:
                yaml.dump(self.metadata, f, default_flow_style=False)
        elif metadata_file.suffix.lower() == '.json':
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_file.suffix}")

        logger.info(f"Saved metadata to {metadata_file}")


class DataCore:
    """Main class for data management and preprocessing."""

    def __init__(self,
                 time_series_data: List[TimeSeriesData],
                 metadata_manager: Optional[MetadataManager] = None):
        """Initialize DataCore with time series data."""
        self.time_series_data = time_series_data
        self.metadata_manager = metadata_manager or MetadataManager()

        # Validate data consistency
        self._validate_data_consistency()

        logger.info(f"Initialized DataCore with {len(time_series_data)} time series datasets")

    @classmethod
    def load_from_csv(cls,
                     file_path: Path,
                     time_column: str = 'time',
                     phenotype_columns: Optional[List[str]] = None,
                     metadata_file: Optional[Path] = None,
                     **kwargs) -> 'DataCore':
        """
        Load data from CSV file.

        Parameters:
            file_path: Path to CSV file
            time_column: Name of time column
            phenotype_columns: List of phenotype column names
            metadata_file: Optional metadata file
            **kwargs: Additional arguments for pandas.read_csv()

        Returns:
            DataCore instance
        """
        logger.info(f"Loading data from {file_path}")

        # Load raw data
        raw_data = pd.read_csv(file_path, **kwargs)

        # Auto-detect phenotype columns if not specified
        if phenotype_columns is None:
            # Assume all numeric columns except time are phenotypes
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            phenotype_columns = [col for col in numeric_cols if col != time_column]

        # Create TimeSeriesData object
        time_series = TimeSeriesData(
            data=raw_data,
            time_column=time_column,
            phenotype_columns=phenotype_columns
        )

        # Load metadata if provided
        metadata_manager = None
        if metadata_file:
            metadata_manager = MetadataManager(metadata_file)

        # Create DataCore instance
        instance = cls([time_series], metadata_manager)

        # Add loading step to processing history
        instance.metadata_manager.add_processing_step(
            'load_from_csv',
            {
                'file_path': str(file_path),
                'time_column': time_column,
                'phenotype_columns': phenotype_columns
            }
        )

        return instance

    @classmethod
    def load_from_hdf5(cls,
                      file_path: Path,
                      time_column: str = 'time',
                      phenotype_columns: Optional[List[str]] = None,
                      metadata_file: Optional[Path] = None) -> 'DataCore':
        """
        Load data from HDF5 file.

        Parameters:
            file_path: Path to HDF5 file
            time_column: Name of time column
            phenotype_columns: List of phenotype column names
            metadata_file: Optional metadata file

        Returns:
            DataCore instance
        """
        logger.info(f"Loading data from {file_path}")

        with h5py.File(file_path, 'r') as f:
            # Load data
            data_dict = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data_dict[key] = f[key][:]
                elif isinstance(f[key], h5py.Group):
                    # Handle groups if needed
                    for subkey in f[key].keys():
                        data_dict[f"{key}/{subkey}"] = f[key][subkey][:]

            raw_data = pd.DataFrame(data_dict)

        # Auto-detect phenotype columns if not specified
        if phenotype_columns is None:
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            phenotype_columns = [col for col in numeric_cols if col != time_column]

        # Create TimeSeriesData object
        time_series = TimeSeriesData(
            data=raw_data,
            time_column=time_column,
            phenotype_columns=phenotype_columns
        )

        # Load metadata if provided
        metadata_manager = None
        if metadata_file:
            metadata_manager = MetadataManager(metadata_file)

        # Create DataCore instance
        instance = cls([time_series], metadata_manager)

        # Add loading step to processing history
        instance.metadata_manager.add_processing_step(
            'load_from_hdf5',
            {
                'file_path': str(file_path),
                'time_column': time_column,
                'phenotype_columns': phenotype_columns
            }
        )

        return instance

    def _validate_data_consistency(self) -> None:
        """Validate consistency across time series datasets."""
        if not self.time_series_data:
            raise ValueError("No time series data provided")

        # Check that time columns are consistent
        time_columns = [ts.time_column for ts in self.time_series_data]
        if len(set(time_columns)) > 1:
            raise ValueError("Inconsistent time column names across datasets")

        # Check temporal overlap
        all_time_points = []
        for ts in self.time_series_data:
            all_time_points.extend(ts.time_points)

        if len(set(all_time_points)) != len(all_time_points):
            warnings.warn("Duplicate time points found across datasets")

        logger.info("Data consistency validation completed")

    def preprocess_data(self,
                       normalize: bool = True,
                       remove_outliers: bool = True,
                       interpolate_missing: bool = True,
                       **kwargs) -> None:
        """
        Preprocess all time series data.

        Parameters:
            normalize: Whether to normalize phenotypic data
            remove_outliers: Whether to remove outliers
            interpolate_missing: Whether to interpolate missing data
            **kwargs: Additional preprocessing parameters
        """
        logger.info("Starting data preprocessing")

        for i, ts in enumerate(self.time_series_data):
            logger.info(f"Preprocessing dataset {i+1}/{len(self.time_series_data)}")

            # Interpolate missing data
            if interpolate_missing:
                ts.interpolate_missing_data(method=kwargs.get('interpolation_method', 'linear'))

            # Remove outliers
            if remove_outliers:
                self._remove_outliers(ts)

            # Normalize data
            if normalize:
                self._normalize_data(ts)

        # Add preprocessing step to history
        self.metadata_manager.add_processing_step(
            'preprocess_data',
            {
                'normalize': normalize,
                'remove_outliers': remove_outliers,
                'interpolate_missing': interpolate_missing,
                **kwargs
            }
        )

        logger.info("Data preprocessing completed")

    def _remove_outliers(self, ts: TimeSeriesData, method: str = 'iqr', threshold: float = 1.5) -> None:
        """Remove outliers from time series data."""
        for col in ts.phenotype_columns:
            if method == 'iqr':
                Q1 = ts.data[col].quantile(0.25)
                Q3 = ts.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                mask = (ts.data[col] >= lower_bound) & (ts.data[col] <= upper_bound)
                ts.data = ts.data[mask].copy()

            elif method == 'zscore':
                z_scores = np.abs((ts.data[col] - ts.data[col].mean()) / ts.data[col].std())
                mask = z_scores <= threshold
                ts.data = ts.data[mask].copy()

        logger.info(f"Removed outliers using {method} method with threshold {threshold}")

    def _normalize_data(self, ts: TimeSeriesData, method: str = 'zscore') -> None:
        """Normalize phenotypic data."""
        for col in ts.phenotype_columns:
            if method == 'zscore':
                mean_val = ts.data[col].mean()
                std_val = ts.data[col].std()
                if std_val > 0:
                    ts.data[col] = (ts.data[col] - mean_val) / std_val
            elif method == 'minmax':
                min_val = ts.data[col].min()
                max_val = ts.data[col].max()
                if max_val > min_val:
                    ts.data[col] = (ts.data[col] - min_val) / (max_val - min_val)
            elif method == 'robust':
                median_val = ts.data[col].median()
                mad_val = np.median(np.abs(ts.data[col] - median_val))
                if mad_val > 0:
                    ts.data[col] = (ts.data[col] - median_val) / mad_val

        logger.info(f"Normalized data using {method} method")

    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.

        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'n_datasets': len(self.time_series_data),
            'total_samples': sum(len(ts.data) for ts in self.time_series_data),
            'missing_data_percentage': {},
            'outlier_percentage': {},
            'temporal_consistency': {}
        }

        for i, ts in enumerate(self.time_series_data):
            # Missing data percentage
            missing_pct = ts.data.isnull().sum().sum() / (ts.data.shape[0] * ts.data.shape[1]) * 100
            quality_metrics['missing_data_percentage'][f'dataset_{i}'] = missing_pct

            # Outlier detection (using IQR method)
            outliers = 0
            total_values = 0
            for col in ts.phenotype_columns:
                Q1 = ts.data[col].quantile(0.25)
                Q3 = ts.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                col_outliers = ((ts.data[col] < lower_bound) | (ts.data[col] > upper_bound)).sum()
                outliers += col_outliers
                total_values += len(ts.data)

            outlier_pct = outliers / total_values * 100 if total_values > 0 else 0
            quality_metrics['outlier_percentage'][f'dataset_{i}'] = outlier_pct

            # Temporal consistency
            time_diffs = np.diff(np.sort(ts.data[ts.time_column].unique()))
            if len(time_diffs) > 0:
                mean_diff = np.mean(time_diffs)
                std_diff = np.std(time_diffs)
                quality_metrics['temporal_consistency'][f'dataset_{i}'] = {
                    'mean_interval': mean_diff,
                    'std_interval': std_diff,
                    'regularity_score': 1 - min(std_diff / mean_diff, 1) if mean_diff > 0 else 0
                }

        # Add validation step to history
        self.metadata_manager.add_processing_step(
            'validate_data_quality',
            quality_metrics
        )

        logger.info("Data quality validation completed")
        return quality_metrics

    def save_processed_data(self, output_path: Path, format: str = 'csv') -> None:
        """
        Save processed data to file.

        Parameters:
            output_path: Path to save data
            format: Output format ('csv', 'hdf5', 'parquet')
        """
        if format == 'csv':
            # Combine all datasets
            combined_data = pd.concat([ts.data for ts in self.time_series_data], ignore_index=True)
            combined_data.to_csv(output_path, index=False)
        elif format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                for i, ts in enumerate(self.time_series_data):
                    group = f.create_group(f'dataset_{i}')
                    for col in ts.data.columns:
                        group.create_dataset(col, data=ts.data[col].values)
        elif format == 'parquet':
            combined_data = pd.concat([ts.data for ts in self.time_series_data], ignore_index=True)
            combined_data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved processed data to {output_path} in {format} format")

    def get_aggregated_data(self, aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        Aggregate data across time series.

        Parameters:
            aggregation_method: Method for aggregating across datasets

        Returns:
            Aggregated DataFrame
        """
        if aggregation_method == 'mean':
            # Simple averaging across datasets at each time point
            all_data = []
            for ts in self.time_series_data:
                ts_copy = ts.data.copy()
                ts_copy['dataset_id'] = id(ts)  # Simple identifier
                all_data.append(ts_copy)

            combined = pd.concat(all_data, ignore_index=True)

            # Group by time and compute means
            time_col = self.time_series_data[0].time_column
            phenotype_cols = self.time_series_data[0].phenotype_columns

            aggregated = combined.groupby(time_col)[phenotype_cols].mean().reset_index()
            return aggregated

        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    def filter_by_time_range(self, min_time: float, max_time: float) -> None:
        """Filter all datasets to a specific time range."""
        for ts in self.time_series_data:
            mask = (ts.data[ts.time_column] >= min_time) & (ts.data[ts.time_column] <= max_time)
            ts.data = ts.data[mask].copy()

        logger.info(f"Filtered data to time range [{min_time}, {max_time}]")

    def filter_by_phenotype_range(self, phenotype_column: str, min_val: float, max_val: float) -> None:
        """Filter all datasets by phenotype value range."""
        for ts in self.time_series_data:
            if phenotype_column in ts.phenotype_columns:
                mask = (ts.data[phenotype_column] >= min_val) & (ts.data[phenotype_column] <= max_val)
                ts.data = ts.data[mask].copy()

        logger.info(f"Filtered data by {phenotype_column} range [{min_val}, {max_val}]")
