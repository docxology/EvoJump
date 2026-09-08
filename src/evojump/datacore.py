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
        """Interpolate missing data points.

        Rows are first ordered by the time column so interpolation is temporal
        rather than row-order dependent; the original row order is restored
        positionally afterwards (duplicate index labels are safe). Forward
        fill only applies to leading/trailing gaps of phenotype columns after
        interpolation; the time column itself is never filled. Missing time
        values raise a ValueError because temporal position cannot be invented.

        Parameters:
            method: Interpolation method passed to pandas (default 'linear')

        Raises:
            ValueError: If the time column contains missing values
        """
        if self.data[self.time_column].isna().any():
            raise ValueError(
                f"Time column '{self.time_column}' contains missing values; "
                "temporal position cannot be interpolated. Drop or fix these "
                "rows first."
            )

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        fill_columns = [col for col in numeric_columns if col != self.time_column]
        if not fill_columns:
            return

        # Positional stable sort by time, positional restore afterwards. A
        # label-based restore (.loc[original_index]) would take the cross
        # product of duplicated index labels and silently expand the frame.
        order = np.argsort(self.data[self.time_column].to_numpy(), kind='stable')
        inverse_order = np.argsort(order)

        sorted_data = self.data.iloc[order]
        for col in fill_columns:
            sorted_data[col] = sorted_data[col].interpolate(method=method)
        # ffill/bfill only the boundary gaps of the phenotype columns
        sorted_data[fill_columns] = sorted_data[fill_columns].ffill().bfill()

        self.data[fill_columns] = sorted_data[fill_columns].iloc[inverse_order].to_numpy()
        logger.info(f"Interpolated missing data using {method} method")


class MetadataManager:
    """Handles experimental metadata and provenance tracking."""

    def __init__(self, metadata_file: Optional[Path] = None):
        """Initialize metadata manager."""
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'version': '0.2.0',
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

        # Merge with existing metadata (yaml.safe_load returns None for an
        # empty file; update() would fail with an opaque TypeError on None)
        if loaded_metadata is None:
            raise ValueError(f"Metadata file {metadata_file} is empty")
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
                 time_series_data: Union[TimeSeriesData, List[TimeSeriesData]],
                 metadata_manager: Optional[MetadataManager] = None):
        """Initialize DataCore with one or more time series datasets.

        A single TimeSeriesData is accepted for convenience and normalized to
        a one-element list. Multi-dataset DataCores can be built either by
        passing a list of manually constructed TimeSeriesData objects or by
        appending datasets with append().
        """
        if isinstance(time_series_data, TimeSeriesData):
            time_series_data = [time_series_data]
        self.time_series_data = list(time_series_data)
        self.metadata_manager = metadata_manager or MetadataManager()

        # Validate data consistency
        self._validate_data_consistency()

        logger.info(f"Initialized DataCore with {len(self.time_series_data)} time series datasets")

    def append(self, time_series: TimeSeriesData) -> None:
        """Append a time series dataset and re-validate consistency."""
        self.time_series_data.append(time_series)
        self._validate_data_consistency()
        logger.info(f"Appended time series dataset (now {len(self.time_series_data)} datasets)")

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

        Both layouts are supported and detected automatically:

        - the layout written by ``save_processed_data(format='hdf5')``:
          one ``dataset_<i>`` group per stored series, each group becoming
          one ``TimeSeriesData`` (multi-series files round-trip), and
        - flat files where each top-level dataset is a column (or a group
          of per-column datasets, flattened as ``group/subkey``).

        Parameters:
            file_path: Path to HDF5 file
            time_column: Name of time column
            phenotype_columns: List of phenotype column names
            metadata_file: Optional metadata file

        Returns:
            DataCore instance

        Raises:
            ValueError: If column datasets have unequal lengths or no
                data columns are found
        """
        logger.info(f"Loading data from {file_path}")

        def column_name(key: str) -> str:
            # Flattened group paths cannot be positional column names
            return key.replace('/', '_')

        def to_series(node: h5py.Dataset, key: str) -> pd.Series:
            values = node[:]
            if node.dtype.kind == 'S':
                values = np.char.decode(values, 'utf-8')
            elif node.dtype == object and values.size > 0:
                values = np.array([
                    v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v
                    for v in values.ravel()
                ], dtype=object).reshape(values.shape)
            return pd.Series(values, name=column_name(key))

        datasets: Dict[str, pd.Series] = {}

        with h5py.File(file_path, 'r') as f:
            def is_series_group(key: str) -> bool:
                """True for the 'dataset_<i>' groups written by save_processed_data."""
                node = f[key]
                label = key.rsplit('_', 1)[-1]
                return (
                    key.startswith('dataset_') and label.isdigit()
                    and isinstance(node, h5py.Group)
                )

            keys = list(f.keys())
            save_layout = bool(keys) and all(is_series_group(key) for key in keys)
            dataset_groups = [key for key in keys if is_series_group(key)]
            if save_layout:
                # save_processed_data layout: one group per series
                frames = []
                for group_name in sorted(dataset_groups, key=lambda k: int(k.rsplit('_', 1)[-1])):
                    group = f[group_name]
                    group_data = {}
                    for subkey in group.keys():
                        node = group[subkey]
                        if isinstance(node, h5py.Dataset):
                            group_data[column_name(subkey)] = to_series(node, subkey)
                    lengths = {len(s) for s in group_data.values()}
                    if len(lengths) > 1:
                        raise ValueError(
                            f"Cannot load {file_path}: group '{group_name}' has "
                            f"datasets of unequal lengths "
                            f"{ {k: len(v) for k, v in group_data.items()} }; "
                            "columns of one series must share a length"
                        )
                    frames.append(pd.DataFrame(group_data))
                raw_frames = frames
            else:
                # Flat layout: top-level datasets (and groups) are columns
                def collect(prefix: str, group: h5py.Group) -> None:
                    for key in group.keys():
                        node = group[key]
                        if isinstance(node, h5py.Dataset):
                            datasets[column_name(f"{prefix}{key}")] = to_series(node, f"{prefix}{key}")
                        elif isinstance(node, h5py.Group):
                            collect(f"{prefix}{key}/", node)

                collect('', f)
                lengths = {len(s) for s in datasets.values()}
                if len(lengths) > 1:
                    raise ValueError(
                        f"Cannot load {file_path}: datasets have unequal lengths "
                        f"{ {k: len(v) for k, v in datasets.items()} }; "
                        "a flat HDF5 file must store equal-length column arrays"
                    )
                if not datasets:
                    raise ValueError(f"No data columns found in {file_path}")
                raw_frames = [pd.DataFrame(datasets)]

        raw_frames = [frame for frame in raw_frames if len(frame.columns) > 0]
        time_series_list = []
        for raw_data in raw_frames:
            # Auto-detect phenotype columns if not specified
            if phenotype_columns is None:
                numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
                frame_phenotype_columns = [col for col in numeric_cols if col != time_column]
            else:
                frame_phenotype_columns = phenotype_columns

            time_series_list.append(TimeSeriesData(
                data=raw_data,
                time_column=time_column,
                phenotype_columns=frame_phenotype_columns
            ))

        # Load metadata if provided
        metadata_manager = None
        if metadata_file:
            metadata_manager = MetadataManager(metadata_file)

        # Create DataCore instance
        instance = cls(time_series_list, metadata_manager)

        # Add loading step to processing history
        instance.metadata_manager.add_processing_step(
            'load_from_hdf5',
            {
                'file_path': str(file_path),
                'time_column': time_column,
                'phenotype_columns': phenotype_columns,
                'n_datasets': len(time_series_list)
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
        """Remove outlier rows (row deletion) from time series data.

        One combined keep-mask is computed across all phenotype columns from
        the original data and applied once, so the result is independent of
        the phenotype column order. Missing (NaN) values are never treated as
        outliers. Note that removal deletes rows, which breaks temporal
        contiguity in a time series; run interpolation first when continuity
        matters.

        Raises:
            ValueError: If an unsupported method is requested, or if outlier
                removal would delete every row.
        """
        if method not in ('iqr', 'zscore'):
            raise ValueError(f"Unsupported outlier method: {method}")

        # Positional mask: safe with duplicate index labels
        keep = np.ones(len(ts.data), dtype=bool)
        for col in ts.phenotype_columns:
            series = ts.data[col]
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                col_keep = ((series >= lower_bound) & (series <= upper_bound)).to_numpy()
            else:  # zscore
                std_val = series.std()
                if pd.isna(std_val) or std_val == 0:
                    # Constant or degenerate column: nothing to flag
                    continue
                z_scores = (series - series.mean()) / std_val
                col_keep = (z_scores.abs() <= threshold).to_numpy()
            # NaN fails every bound comparison; treat as "not known to be an
            # outlier" instead of silently deleting the row
            keep &= col_keep | series.isna().to_numpy()

        filtered = ts.data[keep].copy()
        if filtered.empty and not ts.data.empty:
            raise ValueError(
                f"Outlier removal (method={method}, threshold={threshold}) would "
                "remove every row; refusing to empty the dataset. Increase the "
                "threshold or inspect the data."
            )

        ts.data = filtered
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
            'outliers_by_column': {},
            'temporal_consistency': {}
        }

        for i, ts in enumerate(self.time_series_data):
            # Missing data percentage
            missing_pct = ts.data.isnull().sum().sum() / (ts.data.shape[0] * ts.data.shape[1]) * 100
            quality_metrics['missing_data_percentage'][f'dataset_{i}'] = missing_pct

            # Outlier detection (using IQR method), with a per-column
            # breakdown so an affected column can be identified
            outliers = 0
            total_values = 0
            outliers_by_column = {}
            for col in ts.phenotype_columns:
                Q1 = ts.data[col].quantile(0.25)
                Q3 = ts.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                col_outliers = ((ts.data[col] < lower_bound) | (ts.data[col] > upper_bound)).sum()
                outliers += col_outliers
                total_values += len(ts.data)
                outliers_by_column[col] = (
                    col_outliers / len(ts.data) * 100 if len(ts.data) > 0 else 0
                )

            outlier_pct = outliers / total_values * 100 if total_values > 0 else 0
            quality_metrics['outlier_percentage'][f'dataset_{i}'] = outlier_pct
            quality_metrics['outliers_by_column'][f'dataset_{i}'] = outliers_by_column

            # Temporal consistency — always emitted, even for a single time
            # point, so consumers see a consistent key set per dataset
            time_diffs = np.diff(np.sort(ts.data[ts.time_column].unique()))
            if len(time_diffs) > 0:
                mean_diff = np.mean(time_diffs)
                std_diff = np.std(time_diffs)
                quality_metrics['temporal_consistency'][f'dataset_{i}'] = {
                    'mean_interval': mean_diff,
                    'std_interval': std_diff,
                    'regularity_score': 1 - min(std_diff / mean_diff, 1) if mean_diff > 0 else 0
                }
            else:
                quality_metrics['temporal_consistency'][f'dataset_{i}'] = {
                    'mean_interval': None,
                    'std_interval': None,
                    'regularity_score': None
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
                        values = ts.data[col]
                        if pd.api.types.is_numeric_dtype(values):
                            group.create_dataset(col, data=values.to_numpy())
                        else:
                            # Non-numeric columns (e.g. strain labels) are
                            # encoded as UTF-8 strings so the save never fails
                            logger.warning(
                                f"Encoding non-numeric column '{col}' as strings in HDF5 output")
                            group.create_dataset(
                                col,
                                data=values.astype(str).to_numpy(),
                                dtype=h5py.string_dtype(encoding='utf-8'))
        elif format == 'parquet':
            combined_data = pd.concat([ts.data for ts in self.time_series_data], ignore_index=True)
            try:
                combined_data.to_parquet(output_path, index=False)
            except ImportError as exc:
                raise ValueError(
                    "Parquet output requires pyarrow or fastparquet "
                    f"(install one of them): {exc}") from exc
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
            for ds_idx, ts in enumerate(self.time_series_data):
                ts_copy = ts.data.copy()
                ts_copy['dataset_id'] = ds_idx  # stable, deterministic identifier
                all_data.append(ts_copy)

            combined = pd.concat(all_data, ignore_index=True)

            # Group by time and compute means. Union the phenotype columns
            # across datasets: a dataset lacking a column contributes NaN,
            # which the mean skips.
            time_col = self.time_series_data[0].time_column
            phenotype_cols = []
            for ts in self.time_series_data:
                for col in ts.phenotype_columns:
                    if col not in phenotype_cols:
                        phenotype_cols.append(col)

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
        """Filter all datasets by phenotype value range.

        Raises:
            ValueError: If phenotype_column is not a phenotype column of
                every dataset (an unknown name is otherwise indistinguishable
                from 'no rows matched').
        """
        missing_datasets = [
            i for i, ts in enumerate(self.time_series_data)
            if phenotype_column not in ts.phenotype_columns
        ]
        if missing_datasets:
            raise ValueError(
                f"Phenotype column '{phenotype_column}' not found in datasets "
                f"{missing_datasets}"
            )
        for ts in self.time_series_data:
            mask = (ts.data[phenotype_column] >= min_val) & (ts.data[phenotype_column] <= max_val)
            ts.data = ts.data[mask].copy()

        logger.info(f"Filtered data by {phenotype_column} range [{min_val}, {max_val}]")

