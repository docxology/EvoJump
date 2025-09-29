"""
Test suite for DataCore module.

This module tests the data ingestion, validation, and preprocessing functionality
of the DataCore module using real data and methods.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from evojump import datacore


class TestTimeSeriesData:
    """Test TimeSeriesData class."""

    def test_time_series_data_initialization(self):
        """Test TimeSeriesData initialization with valid data."""
        # Create test data
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18],
            'phenotype2': [20, 22, 24, 26, 28]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        assert ts_data.time_column == 'time'
        assert ts_data.phenotype_columns == ['phenotype1', 'phenotype2']
        assert ts_data.n_timepoints == 5
        assert ts_data.n_phenotypes == 2

    def test_time_series_data_invalid_time_column(self):
        """Test TimeSeriesData with invalid time column."""
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        with pytest.raises(ValueError, match="Time column 'invalid_time' not found"):
            datacore.TimeSeriesData(
                data=data,
                time_column='invalid_time',
                phenotype_columns=['phenotype1']
            )

    def test_time_series_data_invalid_phenotype_columns(self):
        """Test TimeSeriesData with invalid phenotype columns."""
        data = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        with pytest.raises(ValueError, match="Phenotype columns not found"):
            datacore.TimeSeriesData(
                data=data,
                time_column='time',
                phenotype_columns=['phenotype1', 'invalid_phenotype']
            )

    def test_time_points_property(self):
        """Test time_points property."""
        data = pd.DataFrame({
            'time': [3, 1, 4, 1, 5, 9, 2],
            'phenotype1': [10, 12, 14, 16, 18, 20, 22]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        expected_times = np.array([1, 2, 3, 4, 5, 9])
        np.testing.assert_array_equal(ts_data.time_points, expected_times)

    def test_get_phenotype_at_time(self):
        """Test get_phenotype_at_time method."""
        data = pd.DataFrame({
            'time': [1, 1, 2, 2, 3, 3],
            'phenotype1': [10, 11, 12, 13, 14, 15]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        result = ts_data.get_phenotype_at_time(2)
        expected_data = {'phenotype1': [12, 13]}

        assert result is not None
        assert len(result) == 2
        assert result['phenotype1'].tolist() == [12, 13]

    def test_interpolate_missing_data(self):
        """Test interpolate_missing_data method."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, np.nan, 14, np.nan, 18]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        ts_data.interpolate_missing_data(method='linear')

        # Check that missing values were interpolated
        assert not ts_data.data['phenotype1'].isna().any()
        assert ts_data.data['phenotype1'].iloc[1] == 12  # Interpolated value
        assert ts_data.data['phenotype1'].iloc[3] == 16  # Interpolated value


class TestMetadataManager:
    """Test MetadataManager class."""

    def test_metadata_manager_initialization(self):
        """Test MetadataManager initialization."""
        metadata_mgr = datacore.MetadataManager()

        assert 'created_at' in metadata_mgr.metadata
        assert 'version' in metadata_mgr.metadata
        assert len(metadata_mgr.metadata['processing_history']) == 0

    def test_add_processing_step(self):
        """Test adding processing step."""
        metadata_mgr = datacore.MetadataManager()

        metadata_mgr.add_processing_step('test_step', {'param1': 'value1'})

        assert len(metadata_mgr.metadata['processing_history']) == 1
        assert metadata_mgr.metadata['processing_history'][0]['step'] == 'test_step'
        assert metadata_mgr.metadata['processing_history'][0]['parameters'] == {'param1': 'value1'}

    def test_save_and_load_metadata_yaml(self):
        """Test saving and loading metadata in YAML format."""
        metadata_mgr = datacore.MetadataManager()
        metadata_mgr.add_processing_step('test_step', {'param': 'value'})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = Path(f.name)

        try:
            metadata_mgr.save_metadata(temp_file)

            # Create new metadata manager and load
            metadata_mgr2 = datacore.MetadataManager(temp_file)

            assert metadata_mgr2.metadata['processing_history'][0]['step'] == 'test_step'
            assert metadata_mgr2.metadata['processing_history'][0]['parameters'] == {'param': 'value'}

        finally:
            temp_file.unlink()

    def test_save_and_load_metadata_json(self):
        """Test saving and loading metadata in JSON format."""
        metadata_mgr = datacore.MetadataManager()
        metadata_mgr.add_processing_step('test_step', {'param': 'value'})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = Path(f.name)

        try:
            metadata_mgr.save_metadata(temp_file)

            # Create new metadata manager and load
            metadata_mgr2 = datacore.MetadataManager(temp_file)

            assert metadata_mgr2.metadata['processing_history'][0]['step'] == 'test_step'
            assert metadata_mgr2.metadata['processing_history'][0]['parameters'] == {'param': 'value'}

        finally:
            temp_file.unlink()


class TestDataCore:
    """Test DataCore class."""

    def create_test_data(self):
        """Create test data for DataCore tests."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18, 11, 13, 15, 17, 19],
            'phenotype2': [20, 22, 24, 26, 28, 21, 23, 25, 27, 29]
        })
        return data

    def test_data_core_initialization(self):
        """Test DataCore initialization."""
        data = self.create_test_data()

        ts_data1 = datacore.TimeSeriesData(
            data=data.iloc[:5],
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        ts_data2 = datacore.TimeSeriesData(
            data=data.iloc[5:],
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        data_core = datacore.DataCore([ts_data1, ts_data2])

        assert len(data_core.time_series_data) == 2
        assert data_core.metadata_manager is not None

    def test_load_from_csv(self):
        """Test loading data from CSV file."""
        data = self.create_test_data()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = Path(f.name)

        try:
            data_core = datacore.DataCore.load_from_csv(
                file_path=temp_file,
                time_column='time',
                phenotype_columns=['phenotype1', 'phenotype2']
            )

            assert len(data_core.time_series_data) == 1
            assert len(data_core.time_series_data[0].data) == 10

        finally:
            temp_file.unlink()

    def test_preprocess_data(self):
        """Test data preprocessing."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, np.nan, 14, 1000, 18],  # 1000 is outlier
            'phenotype2': [20, 22, np.nan, 26, 28]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        data_core = datacore.DataCore([ts_data])

        # Preprocess with outlier removal and interpolation
        data_core.preprocess_data(
            normalize=True,
            remove_outliers=True,
            interpolate_missing=True,
            interpolation_method='linear'
        )

        # Check that missing values were interpolated
        assert not data_core.time_series_data[0].data['phenotype2'].isna().any()

        # Check that outlier was handled
        assert data_core.time_series_data[0].data['phenotype1'].iloc[3] != 1000

    def test_validate_data_quality(self):
        """Test data quality validation."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, np.nan, 14, 16, 18],
            'phenotype2': [20, 22, 24, 26, 28]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        data_core = datacore.DataCore([ts_data])

        quality_metrics = data_core.validate_data_quality()

        assert 'missing_data_percentage' in quality_metrics
        assert 'outlier_percentage' in quality_metrics
        assert 'temporal_consistency' in quality_metrics

        # Check missing data percentage
        assert quality_metrics['missing_data_percentage']['dataset_0'] > 0

    def test_save_processed_data(self):
        """Test saving processed data."""
        data = self.create_test_data()

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        data_core = datacore.DataCore([ts_data])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = Path(f.name)

        try:
            data_core.save_processed_data(temp_file, format='csv')

            # Verify file was created and has correct content
            saved_data = pd.read_csv(temp_file)
            assert len(saved_data) == 10
            assert 'phenotype1' in saved_data.columns
            assert 'phenotype2' in saved_data.columns

        finally:
            temp_file.unlink()

    def test_get_aggregated_data(self):
        """Test data aggregation."""
        data1 = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        data2 = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [11, 13, 15]
        })

        ts_data1 = datacore.TimeSeriesData(
            data=data1,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        ts_data2 = datacore.TimeSeriesData(
            data=data2,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data1, ts_data2])

        aggregated = data_core.get_aggregated_data(aggregation_method='mean')

        assert len(aggregated) == 3
        assert aggregated.loc[aggregated['time'] == 1, 'phenotype1'].iloc[0] == 10.5  # Mean of 10 and 11
        assert aggregated.loc[aggregated['time'] == 2, 'phenotype1'].iloc[0] == 12.5  # Mean of 12 and 13

    def test_filter_by_time_range(self):
        """Test filtering by time range."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        data_core.filter_by_time_range(2, 4)

        assert len(data_core.time_series_data[0].data) == 3
        assert all(data_core.time_series_data[0].data['time'].isin([2, 3, 4]))

    def test_filter_by_phenotype_range(self):
        """Test filtering by phenotype range."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        data_core.filter_by_phenotype_range('phenotype1', 12, 16)

        assert len(data_core.time_series_data[0].data) == 3
        assert all(data_core.time_series_data[0].data['phenotype1'].isin([12, 14, 16]))

    def test_validate_data_consistency_multiple_datasets(self):
        """Test validation of data consistency across multiple datasets."""
        # Create datasets with different time columns
        data1 = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })

        data2 = pd.DataFrame({
            'timepoint': [1, 2, 3],  # Different time column name
            'phenotype1': [11, 13, 15]
        })

        ts_data1 = datacore.TimeSeriesData(
            data=data1,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        ts_data2 = datacore.TimeSeriesData(
            data=data2,
            time_column='timepoint',
            phenotype_columns=['phenotype1']
        )

        with pytest.raises(ValueError, match="Inconsistent time column names"):
            datacore.DataCore([ts_data1, ts_data2])

    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 1000, 16, 18]  # 1000 is outlier
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        data_core._remove_outliers(ts_data, method='iqr', threshold=1.5)

        # Check that outlier was removed
        assert 1000 not in data_core.time_series_data[0].data['phenotype1'].values

    def test_remove_outliers_zscore(self):
        """Test outlier removal using z-score method."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 1000, 16, 18]  # 1000 is extreme outlier
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        # Use a very strict threshold to ensure outlier removal
        data_core._remove_outliers(ts_data, method='zscore', threshold=1.0)

        # Check that outlier was removed
        assert 1000 not in data_core.time_series_data[0].data['phenotype1'].values

    def test_normalize_data_zscore(self):
        """Test data normalization using z-score."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        data_core._normalize_data(ts_data, method='zscore')

        # Check that data is standardized
        mean_val = data_core.time_series_data[0].data['phenotype1'].mean()
        std_val = data_core.time_series_data[0].data['phenotype1'].std()

        assert abs(mean_val) < 1e-10  # Mean should be approximately 0
        assert abs(std_val - 1.0) < 1e-10  # Std should be approximately 1

    def test_normalize_data_minmax(self):
        """Test data normalization using min-max scaling."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        data_core._normalize_data(ts_data, method='minmax')

        # Check that data is in [0, 1] range
        min_val = data_core.time_series_data[0].data['phenotype1'].min()
        max_val = data_core.time_series_data[0].data['phenotype1'].max()

        assert abs(min_val - 0.0) < 1e-10
        assert abs(max_val - 1.0) < 1e-10

    def test_normalize_data_robust(self):
        """Test data normalization using robust scaling."""
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'phenotype1': [10, 12, 14, 16, 18]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        data_core = datacore.DataCore([ts_data])

        data_core._normalize_data(ts_data, method='robust')

        # Check that median is approximately 0 and MAD is approximately 1
        median_val = data_core.time_series_data[0].data['phenotype1'].median()

        assert abs(median_val) < 1e-10  # Median should be approximately 0
