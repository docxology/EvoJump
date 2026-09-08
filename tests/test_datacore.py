"""
Test suite for DataCore module.

This module tests the data ingestion, validation, and preprocessing functionality
of the DataCore module using real data and methods.
"""

import pytest
import pandas as pd
import h5py
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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


    def test_interpolate_missing_data_duplicate_index(self):
        """Test that duplicate index labels do not expand the DataFrame."""
        data = pd.DataFrame(
            {'time': [1.0, 1.0, 2.0], 'phenotype1': [10.0, np.nan, 20.0]},
            index=[0, 0, 1]  # duplicate label 0, as produced by pd.concat
        )

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        ts_data.interpolate_missing_data(method='linear')

        # Row count must be preserved (label-based restore would cross-join)
        assert len(ts_data.data) == 3
        assert ts_data.data['phenotype1'].tolist() == [10.0, 15.0, 20.0]

    def test_interpolate_missing_data_restores_row_order(self):
        """Test that interpolation over time-sorted rows restores original order."""
        data = pd.DataFrame({
            'time': [3.0, 1.0, 2.0, 5.0, 4.0],
            'phenotype1': [np.nan, 10.0, np.nan, 30.0, np.nan]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        ts_data.interpolate_missing_data(method='linear')

        # Original row order restored
        assert ts_data.data['time'].tolist() == [3.0, 1.0, 2.0, 5.0, 4.0]
        # Values interpolated temporally: t=2,3,4 fall between 10 (t=1) and 30 (t=5)
        assert ts_data.data['phenotype1'].tolist() == [20.0, 10.0, 15.0, 30.0, 25.0]

    def test_interpolate_missing_data_boundary_fill(self):
        """Test leading/trailing NaN handling (ffill/bfill boundary path)."""
        data = pd.DataFrame({
            'time': [1.0, 2.0, 3.0, 4.0],
            'lead': [np.nan, 12.0, 14.0, 16.0],
            'trail': [10.0, 12.0, 14.0, np.nan]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['lead', 'trail']
        )

        ts_data.interpolate_missing_data(method='linear')

        assert ts_data.data['lead'].iloc[0] == 12.0   # backfilled at leading edge
        assert ts_data.data['trail'].iloc[3] == 14.0  # forward-filled at trailing edge
        assert not ts_data.data[['lead', 'trail']].isna().any().any()

    def test_interpolate_missing_data_nan_time_raises(self):
        """Test that a missing time value raises instead of being backfilled."""
        data = pd.DataFrame({
            'time': [1.0, np.nan, 3.0],
            'phenotype1': [10.0, 20.0, 30.0]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        with pytest.raises(ValueError, match="contains missing values"):
            ts_data.interpolate_missing_data(method='linear')

    def test_interpolate_missing_data_method_propagation(self):
        """Test that a non-default method argument is propagated to pandas."""
        data = pd.DataFrame({
            'time': [1.0, 2.0, 3.0, 4.0],
            'phenotype1': [10.0, np.nan, np.nan, 31.0]
        })

        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1']
        )

        ts_data.interpolate_missing_data(method='nearest')

        # 'nearest' snaps to the closest observation instead of interpolating
        # (linear would give 17.0 and 24.0)
        assert ts_data.data['phenotype1'].tolist() == [10.0, 10.0, 31.0, 31.0]
    def test_interpolate_missing_data_no_numeric_phenotypes_is_noop(self):
        """A frame whose only numeric column is time has nothing to fill."""
        data = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'strain': ['WT', 'MUT', 'WT'],
        })
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['strain'])

        ts_data.interpolate_missing_data(method='linear')

        # No-op path: the non-numeric phenotype column is left untouched
        assert ts_data.data['strain'].tolist() == ['WT', 'MUT', 'WT']
        assert ts_data.data['time'].tolist() == [1.0, 2.0, 3.0]
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

    @pytest.mark.parametrize("suffix", ['.yaml', '.json'])
    def test_save_and_load_metadata_roundtrip(self, suffix):
        """Saved metadata reloads through a new MetadataManager in every format."""
        metadata_mgr = datacore.MetadataManager()
        metadata_mgr.add_processing_step('test_step', {'param': 'value'})

        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            temp_file = Path(f.name)

        try:
            metadata_mgr.save_metadata(temp_file)

            # Create new metadata manager and load
            metadata_mgr2 = datacore.MetadataManager(temp_file)

            assert metadata_mgr2.metadata['processing_history'][0]['step'] == 'test_step'
            assert metadata_mgr2.metadata['processing_history'][0]['parameters'] == {'param': 'value'}

        finally:
            temp_file.unlink()

    def test_save_metadata_unsupported_suffix_raises(self):
        """Test that saving to an unsupported metadata format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = Path(f.name)

        try:
            metadata_mgr = datacore.MetadataManager()
            with pytest.raises(ValueError, match="Unsupported metadata format"):
                metadata_mgr.save_metadata(temp_file)
        finally:
            temp_file.unlink()



    def test_load_metadata_empty_file_raises(self):
        """Test that an empty metadata file raises a clear error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = Path(f.name)

        try:
            metadata_mgr = datacore.MetadataManager()
            with pytest.raises(ValueError, match="[Ee]mpty"):
                metadata_mgr.load_metadata(temp_file)
        finally:
            temp_file.unlink()

    def test_load_metadata_unsupported_suffix_raises(self):
        """Test that an unsupported metadata format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('not metadata')
            temp_file = Path(f.name)

        try:
            metadata_mgr = datacore.MetadataManager()
            with pytest.raises(ValueError, match="Unsupported metadata format"):
                metadata_mgr.load_metadata(temp_file)
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

    @pytest.mark.parametrize("method,check", [
        # z-score: mean 0, std 1; min-max: [0, 1]; robust: median 0
        ("zscore", lambda col: abs(col.mean()) < 1e-10 and abs(col.std() - 1.0) < 1e-10),
        ("minmax", lambda col: abs(col.min()) < 1e-10 and abs(col.max() - 1.0) < 1e-10),
        ("robust", lambda col: abs(col.median()) < 1e-10),
    ])
    def test_normalize_data(self, method, check):
        """Normalization satisfies the per-method contract and preserves order."""
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
        before = ts_data.data['phenotype1'].to_numpy()

        data_core._normalize_data(ts_data, method=method)

        normalized = ts_data.data['phenotype1']
        assert check(normalized)
        # All three transforms are monotone: the value ranking must survive
        np.testing.assert_array_equal(
            np.argsort(before), np.argsort(normalized.to_numpy()))


    def test_data_core_accepts_single_time_series(self):
        """Test that DataCore accepts a single TimeSeriesData object."""
        data = self.create_test_data()
        ts_data = datacore.TimeSeriesData(
            data=data,
            time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2']
        )

        data_core = datacore.DataCore(ts_data)

        assert len(data_core.time_series_data) == 1

    def test_append_time_series(self):
        """Test appending a dataset to an existing DataCore."""
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

        data_core = datacore.DataCore(ts_data1)
        data_core.append(ts_data2)

        assert len(data_core.time_series_data) == 2

    def test_load_from_hdf5_flat_layout(self):
        """Test loading a flat HDF5 file written directly with h5py."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w') as f:
                f.create_dataset('time', data=np.array([1.0, 2.0, 3.0]))
                f.create_dataset('phenotype1', data=np.array([10.0, 20.0, 30.0]))

            data_core = datacore.DataCore.load_from_hdf5(temp_file, time_column='time')

            ts = data_core.time_series_data[0]
            assert len(ts.data) == 3
            np.testing.assert_array_equal(ts.data['time'].to_numpy(), [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(ts.data['phenotype1'].to_numpy(), [10.0, 20.0, 30.0])
        finally:
            temp_file.unlink()

    def test_load_from_hdf5_flattens_groups(self):
        """Test that group members are flattened into sanitized column names."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w') as f:
                f.create_dataset('time', data=np.array([1.0, 2.0]))
                group = f.create_group('conditions')
                group.create_dataset('dose', data=np.array([5.0, 10.0]))

            data_core = datacore.DataCore.load_from_hdf5(temp_file, time_column='time')

            ts = data_core.time_series_data[0]
            # '/' is not usable as a positional column name; it is replaced
            assert 'conditions_dose' in ts.data.columns
            np.testing.assert_array_equal(ts.data['conditions_dose'].to_numpy(), [5.0, 10.0])
        finally:
            temp_file.unlink()

    def test_load_from_hdf5_unequal_lengths_raises(self):
        """Test that unequal-length datasets produce a descriptive error."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w') as f:
                f.create_dataset('time', data=np.array([1.0, 2.0, 3.0]))
                f.create_dataset('phenotype1', data=np.array([10.0, 20.0]))

            with pytest.raises(ValueError, match="unequal lengths"):
                datacore.DataCore.load_from_hdf5(temp_file, time_column='time')
        finally:
            temp_file.unlink()

    def test_save_and_load_hdf5_roundtrip(self):
        """Test the save_processed_data/load_from_hdf5 round-trip, including strings."""
        data1 = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'phenotype1': [10.0, 12.0, 14.0],
            'strain': ['WT', 'WT', 'WT']
        })
        data2 = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'phenotype1': [11.0, 13.0, 15.0],
            'strain': ['MUT', 'MUT', 'MUT']
        })
        ts_data1 = datacore.TimeSeriesData(
            data=data1, time_column='time', phenotype_columns=['phenotype1'])
        ts_data2 = datacore.TimeSeriesData(
            data=data2, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore([ts_data1, ts_data2])

        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            data_core.save_processed_data(temp_file, format='hdf5')
            loaded = datacore.DataCore.load_from_hdf5(temp_file, time_column='time')

            assert len(loaded.time_series_data) == 2
            np.testing.assert_array_equal(
                loaded.time_series_data[0].data['time'].to_numpy(), [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(
                loaded.time_series_data[0].data['phenotype1'].to_numpy(), [10.0, 12.0, 14.0])
            np.testing.assert_array_equal(
                loaded.time_series_data[1].data['phenotype1'].to_numpy(), [11.0, 13.0, 15.0])
            # Non-numeric column encoded as strings survives the round-trip
            assert loaded.time_series_data[0].data['strain'].tolist() == ['WT', 'WT', 'WT']
            assert loaded.time_series_data[1].data['strain'].tolist() == ['MUT', 'MUT', 'MUT']
        finally:
            temp_file.unlink()

    def test_save_processed_data_unsupported_format_raises(self):
        """Test that an unsupported save format raises ValueError."""
        data = self.create_test_data()
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2'])
        data_core = datacore.DataCore(ts_data)

        temp_file = Path(tempfile.mktemp(suffix='.xml'))
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                data_core.save_processed_data(temp_file, format='xml')
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_get_aggregated_data_unsupported_method_raises(self):
        """Test that an unsupported aggregation method raises ValueError."""
        data = self.create_test_data()
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore(ts_data)

        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            data_core.get_aggregated_data(aggregation_method='median')

    def test_get_aggregated_data_union_of_phenotype_columns(self):
        """Test aggregation with heterogeneous phenotype column sets."""
        data1 = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [10, 12, 14]
        })
        data2 = pd.DataFrame({
            'time': [1, 2, 3],
            'phenotype1': [11, 13, 15],
            'phenotype2': [100, 200, 300]
        })
        ts_data1 = datacore.TimeSeriesData(
            data=data1, time_column='time', phenotype_columns=['phenotype1'])
        ts_data2 = datacore.TimeSeriesData(
            data=data2, time_column='time', phenotype_columns=['phenotype1', 'phenotype2'])
        data_core = datacore.DataCore([ts_data1, ts_data2])

        aggregated = data_core.get_aggregated_data(aggregation_method='mean')

        # phenotype2 exists only in dataset 2 but must not be silently dropped
        assert 'phenotype2' in aggregated.columns
        assert aggregated['phenotype2'].tolist() == [100.0, 200.0, 300.0]
        assert aggregated['phenotype1'].tolist() == [10.5, 12.5, 14.5]

    def test_filter_by_phenotype_range_unknown_column_raises(self):
        """Test that an unknown phenotype column raises instead of filtering nothing."""
        data = self.create_test_data()
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore(ts_data)

        with pytest.raises(ValueError, match="not found"):
            data_core.filter_by_phenotype_range('nonexistent', 0.0, 1.0)

    def test_remove_outliers_combined_across_columns(self):
        """Test that the outlier mask is combined across columns and applied once."""
        data = pd.DataFrame({
            'time': list(range(1, 10)),
            'a': [1000.0, np.nan, 10.0, 10.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            'b': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 5000.0]
        })
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['a', 'b'])
        data_core = datacore.DataCore(ts_data)

        data_core._remove_outliers(ts_data, method='iqr', threshold=1.5)

        kept = ts_data.data
        # Exactly the two outlier rows removed, regardless of column order
        assert kept['time'].tolist() == [2, 3, 4, 5, 6, 7, 8]
        assert 1000.0 not in kept['a'].to_numpy()
        assert 5000.0 not in kept['b'].to_numpy()
        # A NaN value is not an outlier: its row must not be silently dropped
        assert kept['a'].isna().any()

    def test_remove_outliers_would_remove_all_rows_raises(self):
        """Test that outlier removal refuses to empty the dataset."""
        data = pd.DataFrame({
            'time': [1, 2],
            'phenotype1': [0.0, 10.0]  # both points are >0.5 std from the mean
        })
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore(ts_data)

        with pytest.raises(ValueError, match="remove every row"):
            data_core._remove_outliers(ts_data, method='zscore', threshold=0.5)

    def test_validate_data_quality_single_time_point_and_breakdown(self):
        """Test temporal_consistency for a single time point and per-column outliers."""
        data = pd.DataFrame({
            'time': [1.0, 1.0],
            'phenotype1': [10.0, 12.0]
        })
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore(ts_data)

        quality_metrics = data_core.validate_data_quality()

        # Key must always exist per dataset, even without measurable intervals
        assert 'dataset_0' in quality_metrics['temporal_consistency']
        assert quality_metrics['temporal_consistency']['dataset_0']['regularity_score'] is None
        # Per-column outlier breakdown available
        assert 'outliers_by_column' in quality_metrics
        assert 'phenotype1' in quality_metrics['outliers_by_column']['dataset_0']

    def test_data_core_empty_dataset_list_raises(self):
        """Test that constructing DataCore without datasets raises."""
        with pytest.raises(ValueError, match="No time series data provided"):
            datacore.DataCore([])

    def test_load_from_csv_with_metadata_file(self):
        """Test that load_from_csv attaches the metadata manager and records the step."""
        data = self.create_test_data()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_file = Path(f.name)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"source": "unit-test"}')
            meta_file = Path(f.name)

        try:
            data_core = datacore.DataCore.load_from_csv(
                file_path=csv_file,
                time_column='time',
                phenotype_columns=['phenotype1', 'phenotype2'],
                metadata_file=meta_file
            )

            # The provided metadata is adopted and provenance is recorded
            assert data_core.metadata_manager.metadata['source'] == 'unit-test'
            steps = [s['step'] for s in data_core.metadata_manager.metadata['processing_history']]
            assert 'load_from_csv' in steps
        finally:
            csv_file.unlink()
            meta_file.unlink()

    def test_load_from_csv_auto_detects_phenotype_columns(self):
        """Test that phenotype columns are auto-detected when not provided."""
        data = self.create_test_data()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_file = Path(f.name)

        try:
            data_core = datacore.DataCore.load_from_csv(
                file_path=csv_file, time_column='time')

            # Every numeric non-time column becomes a phenotype
            assert data_core.time_series_data[0].phenotype_columns == [
                'phenotype1', 'phenotype2']
        finally:
            csv_file.unlink()

    def test_load_from_hdf5_fixed_length_string_decode(self):
        """Test that fixed-length byte-string datasets decode to unicode."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w') as f:
                f.create_dataset('time', data=np.array([1.0, 2.0]))
                f.create_dataset('strain', data=np.array([b'WT', b'MUT'], dtype='S4'))

            data_core = datacore.DataCore.load_from_hdf5(temp_file, time_column='time')

            ts = data_core.time_series_data[0]
            assert ts.data['strain'].tolist() == ['WT', 'MUT']
        finally:
            temp_file.unlink()

    def test_load_from_hdf5_save_layout_unequal_lengths_raises(self):
        """Test that a save-layout group with unequal-length datasets raises."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w') as f:
                group = f.create_group('dataset_0')
                group.create_dataset('time', data=np.array([1.0, 2.0, 3.0]))
                group.create_dataset('phenotype1', data=np.array([10.0, 20.0]))

            with pytest.raises(ValueError, match="unequal lengths"):
                datacore.DataCore.load_from_hdf5(temp_file, time_column='time')
        finally:
            temp_file.unlink()

    def test_load_from_hdf5_no_data_columns_raises(self):
        """Test that an HDF5 file without datasets raises a clear error."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w'):
                pass

            with pytest.raises(ValueError, match="No data columns found"):
                datacore.DataCore.load_from_hdf5(temp_file, time_column='time')
        finally:
            temp_file.unlink()

    def test_load_from_hdf5_explicit_phenotype_columns(self):
        """Test that explicit phenotype_columns override auto-detection."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        try:
            with h5py.File(temp_file, 'w') as f:
                f.create_dataset('time', data=np.array([1.0, 2.0, 3.0]))
                f.create_dataset('phenotype1', data=np.array([10.0, 20.0, 30.0]))
                f.create_dataset('strain', data=np.array([b'WT', b'MUT', b'WT'], dtype='S4'))

            data_core = datacore.DataCore.load_from_hdf5(
                temp_file, time_column='time', phenotype_columns=['phenotype1'])

            # Only the explicitly listed phenotype is registered, not the
            # string column or the time column
            assert data_core.time_series_data[0].phenotype_columns == ['phenotype1']
        finally:
            temp_file.unlink()

    def test_load_from_hdf5_with_metadata_file(self):
        """Test that load_from_hdf5 attaches the metadata manager and records the step."""
        temp_file = Path(tempfile.mktemp(suffix='.h5'))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"source": "hdf5-meta"}')
            meta_file = Path(f.name)
        try:
            with h5py.File(temp_file, 'w') as f:
                f.create_dataset('time', data=np.array([1.0, 2.0]))
                f.create_dataset('phenotype1', data=np.array([10.0, 20.0]))

            data_core = datacore.DataCore.load_from_hdf5(
                temp_file, time_column='time', metadata_file=meta_file)

            assert data_core.metadata_manager.metadata['source'] == 'hdf5-meta'
            steps = [s['step'] for s in data_core.metadata_manager.metadata['processing_history']]
            assert 'load_from_hdf5' in steps
        finally:
            temp_file.unlink()
            meta_file.unlink()

    def test_remove_outliers_unsupported_method_raises(self):
        """Test that an unsupported outlier method raises ValueError."""
        data = self.create_test_data()
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['phenotype1'])
        data_core = datacore.DataCore(ts_data)

        with pytest.raises(ValueError, match="Unsupported outlier method"):
            data_core._remove_outliers(ts_data, method='mad')

    def test_remove_outliers_zscore_constant_column_skipped(self):
        """Test that a zero-variance phenotype column is skipped, not all-outlier."""
        data = pd.DataFrame({
            'time': list(range(1, 10)),
            'constant': [5.0] * 9,
            'varying': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 5000.0],
        })
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time', phenotype_columns=['constant', 'varying'])
        data_core = datacore.DataCore(ts_data)

        data_core._remove_outliers(ts_data, method='zscore', threshold=1.5)

        kept = ts_data.data
        # The constant column contributes no mask: only the varying outlier
        # row is dropped
        assert len(kept) == 8
        assert kept['constant'].tolist() == [5.0] * 8
        assert 5000.0 not in kept['varying'].to_numpy()

    def test_save_processed_data_parquet(self):
        """Test parquet output round-trips, or names the missing engine."""
        data = self.create_test_data()
        ts_data = datacore.TimeSeriesData(
            data=data, time_column='time',
            phenotype_columns=['phenotype1', 'phenotype2'])
        data_core = datacore.DataCore(ts_data)

        temp_file = Path(tempfile.mktemp(suffix='.parquet'))
        try:
            try:
                data_core.save_processed_data(temp_file, format='parquet')
            except ValueError as exc:
                # No parquet engine installed: the error must name the
                # missing dependency instead of leaking a raw ImportError
                assert 'pyarrow' in str(exc) or 'fastparquet' in str(exc)
            else:
                saved = pd.read_parquet(temp_file)
                assert len(saved) == 10
                assert list(saved.columns) == ['time', 'phenotype1', 'phenotype2']
        finally:
            if temp_file.exists():
                temp_file.unlink()
