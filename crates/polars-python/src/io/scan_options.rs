use std::sync::Arc;

use polars::prelude::default_values::DefaultFieldValues;
use polars::prelude::deletion::DeletionFilesList;
use polars::prelude::{
    CastColumnsPolicy, CloudScheme, ColumnMapping, ExtraColumnsPolicy, MissingColumnsPolicy,
    PlSmallStr, Schema, TableStatistics, UnifiedScanArgs,
};
use polars_io::{HiveOptions, RowIndex};
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use crate::PyDataFrame;
use crate::io::cloud_options::OptPyCloudOptions;
use crate::prelude::Wrap;

/// Interface to `class ScanOptions` on the Python side
pub struct PyScanOptions<'py>(Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for PyScanOptions<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.to_owned()))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Wrap<TableStatistics> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let attr = ob.getattr(intern!(py, "_df"))?;
        Ok(Wrap(TableStatistics(Arc::new(
            PyDataFrame::extract(attr.as_borrowed())?.df.into_inner(),
        ))))
    }
}

impl PyScanOptions<'_> {
    pub fn extract_unified_scan_args(
        &self,
        cloud_scheme: Option<CloudScheme>,
    ) -> PyResult<UnifiedScanArgs> {
        #[derive(FromPyObject)]
        struct Extract<'a> {
            row_index: Option<(Wrap<PlSmallStr>, IdxSize)>,
            pre_slice: Option<(i64, usize)>,
            cast_options: Wrap<CastColumnsPolicy>,
            extra_columns: Wrap<ExtraColumnsPolicy>,
            missing_columns: Wrap<MissingColumnsPolicy>,
            include_file_paths: Option<Wrap<PlSmallStr>>,
            glob: bool,
            hidden_file_prefix: Option<Vec<PyBackedStr>>,
            column_mapping: Option<Wrap<ColumnMapping>>,
            default_values: Option<Wrap<DefaultFieldValues>>,
            hive_partitioning: Option<bool>,
            hive_schema: Option<Wrap<Schema>>,
            try_parse_hive_dates: bool,
            rechunk: bool,
            cache: bool,
            storage_options: OptPyCloudOptions<'a>,
            credential_provider: Option<Py<PyAny>>,
            deletion_files: Option<Wrap<DeletionFilesList>>,
            table_statistics: Option<Wrap<TableStatistics>>,
            row_count: Option<(u64, u64)>,
        }

        let Extract {
            row_index,
            pre_slice,
            cast_options,
            extra_columns,
            missing_columns,
            include_file_paths,
            column_mapping,
            default_values,
            glob,
            hidden_file_prefix,
            hive_partitioning,
            hive_schema,
            try_parse_hive_dates,
            rechunk,
            cache,
            storage_options,
            credential_provider,
            deletion_files,
            table_statistics,
            row_count,
        } = self.0.extract()?;

        let cloud_options =
            storage_options.extract_opt_cloud_options(cloud_scheme, credential_provider)?;

        let hive_schema = hive_schema.map(|s| Arc::new(s.0));

        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.0,
            offset,
        });

        let hive_options = HiveOptions {
            enabled: hive_partitioning,
            hive_start_idx: 0,
            schema: hive_schema,
            try_parse_dates: try_parse_hive_dates,
        };

        let unified_scan_args = UnifiedScanArgs {
            // Schema is currently still stored inside the options per scan type, but we do eventually
            // want to put it here instead.
            schema: None,
            cloud_options,
            hive_options,
            rechunk,
            cache,
            glob,
            hidden_file_prefix: hidden_file_prefix
                .map(|x| x.into_iter().map(|x| (*x).into()).collect()),
            projection: None,
            column_mapping: column_mapping.map(|x| x.0),
            default_values: default_values
                .map(|x| x.0)
                .filter(|DefaultFieldValues::Iceberg(v)| !v.is_empty()),
            row_index,
            pre_slice: pre_slice.map(Slice::from),
            cast_columns_policy: cast_options.0,
            missing_columns_policy: missing_columns.0,
            extra_columns_policy: extra_columns.0,
            include_file_paths: include_file_paths.map(|x| x.0),
            deletion_files: DeletionFilesList::filter_empty(deletion_files.map(|x| x.0)),
            table_statistics: table_statistics.map(|x| x.0),
            row_count,
        };

        Ok(unified_scan_args)
    }
}
