use std::sync::Arc;

use polars::prelude::default_values::DefaultFieldValues;
use polars::prelude::deletion::DeletionFilesList;
use polars::prelude::{
    CastColumnsPolicy, ColumnMapping, ExtraColumnsPolicy, MissingColumnsPolicy, PlSmallStr, Schema,
    UnifiedScanArgs,
};
use polars_io::{HiveOptions, RowIndex};
use polars_utils::IdxSize;
use polars_utils::plpath::PlPathRef;
use polars_utils::slice_enum::Slice;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, PyObject, PyResult};

use crate::prelude::Wrap;

/// Interface to `class ScanOptions` on the Python side
pub struct PyScanOptions<'py>(Bound<'py, pyo3::PyAny>);

impl<'py> FromPyObject<'py> for PyScanOptions<'py> {
    fn extract_bound(ob: &Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(Self(ob.clone()))
    }
}

impl PyScanOptions<'_> {
    pub fn extract_unified_scan_args(
        &self,
        // For cloud_options init
        first_path: Option<PlPathRef>,
    ) -> PyResult<UnifiedScanArgs> {
        #[derive(FromPyObject)]
        struct Extract {
            row_index: Option<(Wrap<PlSmallStr>, IdxSize)>,
            pre_slice: Option<(i64, usize)>,
            cast_options: Wrap<CastColumnsPolicy>,
            extra_columns: Wrap<ExtraColumnsPolicy>,
            missing_columns: Wrap<MissingColumnsPolicy>,
            include_file_paths: Option<Wrap<PlSmallStr>>,
            glob: bool,
            column_mapping: Option<Wrap<ColumnMapping>>,
            default_values: Option<Wrap<DefaultFieldValues>>,
            hive_partitioning: Option<bool>,
            hive_schema: Option<Wrap<Schema>>,
            try_parse_hive_dates: bool,
            rechunk: bool,
            cache: bool,
            storage_options: Option<Vec<(String, String)>>,
            credential_provider: Option<PyObject>,
            retries: usize,
            deletion_files: Option<Wrap<DeletionFilesList>>,
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
            hive_partitioning,
            hive_schema,
            try_parse_hive_dates,
            rechunk,
            cache,
            storage_options,
            credential_provider,
            retries,
            deletion_files,
        } = self.0.extract()?;

        let cloud_options = storage_options;

        let cloud_options = if let Some(first_path) = first_path {
            #[cfg(feature = "cloud")]
            {
                use polars_io::cloud::credential_provider::PlCredentialProvider;

                use crate::prelude::parse_cloud_options;

                let first_path_url = first_path.to_str();
                let cloud_options =
                    parse_cloud_options(first_path_url, cloud_options.unwrap_or_default())?;

                Some(
                    cloud_options
                        .with_max_retries(retries)
                        .with_credential_provider(
                            credential_provider.map(PlCredentialProvider::from_python_builder),
                        ),
                )
            }

            #[cfg(not(feature = "cloud"))]
            {
                None
            }
        } else {
            None
        };

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
        };

        Ok(unified_scan_args)
    }
}
