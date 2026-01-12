use polars::prelude::sink::{PartitionTargetCallback, SinkFinishCallback};
use polars::prelude::sink2::{FileProviderFunction, FileProviderType};
use polars::prelude::{PartitionStrategy, PlPath, SinkDestination, SortColumn, SpecialEq};
use polars_utils::IdxSize;
use polars_utils::python_function::PythonObject;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

use crate::PyExpr;
use crate::prelude::Wrap;

pub struct PyFileSinkDestination<'py>(Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for PyFileSinkDestination<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.to_owned()))
    }
}

impl PyFileSinkDestination<'_> {
    pub fn extract_file_sink_destination(&self) -> PyResult<SinkDestination> {
        let py = self.0.py();

        if let Ok(sink_output_dataclass) = self.0.getattr(intern!(py, "_pl_sink_directory")) {
            return self.extract_from_py_sink_directory(sink_output_dataclass);
        };

        if let Ok(partition_by_dataclass) = self.0.getattr(intern!(py, "_pl_partition_by")) {
            return self.extract_from_py_partition_by(partition_by_dataclass);
        };

        let v: Wrap<polars_plan::dsl::SinkTarget> = self.0.extract()?;

        Ok(SinkDestination::File { target: v.0 })
    }

    fn extract_from_py_sink_directory(
        &self,
        sink_output_dataclass: Bound<'_, PyAny>,
    ) -> PyResult<SinkDestination> {
        /// Extract from `SinkDirectoryInner` dataclass.
        #[derive(FromPyObject)]
        struct Extract {
            base_path: Wrap<PlPath>,
            file_path_provider: Option<Py<PyAny>>,
            partition_by: Option<Vec<PyExpr>>,
            partition_keys_sorted: Option<bool>,
            include_keys: Option<bool>,
            per_partition_sort_by: Option<Vec<PyExpr>>,
            per_file_sort_by: Option<Vec<PyExpr>>,
            max_rows_per_file: Option<IdxSize>,
            finish_callback: Option<Py<PyAny>>,
        }

        let Extract {
            base_path,
            file_path_provider,
            partition_by,
            partition_keys_sorted,
            include_keys,
            per_partition_sort_by,
            per_file_sort_by,
            max_rows_per_file,
            finish_callback,
        } = sink_output_dataclass.extract()?;

        if per_partition_sort_by.is_some() && per_file_sort_by.is_some() {
            return Err(PyValueError::new_err(
                "cannot specify both 'per_partition_sort_by' and 'per_file_sort_by'",
            ));
        }

        let partition_strategy: PartitionStrategy = if let Some(partition_by) = partition_by {
            if max_rows_per_file.is_some() {
                return Err(PyValueError::new_err(
                    "unimplemented: 'max_rows_per_file' with 'partition_by'",
                ));
            }

            if per_file_sort_by.is_some() {
                return Err(PyValueError::new_err(
                    "unimplemented: 'per_file_sort_by' with 'partition_by'",
                ));
            }

            PartitionStrategy::Keyed {
                keys: partition_by.into_iter().map(|x| x.inner).collect(),
                include_keys: include_keys.unwrap_or(true),
                keys_pre_grouped: false,
                per_partition_sort_by: per_partition_sort_by
                    .unwrap_or_default()
                    .into_iter()
                    .map(|x| SortColumn {
                        expr: x.inner,
                        descending: false,
                        nulls_last: false,
                    })
                    .collect(),
            }
        } else if let Some(parameter_name) = partition_keys_sorted
            .as_ref()
            .is_some()
            .then_some("partition_keys_sorted")
            .or(include_keys.is_some().then_some("include_keys"))
            .or(per_partition_sort_by
                .is_some()
                .then_some("per_partition_sort_by"))
        {
            return Err(PyValueError::new_err(format!(
                "cannot use '{parameter_name}' without specifying `partition_by`"
            )));
        } else if max_rows_per_file.is_some() {
            PartitionStrategy::FileSize
        } else {
            return Err(PyValueError::new_err(
                "at least one of ('partition_by', 'max_rows_per_file') \
                must be specified for SinkPartitioned",
            ));
        };

        Ok(SinkDestination::Partitioned {
            base_path: base_path.0,
            file_path_provider: file_path_provider.map(|x| {
                FileProviderType::Legacy(PartitionTargetCallback::Python(PythonObject(x)))
            }),
            partition_strategy,
            finish_callback: finish_callback.map(|x| SinkFinishCallback::Python(PythonObject(x))),
            max_rows_per_file: max_rows_per_file.unwrap_or(IdxSize::MAX),
            approximate_bytes_per_file: u64::MAX,
        })
    }

    fn extract_from_py_partition_by(
        &self,
        partition_by_dataclass: Bound<'_, PyAny>,
    ) -> PyResult<SinkDestination> {
        /// Extract from `PartitionByInner` dataclass.
        #[derive(FromPyObject)]
        struct Extract {
            base_path: Wrap<PlPath>,
            file_path_provider: Option<Py<PyAny>>,
            key: Option<Vec<PyExpr>>,
            include_key: Option<bool>,
            max_rows_per_file: Option<IdxSize>,
            approximate_bytes_per_file: u64,
        }

        let Extract {
            base_path,
            file_path_provider,
            key,
            include_key,
            max_rows_per_file,
            approximate_bytes_per_file,
        } = partition_by_dataclass.extract()?;

        let partition_strategy: PartitionStrategy = if let Some(partition_by) = key {
            PartitionStrategy::Keyed {
                keys: partition_by.into_iter().map(|x| x.inner).collect(),
                include_keys: include_key.unwrap_or(true),
                keys_pre_grouped: false,
                per_partition_sort_by: vec![],
            }
        } else {
            // Should be validated on Python side
            assert!(include_key.is_none());

            PartitionStrategy::FileSize
        };

        Ok(SinkDestination::Partitioned {
            base_path: base_path.0,
            file_path_provider: file_path_provider.map(|x| {
                FileProviderType::Function(FileProviderFunction::Python(SpecialEq::new(
                    PythonObject(x).into(),
                )))
            }),
            partition_strategy,
            finish_callback: None,
            max_rows_per_file: max_rows_per_file.unwrap_or(IdxSize::MAX),
            approximate_bytes_per_file,
        })
    }
}
