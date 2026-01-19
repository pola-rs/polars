use polars::prelude::file_provider::{FileProviderFunction, FileProviderType};
use polars::prelude::{PartitionStrategy, PlRefPath, SinkDestination, SpecialEq};
use polars_utils::IdxSize;
use polars_utils::python_function::PythonObject;
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

        if let Ok(partition_by_dataclass) = self.0.getattr(intern!(py, "_pl_partition_by")) {
            return self.extract_from_py_partition_by(partition_by_dataclass);
        };

        let v: Wrap<polars_plan::dsl::SinkTarget> = self.0.extract()?;

        Ok(SinkDestination::File { target: v.0 })
    }

    fn extract_from_py_partition_by(
        &self,
        partition_by_dataclass: Bound<'_, PyAny>,
    ) -> PyResult<SinkDestination> {
        /// Extract from `PartitionByInner` dataclass.
        #[derive(FromPyObject)]
        struct Extract {
            base_path: Wrap<PlRefPath>,
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
            max_rows_per_file: max_rows_per_file.unwrap_or(IdxSize::MAX),
            approximate_bytes_per_file,
        })
    }
}
