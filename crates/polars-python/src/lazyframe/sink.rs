use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{PartitionVariant, SinkOptions};
use polars_utils::IdxSize;
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use pyo3::{Bound, FromPyObject, PyAny, PyResult, pyclass, pymethods};

use crate::expr::PyExpr;
use crate::prelude::Wrap;

#[derive(Clone)]
pub enum SinkTarget {
    Path(PathBuf),
    Partition(PyPartitioning),
}

#[pyclass]
#[derive(Clone)]
pub struct PyPartitioning {
    pub path: Arc<PathBuf>,
    pub variant: PartitionVariant,
}

#[cfg(feature = "pymethods")]
#[pymethods]
impl PyPartitioning {
    #[staticmethod]
    pub fn new_max_size(path: PathBuf, max_size: IdxSize) -> PyPartitioning {
        PyPartitioning {
            path: Arc::new(path),
            variant: PartitionVariant::MaxSize(max_size),
        }
    }

    #[staticmethod]
    pub fn new_by_key(path: PathBuf, by: Vec<PyExpr>, include_key: bool) -> PyPartitioning {
        PyPartitioning {
            path: Arc::new(path),
            variant: PartitionVariant::ByKey {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
        }
    }

    #[staticmethod]
    pub fn new_parted(path: PathBuf, by: Vec<PyExpr>, include_key: bool) -> PyPartitioning {
        PyPartitioning {
            path: Arc::new(path),
            variant: PartitionVariant::Parted {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
        }
    }

    #[getter]
    fn path(&self) -> &str {
        self.path.to_str().unwrap()
    }
}

impl<'py> FromPyObject<'py> for SinkTarget {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyPartitioning>() {
            return Ok(Self::Partition(v));
        }

        Ok(Self::Path(ob.extract::<PathBuf>()?))
    }
}

impl SinkTarget {
    pub fn unformatted_path(&self) -> &Path {
        match self {
            Self::Path(path) => path.as_path(),
            Self::Partition(partition) => partition.path.as_ref().as_path(),
        }
    }
}

impl<'py> FromPyObject<'py> for Wrap<SyncOnCloseType> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "none" => SyncOnCloseType::None,
            "data" => SyncOnCloseType::Data,
            "all" => SyncOnCloseType::All,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`sync_on_close` must be one of {{'none', 'data', 'all'}}, got {v}",
                )));
            },
        };
        Ok(Wrap(parsed))
    }
}

impl<'py> FromPyObject<'py> for Wrap<SinkOptions> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = ob.extract::<pyo3::Bound<'_, PyDict>>()?;

        if parsed.len() != 3 {
            return Err(PyValueError::new_err(
                "`sink_options` must be a dictionary with the exactly 3 field.",
            ));
        }

        let sync_on_close =
            PyDictMethods::get_item(&parsed, "sync_on_close")?.ok_or_else(|| {
                PyValueError::new_err("`sink_options` must contain `sync_on_close` field")
            })?;
        let sync_on_close = sync_on_close.extract::<Wrap<SyncOnCloseType>>()?.0;

        let maintain_order =
            PyDictMethods::get_item(&parsed, "maintain_order")?.ok_or_else(|| {
                PyValueError::new_err("`sink_options` must contain `maintain_order` field")
            })?;
        let maintain_order = maintain_order.extract::<bool>()?;

        let mkdir = PyDictMethods::get_item(&parsed, "mkdir")?
            .ok_or_else(|| PyValueError::new_err("`sink_options` must contain `mkdir` field"))?;
        let mkdir = mkdir.extract::<bool>()?;

        Ok(Wrap(SinkOptions {
            sync_on_close,
            maintain_order,
            mkdir,
        }))
    }
}
