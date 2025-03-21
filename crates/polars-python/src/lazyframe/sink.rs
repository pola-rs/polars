use std::path::{Path, PathBuf};

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{PartitionVariant, SinkOptions};
use polars_utils::IdxSize;
use polars_utils::python_function::{PythonFunction, PythonObject};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use pyo3::{Bound, FromPyObject, PyAny, PyObject, PyResult, pyclass, pymethods};

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
    #[pyo3(get)]
    pub base_path: PathBuf,
    pub file_path_cb: Option<PythonFunction>,
    pub variant: PartitionVariant,
}

#[cfg(feature = "pymethods")]
#[pymethods]
impl PyPartitioning {
    #[staticmethod]
    #[pyo3(signature = (base_path, file_path_cb, max_size))]
    pub fn new_max_size(
        base_path: PathBuf,
        file_path_cb: Option<PyObject>,
        max_size: IdxSize,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        PyPartitioning {
            base_path,
            file_path_cb,
            variant: PartitionVariant::MaxSize(max_size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base_path, file_path_cb, by, include_key))]
    pub fn new_by_key(
        base_path: PathBuf,
        file_path_cb: Option<PyObject>,
        by: Vec<PyExpr>,
        include_key: bool,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        PyPartitioning {
            base_path,
            file_path_cb,
            variant: PartitionVariant::ByKey {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base_path, file_path_cb, by, include_key))]
    pub fn new_parted(
        base_path: PathBuf,
        file_path_cb: Option<PyObject>,
        by: Vec<PyExpr>,
        include_key: bool,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        PyPartitioning {
            base_path,
            file_path_cb,
            variant: PartitionVariant::Parted {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
        }
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
    pub fn base_path(&self) -> &Path {
        match self {
            Self::Path(path) => path.as_path(),
            Self::Partition(partition) => partition.base_path.as_path(),
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
