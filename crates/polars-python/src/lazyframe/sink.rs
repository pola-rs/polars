use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{InMemoryPartition, PartitionBase, PartitionVariant, SinkOptions, SpecialEq};
use polars_error::{PolarsResult, polars_bail, to_compute_err};
use polars_utils::IdxSize;
use polars_utils::python_function::{PythonFunction, PythonObject};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use pyo3::{
    Bound, FromPyObject, IntoPyObjectExt, Py, PyAny, PyObject, PyResult, Python, pyclass, pymethods,
};

use crate::expr::PyExpr;
use crate::file::EitherRustPythonFile;
use crate::prelude::Wrap;

#[derive(Clone)]
pub enum SinkTarget {
    File(polars_plan::dsl::SinkTarget),
    Partition(PyPartitioning),
}

#[pyclass]
#[derive(Clone)]
pub struct PyPartitioning {
    pub base: Wrap<PartitionBase>,
    pub file_path_cb: Option<PythonFunction>,
    pub variant: PartitionVariant,
}

#[cfg(feature = "pymethods")]
#[pymethods]
impl PyPartitioning {
    #[staticmethod]
    #[pyo3(signature = (base, file_path_cb, max_size))]
    pub fn new_max_size(
        base: Wrap<PartitionBase>,
        file_path_cb: Option<PyObject>,
        max_size: IdxSize,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        PyPartitioning {
            base,
            file_path_cb,
            variant: PartitionVariant::MaxSize(max_size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base, file_path_cb, by, include_key))]
    pub fn new_by_key(
        base: Wrap<PartitionBase>,
        file_path_cb: Option<PyObject>,
        by: Vec<PyExpr>,
        include_key: bool,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        PyPartitioning {
            base,
            file_path_cb,
            variant: PartitionVariant::ByKey {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base, file_path_cb, by, include_key))]
    pub fn new_parted(
        base: Wrap<PartitionBase>,
        file_path_cb: Option<PyObject>,
        by: Vec<PyExpr>,
        include_key: bool,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        PyPartitioning {
            base,
            file_path_cb,
            variant: PartitionVariant::Parted {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
        }
    }

    #[getter]
    pub fn base_path(&self) -> Option<&Path> {
        match &self.base.0 {
            PartitionBase::Path(p) => Some(p.as_path()),
            PartitionBase::Memory(_) => None,
        }
    }
}

impl<'py> FromPyObject<'py> for SinkTarget {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyPartitioning>() {
            Ok(Self::Partition(v))
        } else if let Ok(v) = ob.extract::<PathBuf>() {
            Ok(Self::File(polars::prelude::SinkTarget::Path(Arc::new(v))))
        } else {
            let writer = Python::with_gil(|py| {
                let py_f = ob.clone();
                PyResult::Ok(match crate::file::try_get_pyfile(py, py_f, true)?.0 {
                    EitherRustPythonFile::Py(py_file_like_object) => {
                        Box::new(py_file_like_object) as _
                    },
                    EitherRustPythonFile::Rust(closable_file) => Box::new(closable_file) as _,
                })
            })?;

            Ok(Self::File(polars::prelude::SinkTarget::Memory(
                SpecialEq::new(Arc::new(Mutex::new(Some(writer)))),
            )))
        }
    }
}

struct PyDictInMemoryPartition(Py<PyAny>);

impl InMemoryPartition for PyDictInMemoryPartition {
    fn open_partition(&mut self, path: PathBuf) -> PolarsResult<Box<dyn std::io::Write + Send>> {
        Python::with_gil(|py| {
            let io = py.import("io").map_err(to_compute_err)?;
            let bytes_io = io.getattr("BytesIO").map_err(to_compute_err)?;
            let bytes_io = bytes_io.call0().map_err(to_compute_err)?;

            let dict = self
                .0
                .extract::<pyo3::Bound<'_, PyDict>>(py)
                .map_err(to_compute_err)?;

            dict.set_item(path, bytes_io.clone())
                .map_err(to_compute_err)?;

            let EitherRustPythonFile::Py(py_file_like_object) =
                crate::file::try_get_pyfile(py, bytes_io, true)
                    .map_err(to_compute_err)?
                    .0
            else {
                polars_bail!(ComputeError: "expected python bytes io object");
            };

            PolarsResult::Ok(Box::new(py_file_like_object) as _)
        })
    }
}

impl<'py> FromPyObject<'py> for Wrap<PartitionBase> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<pyo3::Bound<'_, PyDict>>() {
            let dict = Python::with_gil(|py| v.into_py_any(py))?;
            Ok(Wrap(PartitionBase::Memory(SpecialEq::new(Arc::new(
                Mutex::new(Box::new(PyDictInMemoryPartition(dict))),
            )))))
        } else {
            let path = ob.extract::<PathBuf>()?;
            Ok(Wrap(PartitionBase::Path(Arc::new(path))))
        }
    }
}

impl SinkTarget {
    pub fn base_path(&self) -> Option<&Path> {
        match self {
            Self::File(t) => match t {
                polars::prelude::SinkTarget::Path(p) => Some(p.as_path()),
                polars::prelude::SinkTarget::Memory(_) => None,
            },
            Self::Partition(p) => match &p.base.0 {
                PartitionBase::Path(p) => Some(p.as_path()),
                PartitionBase::Memory(_) => None,
            },
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
