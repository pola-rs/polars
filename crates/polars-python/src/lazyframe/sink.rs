use std::sync::{Arc, Mutex};

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{
    PartitionTargetCallbackResult, PartitionVariant, PlPath, SinkFinishCallback, SinkOptions,
    SortColumn, SpecialEq,
};
use polars_utils::IdxSize;
use polars_utils::plpath::PlPathRef;
use polars_utils::python_function::{PythonFunction, PythonObject};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use pyo3::{Bound, FromPyObject, PyAny, PyObject, PyResult, Python, pyclass, pymethods};

use crate::expr::PyExpr;
use crate::prelude::Wrap;

#[derive(Clone)]
pub enum SinkTarget {
    File(polars_plan::dsl::SinkTarget),
    Partition(PyPartitioning),
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyPartitioning {
    #[pyo3(get)]
    pub base_path: Wrap<PlPath>,
    pub file_path_cb: Option<PythonFunction>,
    pub variant: PartitionVariant,
    pub per_partition_sort_by: Option<Vec<SortColumn>>,
    pub finish_callback: Option<SinkFinishCallback>,
}

fn parse_per_partition_sort_by(sort_by: Option<Vec<PyExpr>>) -> Option<Vec<SortColumn>> {
    sort_by.map(|exprs| {
        exprs
            .into_iter()
            .map(|e| SortColumn {
                expr: e.inner,
                descending: false,
                nulls_last: false,
            })
            .collect()
    })
}

#[cfg(feature = "pymethods")]
#[pymethods]
impl PyPartitioning {
    #[staticmethod]
    #[pyo3(signature = (base_path, file_path_cb, max_size, per_partition_sort_by, finish_callback))]
    pub fn new_max_size(
        base_path: Wrap<PlPath>,
        file_path_cb: Option<PyObject>,
        max_size: IdxSize,
        per_partition_sort_by: Option<Vec<PyExpr>>,
        finish_callback: Option<PyObject>,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        let finish_callback =
            finish_callback.map(|f| SinkFinishCallback::Python(PythonObject(f.into_any())));

        PyPartitioning {
            base_path,
            file_path_cb,
            variant: PartitionVariant::MaxSize(max_size),
            per_partition_sort_by: parse_per_partition_sort_by(per_partition_sort_by),
            finish_callback,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base_path, file_path_cb, by, include_key, per_partition_sort_by, finish_callback))]
    pub fn new_by_key(
        base_path: Wrap<PlPath>,
        file_path_cb: Option<PyObject>,
        by: Vec<PyExpr>,
        include_key: bool,
        per_partition_sort_by: Option<Vec<PyExpr>>,
        finish_callback: Option<PyObject>,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        let finish_callback =
            finish_callback.map(|f| SinkFinishCallback::Python(PythonObject(f.into_any())));

        PyPartitioning {
            base_path,
            file_path_cb,
            variant: PartitionVariant::ByKey {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
            per_partition_sort_by: parse_per_partition_sort_by(per_partition_sort_by),
            finish_callback,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base_path, file_path_cb, by, include_key, per_partition_sort_by, finish_callback))]
    pub fn new_parted(
        base_path: Wrap<PlPath>,
        file_path_cb: Option<PyObject>,
        by: Vec<PyExpr>,
        include_key: bool,
        per_partition_sort_by: Option<Vec<PyExpr>>,
        finish_callback: Option<PyObject>,
    ) -> PyPartitioning {
        let file_path_cb = file_path_cb.map(|f| PythonObject(f.into_any()));
        let finish_callback =
            finish_callback.map(|f| SinkFinishCallback::Python(PythonObject(f.into_any())));

        PyPartitioning {
            base_path,
            file_path_cb,
            variant: PartitionVariant::Parted {
                key_exprs: by.into_iter().map(|e| e.inner).collect(),
                include_key,
            },
            per_partition_sort_by: parse_per_partition_sort_by(per_partition_sort_by),
            finish_callback,
        }
    }
}

impl<'py> FromPyObject<'py> for Wrap<polars_plan::dsl::SinkTarget> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(polars::prelude::SinkTarget::Path(PlPath::new(&v))))
        } else {
            let writer = Python::with_gil(|py| {
                let py_f = ob.clone();
                PyResult::Ok(
                    crate::file::try_get_pyfile(py, py_f, true)?
                        .0
                        .into_writeable(),
                )
            })?;

            Ok(Wrap(polars_plan::prelude::SinkTarget::Dyn(SpecialEq::new(
                Arc::new(Mutex::new(Some(writer))),
            ))))
        }
    }
}

impl<'py> FromPyObject<'py> for Wrap<PartitionTargetCallbackResult> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(polars::prelude::PartitionTargetCallbackResult::Str(
                v.to_string(),
            )))
        } else if let Ok(v) = ob.extract::<std::path::PathBuf>() {
            Ok(Wrap(polars::prelude::PartitionTargetCallbackResult::Str(
                v.to_str().unwrap().to_string(),
            )))
        } else {
            let writer = Python::with_gil(|py| {
                let py_f = ob.clone();
                PyResult::Ok(
                    crate::file::try_get_pyfile(py, py_f, true)?
                        .0
                        .into_writeable(),
                )
            })?;

            Ok(Wrap(
                polars_plan::prelude::PartitionTargetCallbackResult::Dyn(SpecialEq::new(Arc::new(
                    Mutex::new(Some(writer)),
                ))),
            ))
        }
    }
}

impl<'py> FromPyObject<'py> for SinkTarget {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyPartitioning>() {
            Ok(Self::Partition(v))
        } else {
            Ok(Self::File(
                <Wrap<polars_plan::dsl::SinkTarget>>::extract_bound(ob)?.0,
            ))
        }
    }
}

impl SinkTarget {
    pub fn base_path(&self) -> Option<PlPathRef<'_>> {
        match self {
            Self::File(t) => match t {
                polars::prelude::SinkTarget::Path(p) => Some(p.as_ref()),
                polars::prelude::SinkTarget::Dyn(_) => None,
            },
            Self::Partition(p) => Some(p.base_path.0.as_ref()),
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
