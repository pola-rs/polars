use std::sync::{Arc, Mutex};

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{PartitionTargetCallbackResult, PlPath, SpecialEq};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, PyAny, PyResult, Python};

use crate::prelude::Wrap;

impl<'py> FromPyObject<'py> for Wrap<polars_plan::dsl::SinkTarget> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(polars::prelude::SinkTarget::Path(PlPath::new(&v))))
        } else {
            let writer = Python::attach(|py| {
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
            let writer = Python::attach(|py| {
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
