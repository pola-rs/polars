use std::sync::{Arc, Mutex};

use polars::prelude::file_provider::FileProviderReturn;
use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{PlRefPath, SpecialEq};
use polars_error::polars_err;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use crate::prelude::Wrap;
use crate::utils::to_py_err;

impl<'a, 'py> FromPyObject<'a, 'py> for Wrap<polars_plan::dsl::SinkTarget> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(polars::prelude::SinkTarget::Path(PlRefPath::new(&*v))))
        } else {
            let writer = Python::attach(|py| {
                let py_f = ob.to_owned();
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

impl<'a, 'py> FromPyObject<'a, 'py> for Wrap<FileProviderReturn> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<PyBackedStr>() {
            Ok(Wrap(FileProviderReturn::Path(v.to_string())))
        } else if let Ok(v) = ob.extract::<std::path::PathBuf>() {
            Ok(Wrap(FileProviderReturn::Path(
                v.to_str()
                    .ok_or_else(|| to_py_err(polars_err!(non_utf8_path)))?
                    .to_string(),
            )))
        } else {
            let py = ob.py();

            let writeable = crate::file::try_get_pyfile(py, ob.to_owned(), true)?
                .0
                .into_writeable();

            Ok(Wrap(FileProviderReturn::Writeable(writeable)))
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Wrap<SyncOnCloseType> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
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
