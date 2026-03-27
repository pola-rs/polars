use std::sync::Arc;

use polars::prelude::iceberg_sink_state::IcebergSinkState;
use polars::prelude::sink::SinkedPathsCallback;
use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{CloudScheme, PlanCallback, SpecialEq, UnifiedSinkArgs};
use polars_utils::python_function::PythonObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use crate::io::cloud_options::OptPyCloudOptions;
use crate::prelude::Wrap;

/// Interface to `class SinkOptions` on the Python side
pub struct PySinkOptions<'py>(Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for PySinkOptions<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.to_owned()))
    }
}

impl PySinkOptions<'_> {
    pub fn extract_unified_sink_args(
        &self,
        cloud_scheme: Option<CloudScheme>,
    ) -> PyResult<UnifiedSinkArgs> {
        #[derive(FromPyObject)]
        struct Extract<'a> {
            mkdir: bool,
            maintain_order: bool,
            sync_on_close: Option<Wrap<SyncOnCloseType>>,
            storage_options: OptPyCloudOptions<'a>,
            credential_provider: Option<Py<PyAny>>,
            sinked_paths_callback: Option<Py<PyAny>>,
        }

        let Extract {
            mkdir,
            maintain_order,
            sync_on_close,
            storage_options,
            credential_provider,
            sinked_paths_callback,
        } = self.0.extract()?;

        let cloud_options =
            storage_options.extract_opt_cloud_options(cloud_scheme, credential_provider)?;

        let sync_on_close = sync_on_close.map_or(SyncOnCloseType::default(), |x| x.0);

        let py = self.0.py();

        let sinked_paths_callback = if let Some(obj) = sinked_paths_callback {
            Some(
                if let Ok((callback_type, obj)) = obj.extract::<(PyBackedStr, Py<PyAny>)>(py) {
                    match &*callback_type {
                        "iceberg-commit" => {
                            let table_wrap: IcebergSinkState =
                                Python::attach(|py| obj.extract(py))?;

                            SinkedPathsCallback::IcebergCommit {
                                state: table_wrap,
                                orig_sink_state_obj: Some(SpecialEq::new(Arc::new(PythonObject(
                                    obj,
                                )))),
                            }
                        },
                        type_name => {
                            return Err(PyValueError::new_err(format!(
                                "unknown callback type '{type_name}'"
                            )));
                        },
                    }
                } else {
                    SinkedPathsCallback::Callback(PlanCallback::Python(SpecialEq::new(Arc::new(
                        PythonObject(obj),
                    ))))
                },
            )
        } else {
            None
        };

        let unified_sink_args = UnifiedSinkArgs {
            mkdir,
            maintain_order,
            sync_on_close,
            cloud_options: cloud_options.map(Arc::new),
            sinked_paths_callback,
        };

        Ok(unified_sink_args)
    }
}
