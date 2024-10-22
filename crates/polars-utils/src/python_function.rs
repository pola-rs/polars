use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
#[cfg(feature = "serde")]
use serde::ser::Error;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde")]
pub const PYTHON_SERDE_MAGIC_BYTE_MARK: &[u8] = "PLPYUDF".as_bytes();
pub static PYTHON_VERSION_MINOR: Lazy<u8> = Lazy::new(get_python_minor_version);

#[derive(Clone, Debug)]
pub struct PythonFunction(pub PyObject);

impl From<PyObject> for PythonFunction {
    fn from(value: PyObject) -> Self {
        Self(value)
    }
}

impl Eq for PythonFunction {}

impl PartialEq for PythonFunction {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            let eq = self.0.getattr(py, "__eq__").unwrap();
            eq.call1(py, (other.0.clone_ref(py),))
                .unwrap()
                .extract::<bool>(py)
                // equality can be not implemented, so default to false
                .unwrap_or(false)
        })
    }
}

#[cfg(feature = "serde")]
impl Serialize for PythonFunction {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Python::with_gil(|py| {
            let pickle = PyModule::import_bound(py, "cloudpickle")
                .or_else(|_| PyModule::import_bound(py, "pickle"))
                .expect("unable to import 'cloudpickle' or 'pickle'")
                .getattr("dumps")
                .unwrap();

            let python_function = self.0.clone_ref(py);

            let dumped = pickle
                .call1((python_function,))
                .map_err(|s| S::Error::custom(format!("cannot pickle {s}")))?;
            let dumped = dumped.extract::<PyBackedBytes>().unwrap();

            serializer.serialize_bytes(&dumped)
        })
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for PythonFunction {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        let bytes = Vec::<u8>::deserialize(deserializer)?;

        Python::with_gil(|py| {
            let pickle = PyModule::import_bound(py, "pickle")
                .expect("unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new_bound(py, &bytes),);
            let python_function = pickle
                .call1(arg)
                .map_err(|s| D::Error::custom(format!("cannot pickle {s}")))?;

            Ok(Self(python_function.into()))
        })
    }
}

/// Get the minor Python version from the `sys` module.
fn get_python_minor_version() -> u8 {
    Python::with_gil(|py| {
        PyModule::import_bound(py, "sys")
            .unwrap()
            .getattr("version_info")
            .unwrap()
            .getattr("minor")
            .unwrap()
            .extract()
            .unwrap()
    })
}
