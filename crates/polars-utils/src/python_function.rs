use polars_error::{polars_bail, PolarsError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
#[cfg(feature = "serde")]
pub use serde_wrap::{
    PySerializeWrap, TrySerializeToBytes, PYTHON3_VERSION,
    SERDE_MAGIC_BYTE_MARK as PYTHON_SERDE_MAGIC_BYTE_MARK,
};

use crate::pl_serialize::deserialize_map_bytes;

#[derive(Debug)]
pub struct PythonFunction(pub PyObject);

impl Clone for PythonFunction {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self(self.0.clone_ref(py)))
    }
}

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
impl serde::Serialize for PythonFunction {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        let bytes = self
            .try_serialize_to_bytes()
            .map_err(|e| S::Error::custom(e.to_string()))?;

        Vec::<u8>::serialize(&bytes, serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a> serde::Deserialize<'a> for PythonFunction {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        use serde::de::Error;
        deserialize_map_bytes(deserializer, &mut |bytes| {
            Self::try_deserialize_bytes(&bytes).map_err(|e| D::Error::custom(e.to_string()))
        })?
    }
}

#[cfg(feature = "serde")]
impl TrySerializeToBytes for PythonFunction {
    fn try_serialize_to_bytes(&self) -> polars_error::PolarsResult<Vec<u8>> {
        serde_wrap::serialize_pyobject_with_cloudpickle_fallback(&self.0)
    }

    fn try_deserialize_bytes(bytes: &[u8]) -> polars_error::PolarsResult<Self> {
        serde_wrap::deserialize_pyobject_bytes_maybe_cloudpickle(bytes)
    }
}

#[cfg(feature = "serde")]
mod serde_wrap {
    use once_cell::sync::Lazy;
    use polars_error::PolarsResult;

    use super::*;
    use crate::pl_serialize::deserialize_map_bytes;

    pub const SERDE_MAGIC_BYTE_MARK: &[u8] = "PLPYFN".as_bytes();
    /// [minor, micro]
    pub static PYTHON3_VERSION: Lazy<[u8; 2]> = Lazy::new(super::get_python3_version);

    /// Serializes a Python object without additional system metadata. This is intended to be used
    /// together with `PySerializeWrap`, which attaches e.g. Python version metadata.
    pub trait TrySerializeToBytes: Sized {
        fn try_serialize_to_bytes(&self) -> PolarsResult<Vec<u8>>;
        fn try_deserialize_bytes(bytes: &[u8]) -> PolarsResult<Self>;
    }

    /// Serialization wrapper for T: TrySerializeToBytes that attaches Python
    /// version metadata.
    pub struct PySerializeWrap<T>(pub T);

    impl<T: TrySerializeToBytes> serde::Serialize for PySerializeWrap<&T> {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            use serde::ser::Error;
            let dumped = self
                .0
                .try_serialize_to_bytes()
                .map_err(|e| S::Error::custom(e.to_string()))?;

            serializer.serialize_bytes(
                &[SERDE_MAGIC_BYTE_MARK, &*PYTHON3_VERSION, dumped.as_slice()].concat(),
            )
        }
    }

    impl<'a, T: TrySerializeToBytes> serde::Deserialize<'a> for PySerializeWrap<T> {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'a>,
        {
            use serde::de::Error;

            deserialize_map_bytes(deserializer, &mut |bytes| {
                let Some((magic, rem)) = bytes.split_at_checked(SERDE_MAGIC_BYTE_MARK.len()) else {
                    return Err(D::Error::custom(
                        "unexpected EOF when reading serialized pyobject version",
                    ));
                };

                if magic != SERDE_MAGIC_BYTE_MARK {
                    return Err(D::Error::custom(
                        "serialized pyobject did not begin with magic byte mark",
                    ));
                }

                let bytes = rem;

                let [a, b, rem @ ..] = bytes else {
                    return Err(D::Error::custom(
                        "unexpected EOF when reading serialized pyobject metadata",
                    ));
                };

                let py3_version = [*a, *b];
                // The validity of cloudpickle is check later when called `try_deserialize`.
                let used_cloud_pickle = rem.first();

                // Cloudpickle uses bytecode to serialize, which is unstable between versions
                // So we only allow strict python versions if cloudpickle is used.
                if py3_version != *PYTHON3_VERSION && used_cloud_pickle == Some(&1) {
                    return Err(D::Error::custom(format!(
                        "python version that pyobject was serialized with {:?} \
                        differs from system python version {:?}",
                        (3, py3_version[0], py3_version[1]),
                        (3, PYTHON3_VERSION[0], PYTHON3_VERSION[1]),
                    )));
                }

                let bytes = rem;

                T::try_deserialize_bytes(bytes)
                    .map(Self)
                    .map_err(|e| D::Error::custom(e.to_string()))
            })?
        }
    }

    pub fn serialize_pyobject_with_cloudpickle_fallback(
        py_object: &PyObject,
    ) -> PolarsResult<Vec<u8>> {
        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "pickle")
                .expect("unable to import 'pickle'")
                .getattr("dumps")
                .unwrap();

            let dumped = pickle.call1((py_object.clone_ref(py),));

            let (dumped, used_cloudpickle) = if let Ok(v) = dumped {
                (v, false)
            } else {
                let cloudpickle = PyModule::import(py, "cloudpickle")
                    .map_err(from_pyerr)?
                    .getattr("dumps")
                    .unwrap();
                let dumped = cloudpickle
                    .call1((py_object.clone_ref(py),))
                    .map_err(from_pyerr)?;
                (dumped, true)
            };

            let py_bytes = dumped.extract::<PyBackedBytes>().map_err(from_pyerr)?;

            Ok([&[used_cloudpickle as u8, b'C'][..], py_bytes.as_ref()].concat())
        })
    }

    pub fn deserialize_pyobject_bytes_maybe_cloudpickle<T: for<'a> From<PyObject>>(
        bytes: &[u8],
    ) -> PolarsResult<T> {
        // TODO: Actually deserialize with cloudpickle if it's set.
        let [used_cloudpickle @ 0 | used_cloudpickle @ 1, b'C', rem @ ..] = bytes else {
            polars_bail!(ComputeError: "deserialize_pyobject_bytes_maybe_cloudpickle: invalid start bytes")
        };

        let bytes = rem;

        Python::with_gil(|py| {
            let p = if *used_cloudpickle == 1 {
                "cloudpickle"
            } else {
                "pickle"
            };

            let pickle = PyModule::import(py, p)
                .expect("unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new(py, bytes),);
            let pyany_bound = pickle.call1(arg).map_err(from_pyerr)?;
            Ok(PyObject::from(pyany_bound).into())
        })
    }
}

/// Get the [minor, micro] Python3 version from the `sys` module.
fn get_python3_version() -> [u8; 2] {
    Python::with_gil(|py| {
        let version_info = PyModule::import(py, "sys")
            .unwrap()
            .getattr("version_info")
            .unwrap();

        [
            version_info.getattr("minor").unwrap().extract().unwrap(),
            version_info.getattr("micro").unwrap().extract().unwrap(),
        ]
    })
}

fn from_pyerr(e: PyErr) -> PolarsError {
    PolarsError::ComputeError(format!("error raised in python: {e}").into())
}
