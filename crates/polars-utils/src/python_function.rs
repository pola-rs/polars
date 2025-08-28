use pyo3::prelude::*;
#[cfg(feature = "serde")]
pub use serde_wrap::{
    PYTHON3_VERSION, PySerializeWrap, SERDE_MAGIC_BYTE_MARK as PYTHON_SERDE_MAGIC_BYTE_MARK,
    TrySerializeToBytes,
};

/// Wrapper around PyObject from pyo3 with additional trait impls.
#[derive(Debug)]
pub struct PythonObject(pub PyObject);
// Note: We have this because the struct itself used to be called `PythonFunction`, so it's
// referred to as such from a lot of places.
pub type PythonFunction = PythonObject;

impl std::ops::Deref for PythonObject {
    type Target = PyObject;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PythonObject {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Clone for PythonObject {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self(self.0.clone_ref(py)))
    }
}

impl From<PyObject> for PythonObject {
    fn from(value: PyObject) -> Self {
        Self(value)
    }
}

impl<'py> pyo3::conversion::IntoPyObject<'py> for PythonObject {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.into_bound(py))
    }
}

impl<'py> pyo3::conversion::IntoPyObject<'py> for &PythonObject {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.bind(py).clone())
    }
}

impl Eq for PythonObject {}

impl PartialEq for PythonObject {
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

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PythonObject {
    fn schema_name() -> String {
        "PythonObject".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PythonObject"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

#[cfg(feature = "serde")]
mod _serde_impls {
    use super::{PySerializeWrap, PythonObject, TrySerializeToBytes};
    use crate::pl_serialize::deserialize_map_bytes;

    impl PythonObject {
        pub fn serialize_with_pyversion<T, S>(
            value: &T,
            serializer: S,
        ) -> std::result::Result<S::Ok, S::Error>
        where
            T: AsRef<PythonObject>,
            S: serde::ser::Serializer,
        {
            use serde::Serialize;
            PySerializeWrap(value.as_ref()).serialize(serializer)
        }

        pub fn deserialize_with_pyversion<'de, T, D>(d: D) -> Result<T, D::Error>
        where
            T: From<PythonObject>,
            D: serde::de::Deserializer<'de>,
        {
            use serde::Deserialize;
            let v: PySerializeWrap<PythonObject> = PySerializeWrap::deserialize(d)?;

            Ok(v.0.into())
        }
    }

    impl TrySerializeToBytes for PythonObject {
        fn try_serialize_to_bytes(&self) -> polars_error::PolarsResult<Vec<u8>> {
            let mut buf = Vec::new();
            crate::pl_serialize::python_object_serialize(&self.0, &mut buf)?;
            Ok(buf)
        }

        fn try_deserialize_bytes(bytes: &[u8]) -> polars_error::PolarsResult<Self> {
            crate::pl_serialize::python_object_deserialize(bytes).map(PythonObject)
        }
    }

    impl serde::Serialize for PythonObject {
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

    impl<'a> serde::Deserialize<'a> for PythonObject {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'a>,
        {
            use serde::de::Error;
            deserialize_map_bytes(deserializer, |bytes| {
                Self::try_deserialize_bytes(&bytes).map_err(|e| D::Error::custom(e.to_string()))
            })?
        }
    }
}

#[cfg(feature = "serde")]
mod serde_wrap {
    use std::sync::LazyLock;

    use polars_error::PolarsResult;

    use crate::pl_serialize::deserialize_map_bytes;

    pub const SERDE_MAGIC_BYTE_MARK: &[u8] = "PLPYFN".as_bytes();
    /// [minor, micro]
    pub static PYTHON3_VERSION: LazyLock<[u8; 2]> = LazyLock::new(super::get_python3_version);

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

            serializer.serialize_bytes(dumped.as_slice())
        }
    }

    impl<'a, T: TrySerializeToBytes> serde::Deserialize<'a> for PySerializeWrap<T> {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'a>,
        {
            use serde::de::Error;

            deserialize_map_bytes(deserializer, |bytes| {
                T::try_deserialize_bytes(bytes.as_ref())
                    .map(Self)
                    .map_err(|e| D::Error::custom(e.to_string()))
            })?
        }
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
