use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_parquet::write::KeyValue;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
#[cfg(feature = "python")]
use pyo3::PyObject;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize, de, ser};

/// Context that can be used to construct custom file-level key value metadata for a Parquet file.
pub struct ParquetMetadataContext<'a> {
    pub arrow_schema: &'a str,
}

/// Key/value pairs that can be attached to a Parquet file as file-level metadtaa.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum KeyValueMetadata {
    /// Static key value metadata.
    Static(
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "serialize_vec_key_value",
                deserialize_with = "deserialize_vec_key_value"
            )
        )]
        Vec<KeyValue>,
    ),
    /// Rust function to dynamically compute key value metadata.
    DynamicRust(RustKeyValueMetadataFunction),
    /// Python function to dynamically compute key value metadata.
    #[cfg(feature = "python")]
    DynamicPython(python_impl::PythonKeyValueMetadataFunction),
}

#[cfg(feature = "serde")]
fn serialize_vec_key_value<S>(kv: &[KeyValue], serializer: S) -> Result<S::Ok, S::Error>
where
    S: ser::Serializer,
{
    kv.iter()
        .map(|item| (&item.key, item.value.as_ref()))
        .collect::<Vec<_>>()
        .serialize(serializer)
}

#[cfg(feature = "serde")]
fn deserialize_vec_key_value<'de, D>(deserializer: D) -> Result<Vec<KeyValue>, D::Error>
where
    D: de::Deserializer<'de>,
{
    let data = Vec::<(String, Option<String>)>::deserialize(deserializer)?;
    let result = data
        .into_iter()
        .map(|(key, value)| KeyValue { key, value })
        .collect::<Vec<_>>();
    Ok(result)
}

impl KeyValueMetadata {
    /// Create a key value metadata object from a static key value mapping.
    pub fn from_static(kv: Vec<(String, String)>) -> Self {
        Self::Static(
            kv.into_iter()
                .map(|(key, value)| KeyValue {
                    key,
                    value: Some(value),
                })
                .collect(),
        )
    }

    /// Create a key value metadata object from a Python function.
    #[cfg(feature = "python")]
    pub fn from_py_function(py_object: PyObject) -> Self {
        Self::DynamicPython(python_impl::PythonKeyValueMetadataFunction(Arc::new(
            PythonObject(py_object),
        )))
    }

    /// Turn the metadata into the key/value pairs to write to the Parquet file.
    /// The context is used to dynamically construct key/value pairs.
    pub fn collect(&self, ctx: ParquetMetadataContext) -> PolarsResult<Vec<KeyValue>> {
        match self {
            Self::Static(kv) => Ok(kv.clone()),
            Self::DynamicRust(func) => Ok(func.0(ctx)),
            #[cfg(feature = "python")]
            Self::DynamicPython(py_func) => py_func.call(ctx),
        }
    }
}

#[derive(Clone)]
pub struct RustKeyValueMetadataFunction(
    Arc<dyn Fn(ParquetMetadataContext) -> Vec<KeyValue> + Send + Sync>,
);

impl Debug for RustKeyValueMetadataFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "key value metadata function at 0x{:016x}",
            self.0.as_ref() as *const _ as *const () as usize
        )
    }
}

impl Eq for RustKeyValueMetadataFunction {}

impl PartialEq for RustKeyValueMetadataFunction {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Hash for RustKeyValueMetadataFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as *const () as usize);
    }
}

#[cfg(feature = "serde")]
impl Serialize for RustKeyValueMetadataFunction {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        Err(S::Error::custom(format!("cannot serialize {:?}", self)))
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RustKeyValueMetadataFunction {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "cannot deserialize RustKeyValueMetadataFn",
        ))
    }
}

#[cfg(feature = "python")]
mod python_impl {
    use std::hash::Hash;
    use std::sync::Arc;

    use polars_error::{PolarsResult, to_compute_err};
    use polars_parquet::write::KeyValue;
    use polars_utils::python_function::PythonObject;
    use pyo3::types::PyAnyMethods;
    use pyo3::{PyResult, Python, pyclass};
    use serde::{Deserialize, Serialize};

    use super::ParquetMetadataContext;

    #[derive(Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct PythonKeyValueMetadataFunction(
        #[cfg(feature = "python")]
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "PythonObject::serialize_with_pyversion",
                deserialize_with = "PythonObject::deserialize_with_pyversion"
            )
        )]
        pub Arc<polars_utils::python_function::PythonFunction>,
    );

    impl PythonKeyValueMetadataFunction {
        pub fn call(&self, ctx: ParquetMetadataContext) -> PolarsResult<Vec<KeyValue>> {
            let ctx = PythonParquetMetadataContext::from_key_value_metadata_context(ctx);
            Python::with_gil(|py| {
                let args = (ctx,);
                let out: Vec<(String, String)> =
                    self.0.call1(py, args)?.into_bound(py).extract()?;
                let result = out
                    .into_iter()
                    .map(|item| KeyValue {
                        key: item.0,
                        value: Some(item.1),
                    })
                    .collect::<Vec<_>>();
                PyResult::Ok(result)
            })
            .map_err(to_compute_err)
        }
    }

    impl Hash for PythonKeyValueMetadataFunction {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write_usize(Arc::as_ptr(&self.0) as *const () as usize);
        }
    }

    #[pyclass]
    pub struct PythonParquetMetadataContext {
        #[pyo3(get)]
        arrow_schema: String,
    }

    impl PythonParquetMetadataContext {
        pub fn from_key_value_metadata_context(ctx: ParquetMetadataContext) -> Self {
            Self {
                arrow_schema: ctx.arrow_schema.to_string(),
            }
        }
    }
}
