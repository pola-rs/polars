use std::sync::Arc;

use polars_lazy::dsl::{set_udf_deserializer, UdfDeserializer};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::*;
use crate::py_modules::POLARS;

pub struct PolarsUdfDeserializer;
impl UdfDeserializer for PolarsUdfDeserializer {
    fn deserialize_udf(
        &self,
        deserializer: &mut dyn polars_lazy::dsl::ErasedDeserializer,
    ) -> Result<Arc<dyn polars_lazy::dsl::SerializableUdf>, polars_lazy::dsl::ErasedError> {
        Ok(Arc::new(PyUdfLambda::deserialize(deserializer)?) as _)
    }
}

pub struct PyUdfSerializerObject(PyObject);

impl PyUdfSerializerObject {
    pub fn set_current(ser: Option<PyObject>) {
        // register our registry if not already done (lazy init)
        let _ = set_udf_deserializer(Box::new(PolarsUdfDeserializer));

        Python::with_gil(|py| {
            let pypolars = POLARS.cast_as::<PyModule>(py).unwrap();
            let val = ser.unwrap_or_else(|| py.None());
            pypolars.setattr("_current_udf_serializer", val).unwrap()
        })
    }

    fn get_current(py: Python) -> Option<PyUdfSerializerObject> {
        let pypolars = POLARS.cast_as::<PyModule>(py).unwrap();
        let val = pypolars.getattr("_current_udf_serializer").ok()?;

        if val.is_none() {
            return None;
        }
        Some(PyUdfSerializerObject(val.to_object(py)))
    }

    fn call_serialize(&self, py: Python, lambda: &PyLambda) -> PyObject {
        self.0
            .getattr(py, "serialize_udf")
            .unwrap()
            .call1(py, (lambda.0.clone(),))
            .unwrap()
    }

    fn call_deserialize(&self, py: Python, data: PyObject) -> PyLambda {
        PyLambda(
            self.0
                .getattr(py, "deserialize_udf")
                .unwrap()
                .call1(py, (data,))
                .unwrap(),
        )
    }
}

impl Serialize for PyLambda {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Python::with_gil(|py| {
            let py_obj = PyUdfSerializerObject::get_current(py)
                .ok_or_else(|| {
                    serde::ser::Error::custom(
                        "Cannot serialize User-Defined Functions without an UDF serializer.",
                    )
                })?
                .call_serialize(py, self);

            if let Ok(r) = py_obj.extract::<&[u8]>(py) {
                serializer.serialize_bytes(r)
            } else if let Ok(r) = py_obj.extract::<&str>(py) {
                serializer.serialize_str(r)
            } else {
                Err(serde::ser::Error::custom(format!(
                    "Serializer instance returned {py_obj} which cannot be serialized."
                )))
            }
        })
    }
}
impl<'de> Deserialize<'de> for PyLambda {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LVisitor<'py>(Python<'py>);
        impl<'py, 'de> serde::de::Visitor<'de> for LVisitor<'py> {
            type Value = PyObject;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a user-defined function")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(PyBytes::new(self.0, v).to_object(self.0))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(PyString::new(self.0, v).to_object(self.0))
            }
        }

        Python::with_gil(|py| {
            let udfser = PyUdfSerializerObject::get_current(py).ok_or_else(|| {
                serde::de::Error::custom(
                    "Cannot deserialize User-Defined Functions without an UDF serializer.",
                )
            })?;
            let py_obj_input = deserializer.deserialize_any(LVisitor(py))?;
            Ok(udfser.call_deserialize(py, py_obj_input))
        })
    }
}
