use std::sync::Arc;

use polars_lazy::udf_registry;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::*;
use crate::py_modules::POLARS;

struct UdfSerializer(PyObject);

impl UdfSerializer {
    fn get_current(py: Python) -> Option<UdfSerializer> {
        use udf_registry::*;
        UDF_DESERIALIZE_REGISTRY.get_or_init(|| UdfSerializeRegistry {
            expr_series_udf: Registry::default()
                .with::<PyUdfLambda>("py-lambda", |e| Arc::new(e) as _),
            expr_fn_output_field: Registry::default()
                .with::<PyUdfLambda>("py-lambda", |e| Arc::new(e) as _),
            ..Default::default()
        });

        let pypolars = POLARS.cast_as::<PyModule>(py).unwrap();
        Some(UdfSerializer(
            pypolars
                .getattr("_current_udf_serializer")
                .ok()?
                .to_object(py),
        ))
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
            let py_obj = UdfSerializer::get_current(py)
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
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(PyString::new(self.0, &v).to_object(self.0))
            }
        }

        Python::with_gil(|py| {
            let udfser = UdfSerializer::get_current(py).ok_or_else(|| {
                serde::de::Error::custom(
                    "Cannot deserialize User-Defined Functions without an UDF serializer.",
                )
            })?;
            let py_obj_input = deserializer.deserialize_any(LVisitor(py))?;
            Ok(udfser.call_deserialize(py, py_obj_input))
        })
    }
}
