//! Catalog properties and storage options of an Iceberg sink.

use serde_json::{Map, Value};

/// Deepest nesting of dicts and lists read from Python, well within what serde_json reads back
/// (128). Also stops a dict that contains itself.
pub const MAX_DEPTH: usize = 64;

/// A `dict[str, ...]` of properties.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergProperties {
    /// A JSON object, in insertion order.
    Json(JsonProperties),
    /// A dict that JSON doesn't carry as is (a session object, an int beyond 64 bits, a tuple):
    /// works within the process, refused when the plan is serialized.
    #[cfg(feature = "python")]
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    #[cfg_attr(feature = "dsl-schema", schemars(skip))]
    Python(PythonProperties),
}

/// A JSON object, equal to another only with the same keys in the same order: of two aliases,
/// the later one wins.
///
/// Serialized as JSON text: bincode can't read a `serde_json::Value`, and a plan sent between
/// processes may go through bincode.
#[derive(Clone, Debug, Default, Eq)]
#[cfg_attr(
    feature = "dsl-schema",
    derive(schemars::JsonSchema),
    schemars(transparent)
)]
pub struct JsonProperties(
    #[cfg_attr(feature = "dsl-schema", schemars(with = "String"))] pub Map<String, Value>,
);

#[cfg(feature = "serde")]
impl serde::Serialize for JsonProperties {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let text = serde_json::to_string(&self.0).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&text)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for JsonProperties {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        serde_json::from_str(&text)
            .map(Self)
            .map_err(|e| serde::de::Error::custom(format!("Iceberg properties: {e}")))
    }
}

impl PartialEq for JsonProperties {
    fn eq(&self, other: &Self) -> bool {
        maps_eq_in_order(&self.0, &other.0)
    }
}

// Ignores the order, which is consistent with `eq`: equal in order is equal in any order.
impl std::hash::Hash for JsonProperties {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

fn maps_eq_in_order(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((ka, va), (kb, vb))| ka == kb && values_eq_in_order(va, vb))
}

fn values_eq_in_order(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => maps_eq_in_order(a, b),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| values_eq_in_order(a, b))
        },
        _ => a == b,
    }
}

/// A dict kept as the Python object it is.
#[cfg(feature = "python")]
pub struct PythonProperties {
    pub dict: pyo3::Py<pyo3::PyAny>,
    /// The first value JSON doesn't carry and its type, e.g. `'auth.session' of type Session`,
    /// for the error when the plan is serialized.
    pub reason: polars_utils::pl_str::PlSmallStr,
}

#[cfg(feature = "python")]
mod _python_impl {
    use polars_utils::aliases::{InitHashMaps, PlIndexSet};
    use polars_utils::pl_str::PlSmallStr;
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
    use pyo3::{Borrowed, intern};
    use serde_json::{Map, Number, Value};

    use super::{IcebergProperties, JsonProperties, MAX_DEPTH, PythonProperties};

    impl Clone for PythonProperties {
        fn clone(&self) -> Self {
            Python::attach(|py| Self {
                dict: self.dict.clone_ref(py),
                reason: self.reason.clone(),
            })
        }
    }

    // Identity, for both equality and hashing.
    impl PartialEq for PythonProperties {
        fn eq(&self, other: &Self) -> bool {
            self.dict.as_ptr() == other.dict.as_ptr()
        }
    }

    impl Eq for PythonProperties {}

    impl std::hash::Hash for PythonProperties {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            (self.dict.as_ptr() as usize).hash(state)
        }
    }

    // The values may hold credentials: show why the dict is kept as is, not what it holds.
    impl std::fmt::Debug for PythonProperties {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.reason)
        }
    }

    impl serde::Serialize for PythonProperties {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(format!(
                "cannot serialize Iceberg property {}: only str, int, float, bool, None, list \
                 and dict values can be serialized",
                self.reason
            )))
        }
    }

    fn type_name(ob: &Bound<'_, PyAny>) -> String {
        let ty = ob.get_type();
        match (ty.module(), ty.qualname()) {
            (Ok(module), Ok(qualname)) if module.to_str().is_ok_and(|m| m != "builtins") => {
                format!("{module}.{qualname}")
            },
            (_, Ok(qualname)) => qualname.to_string(),
            _ => "<unknown>".to_string(),
        }
    }

    fn child_path(path: &str, key: &str) -> String {
        if path.is_empty() {
            key.to_string()
        } else {
            format!("{path}.{key}")
        }
    }

    /// Reads a dict as JSON, noting the first value that JSON doesn't carry.
    #[derive(Default)]
    struct Reader {
        reason: Option<String>,
        /// The top-level key being read: the path of a dict that contains itself repeats.
        top: String,
    }

    impl Reader {
        fn too_deep(&self) -> PyErr {
            PyTypeError::new_err(format!(
                "Iceberg property '{}' is nested deeper than {MAX_DEPTH} levels, \
                 or contains itself",
                self.top
            ))
        }

        fn map(
            &mut self,
            dict: &Bound<'_, PyDict>,
            path: &str,
            depth: usize,
        ) -> PyResult<Map<String, Value>> {
            if depth > MAX_DEPTH {
                return Err(self.too_deep());
            }
            let mut out = Map::with_capacity(dict.len());
            let mut seen = PlIndexSet::with_capacity(dict.len());
            for (key, value) in dict.iter() {
                let Ok(key) = key.cast::<PyString>() else {
                    let at = if path.is_empty() {
                        "at the top level".to_string()
                    } else {
                        format!("in '{path}'")
                    };
                    return Err(PyTypeError::new_err(format!(
                        "Iceberg property keys must be str: got {} {at}",
                        type_name(&key)
                    )));
                };
                let key = key.to_str()?.to_string();
                if path.is_empty() {
                    self.top.clone_from(&key);
                }
                // Distinct `str` subclass keys can have equal text.
                if !seen.insert(key.clone()) {
                    return Err(PyTypeError::new_err(format!(
                        "duplicate Iceberg property key '{}'",
                        child_path(path, &key)
                    )));
                }
                let value = self.value(&value, &child_path(path, &key), depth)?;
                out.insert(key, value);
            }
            Ok(out)
        }

        /// A value of a dict or list at `depth`. Exact types only: `bool` is an `int` subclass,
        /// and other subclasses (e.g. an enum) or a tuple would come back as another type.
        fn value(&mut self, ob: &Bound<'_, PyAny>, path: &str, depth: usize) -> PyResult<Value> {
            if ob.is_none() {
                return Ok(Value::Null);
            }
            if let Ok(b) = ob.cast_exact::<PyBool>() {
                return Ok(Value::Bool(b.is_true()));
            }
            if ob.is_exact_instance_of::<PyInt>() {
                if let Ok(i) = ob.extract::<i64>() {
                    return Ok(i.into());
                }
                if let Ok(u) = ob.extract::<u64>() {
                    return Ok(u.into());
                }
            } else if let Ok(f) = ob.cast_exact::<PyFloat>() {
                // Not NaN or infinities.
                if let Some(n) = Number::from_f64(f.value()) {
                    return Ok(Value::Number(n));
                }
            } else if let Ok(s) = ob.cast_exact::<PyString>() {
                // Not lone surrogates, which UTF-8 can't hold.
                if let Ok(s) = s.to_str() {
                    return Ok(Value::String(s.to_string()));
                }
            } else if let Ok(dict) = ob.cast_exact::<PyDict>() {
                return self.map(dict, path, depth + 1).map(Value::Object);
            } else if let Ok(list) = ob.cast_exact::<PyList>() {
                if depth + 1 > MAX_DEPTH {
                    return Err(self.too_deep());
                }
                return list
                    .iter()
                    .enumerate()
                    .map(|(i, item)| self.value(&item, &format!("{path}[{i}]"), depth + 1))
                    .collect::<PyResult<Vec<_>>>()
                    .map(Value::Array);
            }
            // Keep reading: a bad key or a cycle further on is still an error.
            self.reason
                .get_or_insert_with(|| format!("'{path}' of type {}", type_name(ob)));
            Ok(Value::Null)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for IcebergProperties {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let ob = ob.to_owned();
            let Ok(dict) = ob.cast::<PyDict>() else {
                return Err(PyTypeError::new_err(format!(
                    "Iceberg properties must be a dict: got {}",
                    type_name(&ob)
                )));
            };
            let mut reader = Reader::default();
            let map = reader.map(dict, "", 1)?;
            Ok(match reader.reason {
                None => Self::Json(JsonProperties(map)),
                Some(reason) => Self::Python(PythonProperties {
                    dict: ob.unbind(),
                    reason: PlSmallStr::from_string(reason),
                }),
            })
        }
    }

    impl<'py> IntoPyObject<'py> for IcebergProperties {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
            match self {
                // `json.loads` keeps the key order and the number types.
                Self::Json(JsonProperties(map)) => {
                    let text = serde_json::to_string(&map)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.import(intern!(py, "json"))?
                        .call_method1(intern!(py, "loads"), (text,))
                },
                Self::Python(properties) => Ok(properties.dict.into_bound(py)),
            }
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use polars_utils::pl_serialize::{deserialize_from_reader, serialize_to_bytes};
    use serde_json::json;

    use super::*;

    fn properties(value: Value) -> IcebergProperties {
        let Value::Object(map) = value else {
            unreachable!()
        };
        IcebergProperties::Json(JsonProperties(map))
    }

    /// Round trips through bincode, MessagePack (as plans are) and JSON.
    fn round_trips(properties: &IcebergProperties) -> [IcebergProperties; 3] {
        use polars_utils::pl_serialize::{deserialize_dsl, serialize_dsl};

        let bincode = serialize_to_bytes::<_, false>(properties).unwrap();
        let mut msgpack = vec![];
        serialize_dsl(&mut msgpack, properties).unwrap();
        let json = serde_json::to_string(properties).unwrap();
        [
            deserialize_from_reader::<_, _, false>(bincode.as_slice()).unwrap(),
            deserialize_dsl(&mut msgpack.as_slice()).unwrap(),
            serde_json::from_str(&json).unwrap(),
        ]
    }

    #[test]
    fn test_round_trip_keeps_values_and_order() {
        let properties = properties(json!({
            "uri": "http://catalog",
            "auth": {"type": "google", "scopes": ["a", null]},
            "max_retries": 2,
            "big": u64::MAX,
            "small": i64::MIN,
            "ratio": -0.5,
            "tenth": 0.1,
            "zero": -0.0,
            "flag": true,
        }));
        for back in round_trips(&properties) {
            assert_eq!(back, properties);
            let IcebergProperties::Json(JsonProperties(map)) = back else {
                unreachable!()
            };
            let keys: Vec<_> = map.keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                [
                    "uri",
                    "auth",
                    "max_retries",
                    "big",
                    "small",
                    "ratio",
                    "tenth",
                    "zero",
                    "flag"
                ]
            );
            assert!(map["zero"].as_f64().unwrap().is_sign_negative());
        }
    }

    #[test]
    fn test_round_trip_keeps_floats_exact() {
        let floats = [
            51.248178375505404,
            0.30000000000000004,
            5e-324,
            2.2250738585072014e-308,
            1.7976931348623157e308,
            -1.0000000000000002,
        ];
        let properties = properties(json!({ "f": floats }));
        for back in round_trips(&properties) {
            let IcebergProperties::Json(JsonProperties(map)) = back else {
                unreachable!()
            };
            let got: Vec<u64> = map["f"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap().to_bits())
                .collect();
            assert_eq!(got, floats.map(f64::to_bits));
        }
    }

    #[test]
    fn test_round_trip_at_max_depth() {
        let mut value = Value::Null;
        for _ in 1..MAX_DEPTH {
            value = json!({ "k": value });
        }
        let properties = properties(json!({ "k": value }));
        for back in round_trips(&properties) {
            assert_eq!(back, properties);
        }
    }

    #[test]
    fn test_deserialize_malformed_text() {
        let err = serde_json::from_str::<JsonProperties>(r#""{\"a\": ""#).unwrap_err();
        assert!(err.to_string().starts_with("Iceberg properties: "), "{err}");
    }

    #[test]
    fn test_equality_follows_key_order() {
        use std::hash::BuildHasher;

        let a = properties(json!({"s3.region": "first", "aws_region": "second"}));
        let b = properties(json!({"aws_region": "second", "s3.region": "first"}));
        let nested_a = properties(json!({"x": {"a": 1, "b": [{"c": 1, "d": 2}]}}));
        let nested_b = properties(json!({"x": {"a": 1, "b": [{"d": 2, "c": 1}]}}));
        assert_ne!(a, b);
        assert_ne!(nested_a, nested_b);
        assert_eq!(a, a.clone());
        // Equal values hash equally, so the order-free hash is consistent.
        let hasher = std::hash::RandomState::new();
        assert_eq!(hasher.hash_one(&a), hasher.hash_one(&b));
    }
}
