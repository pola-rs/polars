//! Catalog properties and storage options of an Iceberg sink, as an ordered tree of values.

#[cfg(feature = "serde")]
use polars_utils::aliases::{InitHashMaps, PlIndexSet};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::total_ord::TotalOrdWrap;

/// Deepest nesting of maps and lists. Also stops a structure that contains itself.
pub const MAX_DEPTH: usize = 128;

/// A `dict[str, ...]` of properties, in insertion order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(transparent))]
#[cfg_attr(
    feature = "dsl-schema",
    derive(schemars::JsonSchema),
    schemars(transparent)
)]
pub struct IcebergProperties(pub Vec<(PlSmallStr, IcebergPropertyValue)>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergPropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(TotalOrdWrap<f64>),
    Str(PlSmallStr),
    List(Vec<IcebergPropertyValue>),
    Map(IcebergProperties),
    /// A Python value outside this tree, e.g. a session object, an int beyond 64 bits or a
    /// float that isn't finite.
    /// It works within the process and is refused when the plan is serialized.
    // Last, so that the other variants keep their serialized index.
    #[cfg(feature = "python")]
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    #[cfg_attr(feature = "dsl-schema", schemars(skip))]
    Opaque(OpaqueValue),
}

impl IcebergProperties {
    /// Refuse nesting deeper than [`MAX_DEPTH`] and duplicate keys. Iterative: a deserialized
    /// tree can be deeper than the stack.
    #[cfg(feature = "serde")]
    fn check(&self) -> Result<(), String> {
        let mut stack = Vec::new();
        self.check_keys(1, &mut stack)?;
        while let Some((value, depth)) = stack.pop() {
            match value {
                IcebergPropertyValue::List(values) => {
                    if depth + 1 > MAX_DEPTH {
                        return Err(too_deep());
                    }
                    stack.extend(values.iter().map(|v| (v, depth + 1)));
                },
                IcebergPropertyValue::Map(map) => map.check_keys(depth + 1, &mut stack)?,
                _ => {},
            }
        }
        Ok(())
    }

    #[cfg(feature = "serde")]
    fn check_keys<'a>(
        &'a self,
        depth: usize,
        stack: &mut Vec<(&'a IcebergPropertyValue, usize)>,
    ) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Err(too_deep());
        }
        let mut seen = PlIndexSet::with_capacity(self.0.len());
        for (key, value) in &self.0 {
            if !seen.insert(key) {
                return Err(format!("duplicate Iceberg property key '{key}'"));
            }
            stack.push((value, depth));
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
fn too_deep() -> String {
    format!("Iceberg properties nested deeper than {MAX_DEPTH} levels")
}

// Iterative, as `check` is: dropping a deep tree recursively could overflow the stack.
impl Drop for IcebergProperties {
    fn drop(&mut self) {
        let mut stack: Vec<_> = self.0.drain(..).map(|(_, value)| value).collect();
        while let Some(value) = stack.pop() {
            match value {
                IcebergPropertyValue::List(mut values) => stack.append(&mut values),
                IcebergPropertyValue::Map(mut map) => {
                    stack.extend(map.0.drain(..).map(|(_, value)| value))
                },
                _ => {},
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for IcebergProperties {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let properties = Self(Vec::deserialize(deserializer)?);
        properties.check().map_err(serde::de::Error::custom)?;
        Ok(properties)
    }
}

/// A Python value kept as is, for use within the process.
#[cfg(feature = "python")]
pub struct OpaqueValue {
    /// Where the value sits, e.g. `auth.session`, and its Python type, for the error when
    /// the plan is serialized.
    pub path: PlSmallStr,
    pub type_name: PlSmallStr,
    pub value: pyo3::Py<pyo3::PyAny>,
}

#[cfg(feature = "python")]
mod _opaque_impl {
    use pyo3::Python;

    use super::OpaqueValue;

    impl Clone for OpaqueValue {
        fn clone(&self) -> Self {
            Python::attach(|py| Self {
                path: self.path.clone(),
                type_name: self.type_name.clone(),
                value: self.value.clone_ref(py),
            })
        }
    }

    // Identity, for both equality and hashing.
    impl PartialEq for OpaqueValue {
        fn eq(&self, other: &Self) -> bool {
            self.value.as_ptr() == other.value.as_ptr()
        }
    }

    impl Eq for OpaqueValue {}

    impl std::hash::Hash for OpaqueValue {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            (self.value.as_ptr() as usize).hash(state)
        }
    }

    // The value itself may hold credentials: show where it is and its type only.
    impl std::fmt::Debug for OpaqueValue {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Opaque({}: {})", self.path, self.type_name)
        }
    }

    impl serde::Serialize for OpaqueValue {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(format!(
                "cannot serialize Iceberg property '{}' of type {}: \
                 only str, int, float, bool, None, list and dict values can be serialized",
                self.path, self.type_name
            )))
        }
    }
}

#[cfg(feature = "python")]
mod _python_impl {
    use polars_utils::aliases::{InitHashMaps, PlIndexSet};
    use polars_utils::pl_str::PlSmallStr;
    use polars_utils::total_ord::TotalOrdWrap;
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
    use pyo3::{Borrowed, IntoPyObjectExt};

    use super::{IcebergProperties, IcebergPropertyValue, MAX_DEPTH, OpaqueValue};

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

    fn too_deep(path: &str) -> PyErr {
        // The path of a structure that contains itself repeats: name its start only.
        let top = path.split(['.', '[']).next().unwrap_or(path);
        PyTypeError::new_err(format!(
            "Iceberg property '{top}' is nested deeper than {MAX_DEPTH} levels, \
             or contains itself"
        ))
    }

    impl IcebergProperties {
        fn extract_at(dict: &Bound<'_, PyDict>, path: &str, depth: usize) -> PyResult<Self> {
            if depth > MAX_DEPTH {
                return Err(too_deep(path));
            }
            let mut out = Vec::with_capacity(dict.len());
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
                let key = PlSmallStr::from_str(key.to_str()?);
                // Distinct `str` subclass keys can have equal text.
                if !seen.insert(key.clone()) {
                    return Err(PyTypeError::new_err(format!(
                        "duplicate Iceberg property key '{}'",
                        child_path(path, &key)
                    )));
                }
                let value =
                    IcebergPropertyValue::extract_at(&value, &child_path(path, &key), depth)?;
                out.push((key, value));
            }
            Ok(Self(out))
        }
    }

    impl IcebergPropertyValue {
        fn extract_at(ob: &Bound<'_, PyAny>, path: &str, depth: usize) -> PyResult<Self> {
            // Exact scalar types: `bool` is an `int` subclass, and an `int` or `str` subclass
            // (e.g. an enum) would come back as a plain `int` or `str`.
            if ob.is_none() {
                return Ok(Self::Null);
            }
            if let Ok(b) = ob.cast_exact::<PyBool>() {
                return Ok(Self::Bool(b.is_true()));
            }
            if ob.is_exact_instance_of::<PyInt>() {
                if let Ok(i) = ob.extract::<i64>() {
                    return Ok(Self::Int(i));
                }
            } else if let Ok(f) = ob.cast_exact::<PyFloat>() {
                // NaN and infinities can't round-trip through a JSON plan.
                if f.value().is_finite() {
                    return Ok(Self::Float(TotalOrdWrap(f.value())));
                }
            } else if let Ok(s) = ob.cast_exact::<PyString>() {
                return Ok(Self::Str(PlSmallStr::from_str(s.to_str()?)));
            } else if let Ok(dict) = ob.cast::<PyDict>() {
                // A mapping comes back as a plain `dict`, e.g. an `OrderedDict`.
                return IcebergProperties::extract_at(dict, path, depth + 1).map(Self::Map);
            } else if let Ok(list) = ob.cast_exact::<PyList>() {
                return Self::extract_items(list.iter(), path, depth);
            } else if let Ok(tuple) = ob.cast_exact::<PyTuple>() {
                return Self::extract_items(tuple.iter(), path, depth);
            }
            Ok(Self::Opaque(OpaqueValue {
                path: PlSmallStr::from_str(path),
                type_name: PlSmallStr::from_string(type_name(ob)),
                value: ob.clone().unbind(),
            }))
        }

        fn extract_items<'py>(
            items: impl Iterator<Item = Bound<'py, PyAny>>,
            path: &str,
            depth: usize,
        ) -> PyResult<Self> {
            if depth + 1 > MAX_DEPTH {
                return Err(too_deep(path));
            }
            items
                .enumerate()
                .map(|(i, item)| Self::extract_at(&item, &format!("{path}[{i}]"), depth + 1))
                .collect::<PyResult<Vec<_>>>()
                .map(Self::List)
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
            Self::extract_at(dict, "", 1)
        }
    }

    impl<'py> IntoPyObject<'py> for IcebergPropertyValue {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
            Ok(match self {
                Self::Null => py.None().into_bound(py),
                Self::Bool(b) => b.into_bound_py_any(py)?,
                Self::Int(i) => i.into_bound_py_any(py)?,
                Self::Float(f) => f.0.into_bound_py_any(py)?,
                Self::Str(s) => s.as_str().into_bound_py_any(py)?,
                Self::List(values) => PyList::new(
                    py,
                    values
                        .into_iter()
                        .map(|v| v.into_pyobject(py))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any(),
                Self::Map(map) => map.into_pyobject(py)?.into_any(),
                Self::Opaque(opaque) => opaque.value.into_bound(py),
            })
        }
    }

    impl<'py> IntoPyObject<'py> for IcebergProperties {
        type Target = PyDict;
        type Output = Bound<'py, PyDict>;
        type Error = PyErr;

        fn into_pyobject(mut self, py: Python<'py>) -> PyResult<Self::Output> {
            let dict = PyDict::new(py);
            for (key, value) in std::mem::take(&mut self.0) {
                dict.set_item(key.as_str(), value.into_pyobject(py)?)?;
            }
            Ok(dict)
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use polars_utils::pl_serialize::{deserialize_from_reader, serialize_to_bytes};

    use super::*;

    fn round_trip(properties: &IcebergProperties) -> polars_error::PolarsResult<IcebergProperties> {
        let bytes = serialize_to_bytes::<_, false>(properties)?;
        deserialize_from_reader::<_, _, false>(bytes.as_slice())
    }

    #[test]
    fn test_round_trip_keeps_values_and_order() {
        use IcebergPropertyValue as V;
        let properties = IcebergProperties(vec![
            ("uri".into(), V::Str("http://catalog".into())),
            (
                "auth".into(),
                V::Map(IcebergProperties(vec![
                    ("type".into(), V::Str("google".into())),
                    ("scopes".into(), V::List(vec![V::Str("a".into()), V::Null])),
                ])),
            ),
            ("max_retries".into(), V::Int(2)),
            ("nan".into(), V::Float(TotalOrdWrap(f64::NAN))),
            ("ratio".into(), V::Float(TotalOrdWrap(-0.5))),
            ("flag".into(), V::Bool(true)),
        ]);
        let back = round_trip(&properties).unwrap();
        assert_eq!(back, properties);
        let keys: Vec<_> = back.0.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["uri", "auth", "max_retries", "nan", "ratio", "flag"]);
    }

    #[test]
    fn test_deserialize_refuses_duplicate_keys() {
        use IcebergPropertyValue as V;
        let nested = V::Map(IcebergProperties(vec![
            ("c".into(), V::Null),
            ("c".into(), V::Null),
        ]));
        let err = round_trip(&IcebergProperties(vec![("b".into(), nested)])).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "{err}");
    }

    #[test]
    fn test_deserialize_refuses_deep_nesting_beyond_the_stack() {
        use polars_utils::pl_serialize::{deserialize_dsl, serialize_dsl};

        // A crafted plan nesting 100k lists, beyond what is ever serialized: decoded with
        // stack growth, then checked and dropped iteratively.
        let shallow = IcebergProperties(vec![(
            "k".into(),
            IcebergPropertyValue::List(vec![IcebergPropertyValue::Null]),
        )]);
        let mut bytes = vec![];
        serialize_dsl(&mut bytes, &shallow).unwrap();
        // MessagePack: {"List": [ ... ]} around the `Null` unit variant.
        let open = b"\x81\xa4List\x91";
        let at = bytes.windows(open.len()).position(|w| w == open).unwrap();
        let mut crafted = bytes[..at].to_vec();
        for _ in 0..100_000 {
            crafted.extend_from_slice(open);
        }
        crafted.extend_from_slice(&bytes[at + open.len()..]);

        // Refused by the decoder's own depth limit or by `check`, without a crash.
        deserialize_dsl::<IcebergProperties>(&mut crafted.as_slice()).unwrap_err();
    }

    #[test]
    fn test_deserialize_refuses_deep_nesting() {
        let mut value = IcebergPropertyValue::Null;
        for _ in 0..MAX_DEPTH {
            value = IcebergPropertyValue::Map(IcebergProperties(vec![("k".into(), value)]));
        }
        let err = round_trip(&IcebergProperties(vec![("k".into(), value)])).unwrap_err();
        assert!(err.to_string().contains("deeper"), "{err}");
    }
}
