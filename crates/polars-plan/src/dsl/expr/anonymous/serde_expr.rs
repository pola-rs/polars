use std::sync::Arc;

use polars_core::series::Series;
use polars_error::*;
use polars_utils::pl_serialize::deserialize_map_bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::named_serde::ExprRegistry;
use super::*;
use crate::dsl::LazySerde;

const NAMED_SERDE_MAGIC_BYTE_MARK: &[u8] = "PLNAMEDFN".as_bytes();
const NAMED_SERDE_MAGIC_BYTE_END: u8 = b'!';

fn serialize_named<S: Serializer>(
    serializer: S,
    name: &str,
    payload: Option<&[u8]>,
) -> Result<S::Ok, S::Error> {
    let mut buf = vec![];
    buf.extend_from_slice(NAMED_SERDE_MAGIC_BYTE_MARK);
    buf.extend_from_slice(name.as_bytes());
    buf.push(NAMED_SERDE_MAGIC_BYTE_END);
    if let Some(payload) = payload {
        buf.extend_from_slice(payload);
    }
    serializer.serialize_bytes(&buf)
}

fn deserialize_named_registry(buf: &[u8]) -> PolarsResult<(Arc<dyn ExprRegistry>, &str, &[u8])> {
    let bytes = &buf[NAMED_SERDE_MAGIC_BYTE_MARK.len()..];
    let Some(pos) = bytes.iter().position(|b| *b == NAMED_SERDE_MAGIC_BYTE_END) else {
        polars_bail!(ComputeError: "named-serde expected magic byte end")
    };

    let Ok(name) = std::str::from_utf8(&bytes[..pos]) else {
        polars_bail!(ComputeError: "named-serde name should be valid utf8")
    };
    let payload = &bytes[pos + 1..];

    let registry = named_serde::NAMED_SERDE_REGISTRY_EXPR.read().unwrap();
    match &*registry {
        Some(reg) => Ok((reg.clone(), name, payload)),
        None => polars_bail!(ComputeError: "named serde registry not set"),
    }
}

impl Serialize for SpecialEq<Arc<dyn AnonymousAgg>> {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unreachable!("should not be hit")
    }
}

impl Serialize for SpecialEq<Arc<dyn AnonymousColumnsUdf>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let mut buf = vec![];
        self.as_ref()
            .try_serialize(&mut buf)
            .map_err(|e| S::Error::custom(format!("{e}")))?;
        serializer.serialize_bytes(&buf)
    }
}

impl<T: Serialize + Clone> Serialize for LazySerde<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Named {
                name,
                payload,
                value: _,
            } => serialize_named(serializer, name, payload.as_deref()),
            Self::Deserialized(t) => t.serialize(serializer),
            Self::Bytes(b) => b.serialize(serializer),
        }
    }
}

impl<'a, T: Deserialize<'a> + Clone> Deserialize<'a> for LazySerde<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let buf = bytes::Bytes::deserialize(deserializer)?;
        Ok(Self::Bytes(buf))
    }
}

pub(super) fn deserialize_column_udf(buf: &[u8]) -> PolarsResult<Arc<dyn AnonymousColumnsUdf>> {
    #[cfg(feature = "python")]
    if buf.starts_with(crate::dsl::python_dsl::PYTHON_SERDE_MAGIC_BYTE_MARK) {
        return crate::dsl::python_dsl::PythonUdfExpression::try_deserialize(buf);
    };

    if buf.starts_with(NAMED_SERDE_MAGIC_BYTE_MARK) {
        let (reg, name, payload) = deserialize_named_registry(buf)?;

        if let Some(func) = reg.get_function(name, payload) {
            Ok(func)
        } else {
            polars_bail!(ComputeError: "name not found in named serde registry")
        }
    } else {
        polars_bail!(ComputeError: "deserialization not supported for this 'opaque' function")
    }
}
impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn AnonymousColumnsUdf>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        deserialize_map_bytes(deserializer, |buf| {
            deserialize_column_udf(&buf)
                .map_err(|e| D::Error::custom(format!("{e}")))
                .map(SpecialEq::new)
        })?
    }
}

impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn AnonymousAgg>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        deserialize_map_bytes(deserializer, |buf| {
            deserialize_anon_agg(&buf)
                .map_err(|e| D::Error::custom(format!("{e}")))
                .map(SpecialEq::new)
        })?
    }
}

pub(super) fn deserialize_anon_agg(buf: &[u8]) -> PolarsResult<Arc<dyn AnonymousAgg>> {
    if buf.starts_with(NAMED_SERDE_MAGIC_BYTE_MARK) {
        let (reg, name, payload) = deserialize_named_registry(buf)?;

        if let Some(func) = reg.get_agg(name, payload)? {
            Ok(func)
        } else {
            polars_bail!(ComputeError: "name not found in named serde registry")
        }
    } else {
        polars_bail!(ComputeError: "deserialization not supported for this 'opaque' function")
    }
}

// Serialize SpecialEq<T>

impl Serialize for SpecialEq<Series> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s: &Series = self;
        s.serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for SpecialEq<Series> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let t = Series::deserialize(deserializer)?;
        Ok(SpecialEq::new(t))
    }
}

impl<T: Serialize> Serialize for SpecialEq<Arc<T>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_ref().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a, T: Deserialize<'a>> Deserialize<'a> for SpecialEq<Arc<T>> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let t = T::deserialize(deserializer)?;
        Ok(SpecialEq::new(Arc::new(t)))
    }
}
