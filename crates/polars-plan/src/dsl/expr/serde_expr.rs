use polars_utils::pl_serialize::deserialize_map_bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::*;

impl Serialize for SpecialEq<Arc<dyn ColumnsUdf>> {
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

const NAMED_SERDE_MAGIC_BYTE_MARK: &[u8] = "PLNAMEDFN".as_bytes();

impl<T: Serialize + Clone> Serialize for LazySerde<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Named(name) => {
                let mut buf = vec![];
                buf.extend_from_slice(NAMED_SERDE_MAGIC_BYTE_MARK);
                buf.extend_from_slice(name.as_bytes());
                serializer.serialize_bytes(&buf)
            },
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

// impl<T: Deserialize> Deserialize for crate::dsl::expr::LazySerde<T> {
impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        deserialize_map_bytes(deserializer, |buf| {
            #[cfg(feature = "python")]
            if buf.starts_with(crate::dsl::python_dsl::PYTHON_SERDE_MAGIC_BYTE_MARK) {
                let udf = crate::dsl::python_dsl::PythonUdfExpression::try_deserialize(&buf)
                    .map_err(|e| D::Error::custom(format!("{e}")))?;
                return Ok(SpecialEq::new(udf));
            };

            if buf.starts_with(NAMED_SERDE_MAGIC_BYTE_MARK) {
                let bytes = &buf[NAMED_SERDE_MAGIC_BYTE_MARK.len()..];
                let Ok(name) = std::str::from_utf8(bytes) else {
                    return Err(D::Error::custom("named-serde name should be valid utf8"));
                };

                let registry = named_serde::NAMED_SERDE_REGISTRY_EXPR.read().unwrap();
                let msg = match &*registry {
                    Some(reg) => {
                        if let Some(func) = reg.get_function(name) {
                            return Ok(SpecialEq::new(func));
                        } else {
                            "name not found in named serde registry"
                        }
                    },
                    None => "named serde registry not set",
                };

                Err(D::Error::custom(
                    "deserialization not supported for this 'opaque' function",
                ))
            } else {
                Err(D::Error::custom(
                    "deserialization not supported for this 'opaque' function",
                ))
            }
        })?
    }
}

impl Serialize for GetOutput {
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

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for GetOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        #[cfg(feature = "python")]
        {
            deserialize_map_bytes(deserializer, |buf| {
                if buf.starts_with(self::python_dsl::PYTHON_SERDE_MAGIC_BYTE_MARK) {
                    let get_output = self::python_dsl::PythonGetOutput::try_deserialize(&buf)
                        .map_err(|e| D::Error::custom(format!("{e}")))?;
                    Ok(SpecialEq::new(get_output))
                } else {
                    Err(D::Error::custom(
                        "deserialization not supported for this output field",
                    ))
                }
            })?
        }
        #[cfg(not(feature = "python"))]
        {
            _ = deserializer;

            Err(D::Error::custom(
                "deserialization not supported for this output field",
            ))
        }
    }
}

impl Serialize for SpecialEq<Arc<dyn RenameAliasFn>> {
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

impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn RenameAliasFn>> {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "deserialization not supported for this renaming function",
        ))
    }
}

impl Serialize for SpecialEq<Series> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s: &Series = &self;
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
