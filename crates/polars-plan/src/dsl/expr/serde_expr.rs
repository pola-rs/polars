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

impl<T: Serialize + Clone> Serialize for LazySerde<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Named(name) => todo!(),
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
        #[cfg(feature = "python")]
        {
            deserialize_map_bytes(deserializer, |buf| {
                if buf.starts_with(crate::dsl::python_dsl::PYTHON_SERDE_MAGIC_BYTE_MARK) {
                    let udf = crate::dsl::python_dsl::PythonUdfExpression::try_deserialize(&buf)
                        .map_err(|e| D::Error::custom(format!("{e}")))?;
                    Ok(SpecialEq::new(udf))
                } else {
                    Err(D::Error::custom(
                        "deserialization not supported for this 'opaque' function",
                    ))
                }
            })?
        }
        #[cfg(not(feature = "python"))]
        {
            _ = deserializer;

            Err(D::Error::custom(
                "deserialization not supported for this 'opaque' function",
            ))
        }
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
