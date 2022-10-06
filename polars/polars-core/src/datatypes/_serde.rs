//! Having `Object<&;static> in [`DataType`] make serde tag the `Deserialize` trait bound 'static
//! even though we skip serializing `Object`.
//!
//! We could use https://github.com/serde-rs/serde/issues/1712, but that gave problems caused by
//! https://github.com/rust-lang/rust/issues/96956, so we make a dummy type without static
pub use arrow::datatypes::DataType as ArrowDataType;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::*;

impl<'a> Deserialize<'a> for DataType {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        Ok(SerializableDataType::deserialize(deserializer)?.into())
    }
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let dt: SerializableDataType = self.into();
        dt.serialize(serializer)
    }
}

#[derive(Serialize, Deserialize)]
pub enum SerializableDataType {
    Boolean,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    /// String data
    Utf8,
    #[cfg(feature = "dtype-binary")]
    Binary,
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    Date,
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in the given ms/us/ns TimeUnit (64 bits).
    Datetime(TimeUnit, Option<TimeZone>),
    // 64-bit integer representing difference between times in milli|micro|nano seconds
    Duration(TimeUnit),
    /// A 64-bit time representing elapsed time since midnight in the given TimeUnit.
    Time,
    List(Box<SerializableDataType>),
    Null,
    #[cfg(feature = "dtype-struct")]
    Struct(Vec<Field>),
    // some logical types we cannot know statically, e.g. Datetime
    Unknown,
}

impl From<&DataType> for SerializableDataType {
    fn from(dt: &DataType) -> Self {
        use DataType::*;
        match dt {
            Boolean => Self::Boolean,
            UInt8 => Self::UInt8,
            UInt16 => Self::UInt16,
            UInt32 => Self::UInt32,
            UInt64 => Self::UInt64,
            Int8 => Self::Int8,
            Int16 => Self::Int16,
            Int32 => Self::Int32,
            Int64 => Self::Int64,
            Float32 => Self::Float32,
            Float64 => Self::Float64,
            Utf8 => Self::Utf8,
            #[cfg(feature = "dtype-binary")]
            Binary => Self::Binary,
            Date => Self::Date,
            Datetime(tu, tz) => Self::Datetime(*tu, tz.clone()),
            Duration(tu) => Self::Duration(*tu),
            Time => Self::Time,
            List(dt) => Self::List(Box::new(dt.as_ref().into())),
            Null => Self::Null,
            Unknown => Self::Unknown,
            #[cfg(feature = "dtype-struct")]
            Struct(flds) => Self::Struct(flds.clone()),
            _ => todo!(),
        }
    }
}
impl From<SerializableDataType> for DataType {
    fn from(dt: SerializableDataType) -> Self {
        use SerializableDataType::*;
        match dt {
            Boolean => Self::Boolean,
            UInt8 => Self::UInt8,
            UInt16 => Self::UInt16,
            UInt32 => Self::UInt32,
            UInt64 => Self::UInt64,
            Int8 => Self::Int8,
            Int16 => Self::Int16,
            Int32 => Self::Int32,
            Int64 => Self::Int64,
            Float32 => Self::Float32,
            Float64 => Self::Float64,
            Utf8 => Self::Utf8,
            #[cfg(feature = "dtype-binary")]
            Binary => Self::Binary,
            Date => Self::Date,
            Datetime(tu, tz) => Self::Datetime(tu, tz),
            Duration(tu) => Self::Duration(tu),
            Time => Self::Time,
            List(dt) => Self::List(Box::new((*dt).into())),
            Null => Self::Null,
            Unknown => Self::Unknown,
            #[cfg(feature = "dtype-struct")]
            Struct(flds) => Self::Struct(flds),
        }
    }
}
