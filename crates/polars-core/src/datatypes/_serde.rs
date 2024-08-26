//! Having `Object<&;static> in [`DataType`] make serde tag the `Deserialize` trait bound 'static
//! even though we skip serializing `Object`.
//!
//! We could use [serde_1712](https://github.com/serde-rs/serde/issues/1712), but that gave problems caused by
//! [rust_96956](https://github.com/rust-lang/rust/issues/96956), so we make a dummy type without static

#[cfg(feature = "dtype-categorical")]
use serde::de::SeqAccess;
use serde::{Deserialize, Serialize};

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

#[cfg(feature = "dtype-categorical")]
struct Wrap<T>(T);

#[cfg(feature = "dtype-categorical")]
impl serde::Serialize for Wrap<Utf8ViewArray> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_seq(self.0.values_iter())
    }
}

#[cfg(feature = "dtype-categorical")]
impl<'de> serde::Deserialize<'de> for Wrap<Utf8ViewArray> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Utf8Visitor;

        impl<'de> Visitor<'de> for Utf8Visitor {
            type Value = Wrap<Utf8ViewArray>;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str("Utf8Visitor string sequence.")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut utf8array = MutablePlString::with_capacity(seq.size_hint().unwrap_or(10));
                while let Some(key) = seq.next_element()? {
                    let key: Option<String> = key;
                    utf8array.push(key)
                }
                Ok(Wrap(utf8array.into()))
            }
        }

        deserializer.deserialize_seq(Utf8Visitor)
    }
}

#[derive(Serialize, Deserialize)]
enum SerializableDataType {
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
    String,
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
    #[cfg(feature = "dtype-array")]
    Array(Box<SerializableDataType>, usize),
    Null,
    #[cfg(feature = "dtype-struct")]
    Struct(Vec<Field>),
    // some logical types we cannot know statically, e.g. Datetime
    Unknown(UnknownKind),
    #[cfg(feature = "dtype-categorical")]
    Categorical(Option<Wrap<Utf8ViewArray>>, CategoricalOrdering),
    #[cfg(feature = "dtype-decimal")]
    Decimal(Option<usize>, Option<usize>),
    #[cfg(feature = "dtype-categorical")]
    Enum(Option<Wrap<Utf8ViewArray>>, CategoricalOrdering),
    #[cfg(feature = "object")]
    Object(String),
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
            String => Self::String,
            Binary => Self::Binary,
            Date => Self::Date,
            Datetime(tu, tz) => Self::Datetime(*tu, tz.clone()),
            Duration(tu) => Self::Duration(*tu),
            Time => Self::Time,
            List(dt) => Self::List(Box::new(dt.as_ref().into())),
            #[cfg(feature = "dtype-array")]
            Array(dt, width) => Self::Array(Box::new(dt.as_ref().into()), *width),
            Null => Self::Null,
            Unknown(kind) => Self::Unknown(*kind),
            #[cfg(feature = "dtype-struct")]
            Struct(flds) => Self::Struct(flds.clone()),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, ordering) => Self::Categorical(None, *ordering),
            #[cfg(feature = "dtype-categorical")]
            Enum(Some(rev_map), ordering) => {
                Self::Enum(Some(Wrap(rev_map.get_categories().clone())), *ordering)
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(None, ordering) => Self::Enum(None, *ordering),
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => Self::Decimal(*precision, *scale),
            #[cfg(feature = "object")]
            Object(name, _) => Self::Object(name.to_string()),
            dt => panic!("{dt:?} not supported"),
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
            String => Self::String,
            Binary => Self::Binary,
            Date => Self::Date,
            Datetime(tu, tz) => Self::Datetime(tu, tz),
            Duration(tu) => Self::Duration(tu),
            Time => Self::Time,
            List(dt) => Self::List(Box::new((*dt).into())),
            #[cfg(feature = "dtype-array")]
            Array(dt, width) => Self::Array(Box::new((*dt).into()), width),
            Null => Self::Null,
            Unknown(kind) => Self::Unknown(kind),
            #[cfg(feature = "dtype-struct")]
            Struct(flds) => Self::Struct(flds),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, ordering) => Self::Categorical(None, ordering),
            #[cfg(feature = "dtype-categorical")]
            Enum(Some(categories), _) => create_enum_data_type(categories.0),
            #[cfg(feature = "dtype-categorical")]
            Enum(None, ordering) => Self::Enum(None, ordering),
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => Self::Decimal(precision, scale),
            #[cfg(feature = "object")]
            Object(_) => Self::Object("unknown", None),
        }
    }
}
