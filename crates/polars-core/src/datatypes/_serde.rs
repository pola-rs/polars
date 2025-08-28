//! Having `Object<&;static> in [`DataType`] make serde tag the `Deserialize` trait bound 'static
//! even though we skip serializing `Object`.
//!
//! We could use [serde_1712](https://github.com/serde-rs/serde/issues/1712), but that gave problems caused by
//! [rust_96956](https://github.com/rust-lang/rust/issues/96956), so we make a dummy type without static

use polars_dtype::categorical::CategoricalPhysical;
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

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for DataType {
    fn schema_name() -> String {
        SerializableDataType::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        SerializableDataType::schema_id()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        SerializableDataType::json_schema(generator)
    }
}

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[serde(rename = "DataType")]
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
    Int128,
    Float32,
    Float64,
    String,
    Binary,
    BinaryOffset,
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
    Categorical {
        name: String,
        namespace: String,
        physical: CategoricalPhysical,
    },
    #[cfg(feature = "dtype-categorical")]
    Enum {
        strings: Series,
    },
    #[cfg(feature = "dtype-decimal")]
    Decimal(Option<usize>, Option<usize>),
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
            Int128 => Self::Int128,
            Float32 => Self::Float32,
            Float64 => Self::Float64,
            String => Self::String,
            Binary => Self::Binary,
            BinaryOffset => Self::BinaryOffset,
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
            Categorical(cats, _) => Self::Categorical {
                name: cats.name().to_string(),
                namespace: cats.namespace().to_string(),
                physical: cats.physical(),
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(fcats, _) => Self::Enum {
                strings: StringChunked::with_chunk(
                    PlSmallStr::from_static("categories"),
                    fcats.categories().clone(),
                )
                .into_series(),
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => Self::Decimal(*precision, *scale),
            #[cfg(feature = "object")]
            Object(name) => Self::Object(name.to_string()),
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
            Int128 => Self::Int128,
            Float32 => Self::Float32,
            Float64 => Self::Float64,
            String => Self::String,
            Binary => Self::Binary,
            BinaryOffset => Self::BinaryOffset,
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
            Categorical {
                name,
                namespace,
                physical,
            } => {
                let cats = Categories::new(
                    PlSmallStr::from(name),
                    PlSmallStr::from(namespace),
                    physical,
                );
                let mapping = cats.mapping();
                Self::Categorical(cats, mapping)
            },
            #[cfg(feature = "dtype-categorical")]
            Enum { strings } => {
                let ca = strings.str().unwrap();
                let fcats = FrozenCategories::new(ca.iter().flatten()).unwrap();
                let mapping = fcats.mapping().clone();
                Self::Enum(fcats, mapping)
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => Self::Decimal(precision, scale),
            #[cfg(feature = "object")]
            Object(_) => Self::Object("unknown"),
        }
    }
}
