use arrow::array::IntoBoxedArray;
use polars_error::{PolarsError, PolarsResult, polars_bail};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::Scalar;
use crate::prelude::{AnyValue, DataType, Field};
use crate::series::Series;

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for Scalar {
    fn is_referenceable() -> bool {
        <SerializableScalar as schemars::JsonSchema>::is_referenceable()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        <SerializableScalar as schemars::JsonSchema>::schema_id()
    }

    fn schema_name() -> String {
        <SerializableScalar as schemars::JsonSchema>::schema_name()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        <SerializableScalar as schemars::JsonSchema>::json_schema(generator)
    }
}

#[cfg(feature = "serde")]
impl Serialize for Scalar {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerializableScalar::try_from(self.clone())
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for Scalar {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        SerializableScalar::deserialize(deserializer)
            .and_then(|v| Self::try_from(v).map_err(serde::de::Error::custom))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename = "AnyValue")]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum SerializableScalar {
    Null(DataType),
    /// An 8-bit integer number.
    Int8(i8),
    /// A 16-bit integer number.
    Int16(i16),
    /// A 32-bit integer number.
    Int32(i32),
    /// A 64-bit integer number.
    Int64(i64),
    /// A 128-bit integer number.
    Int128(i128),
    /// An unsigned 8-bit integer number.
    UInt8(u8),
    /// An unsigned 16-bit integer number.
    UInt16(u16),
    /// An unsigned 32-bit integer number.
    UInt32(u32),
    /// An unsigned 64-bit integer number.
    UInt64(u64),
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    /// Nested type, contains arrays that are filled with one of the datatypes.
    List(Series),
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    String(PlSmallStr),
    Binary(Vec<u8>),

    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    #[cfg(feature = "dtype-date")]
    Date(i32),

    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in nanoseconds (64 bits).
    #[cfg(feature = "dtype-datetime")]
    Datetime(
        i64,
        crate::prelude::TimeUnit,
        Option<crate::prelude::TimeZone>,
    ),

    /// A 64-bit integer representing difference between date-times in [`TimeUnit`]
    #[cfg(feature = "dtype-duration")]
    Duration(i64, crate::prelude::TimeUnit),

    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    #[cfg(feature = "dtype-time")]
    Time(i64),

    #[cfg(feature = "dtype-array")]
    Array(Series, usize),

    /// A 128-bit fixed point decimal number with a scale.
    #[cfg(feature = "dtype-decimal")]
    Decimal(i128, usize),

    #[cfg(feature = "dtype-categorical")]
    Categorical {
        value: PlSmallStr,
        name: PlSmallStr,
        namespace: PlSmallStr,
        physical: polars_dtype::categorical::CategoricalPhysical,
    },
    #[cfg(feature = "dtype-categorical")]
    Enum {
        value: polars_dtype::categorical::CatSize,
        categories: Series,
    },

    #[cfg(feature = "dtype-struct")]
    Struct(Vec<(PlSmallStr, SerializableScalar)>),
}

impl TryFrom<Scalar> for SerializableScalar {
    type Error = PolarsError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        let out = match value.value {
            AnyValue::Null => Self::Null(value.dtype),
            AnyValue::Int8(v) => Self::Int8(v),
            AnyValue::Int16(v) => Self::Int16(v),
            AnyValue::Int32(v) => Self::Int32(v),
            AnyValue::Int64(v) => Self::Int64(v),
            AnyValue::Int128(v) => Self::Int128(v),
            AnyValue::UInt8(v) => Self::UInt8(v),
            AnyValue::UInt16(v) => Self::UInt16(v),
            AnyValue::UInt32(v) => Self::UInt32(v),
            AnyValue::UInt64(v) => Self::UInt64(v),
            AnyValue::Float32(v) => Self::Float32(v),
            AnyValue::Float64(v) => Self::Float64(v),
            AnyValue::List(series) => Self::List(series),
            AnyValue::Boolean(v) => Self::Boolean(v),
            AnyValue::String(v) => Self::String(PlSmallStr::from(v)),
            AnyValue::StringOwned(v) => Self::String(v),
            AnyValue::Binary(v) => Self::Binary(v.to_vec()),
            AnyValue::BinaryOwned(v) => Self::Binary(v),

            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => Self::Date(v),

            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, tz) => Self::Datetime(v, tu, tz.cloned()),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::DatetimeOwned(v, time_unit, time_zone) => {
                Self::Datetime(v, time_unit, time_zone.as_deref().cloned())
            },

            #[cfg(feature = "dtype-duration")]
            AnyValue::Duration(v, time_unit) => Self::Duration(v, time_unit),

            #[cfg(feature = "dtype-time")]
            AnyValue::Time(v) => Self::Time(v),

            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(cat, _) | AnyValue::CategoricalOwned(cat, _) => {
                let DataType::Categorical(categories, mapping) = value.dtype() else {
                    unreachable!();
                };

                Self::Categorical {
                    value: PlSmallStr::from(mapping.cat_to_str(cat).unwrap()),
                    name: categories.name().clone(),
                    namespace: categories.namespace().clone(),
                    physical: categories.physical(),
                }
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Enum(idx, _) | AnyValue::EnumOwned(idx, _) => {
                let DataType::Enum(categories, _) = value.dtype() else {
                    unreachable!();
                };

                Self::Enum {
                    value: idx,
                    categories: Series::from_arrow(
                        PlSmallStr::EMPTY,
                        categories.categories().clone().into_boxed(),
                    )
                    .unwrap(),
                }
            },

            #[cfg(feature = "dtype-array")]
            AnyValue::Array(v, width) => Self::Array(v, width),

            #[cfg(feature = "object")]
            AnyValue::Object(..) | AnyValue::ObjectOwned(..) => {
                polars_bail!(nyi = "Cannot serialize object value.")
            },

            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(idx, arr, fields) => {
                assert!(idx < arr.len());
                assert_eq!(arr.values().len(), fields.len());

                Self::Struct(
                    arr.values()
                        .iter()
                        .zip(fields.iter())
                        .map(|(arr, field)| {
                            let series = unsafe {
                                Series::from_chunks_and_dtype_unchecked(
                                    PlSmallStr::EMPTY,
                                    vec![arr.clone()],
                                    field.dtype(),
                                )
                            };
                            let av = unsafe { series.get_unchecked(idx) };
                            PolarsResult::Ok((
                                field.name().clone(),
                                Self::try_from(Scalar::new(field.dtype.clone(), av.into_static()))?,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            },

            #[cfg(feature = "dtype-struct")]
            AnyValue::StructOwned(v) => {
                let (avs, fields) = *v;
                assert_eq!(avs.len(), fields.len());

                Self::Struct(
                    avs.into_iter()
                        .zip(fields.into_iter())
                        .map(|(av, field)| {
                            PolarsResult::Ok((
                                field.name,
                                Self::try_from(Scalar::new(field.dtype, av.into_static()))?,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            },

            #[cfg(feature = "dtype-decimal")]
            AnyValue::Decimal(v, scale) => Self::Decimal(v, scale),
        };
        Ok(out)
    }
}

impl TryFrom<SerializableScalar> for Scalar {
    type Error = PolarsError;

    fn try_from(value: SerializableScalar) -> Result<Self, Self::Error> {
        type S = SerializableScalar;
        Ok(match value {
            S::Null(dtype) => Self::null(dtype),
            S::Int8(v) => Self::from(v),
            S::Int16(v) => Self::from(v),
            S::Int32(v) => Self::from(v),
            S::Int64(v) => Self::from(v),
            S::Int128(v) => Self::from(v),
            S::UInt8(v) => Self::from(v),
            S::UInt16(v) => Self::from(v),
            S::UInt32(v) => Self::from(v),
            S::UInt64(v) => Self::from(v),
            S::Float32(v) => Self::from(v),
            S::Float64(v) => Self::from(v),
            S::List(v) => Self::new_list(v),
            S::Boolean(v) => Self::from(v),
            S::String(v) => Self::from(v),
            S::Binary(v) => Self::from(v),
            #[cfg(feature = "dtype-date")]
            S::Date(v) => Self::new_date(v),
            #[cfg(feature = "dtype-datetime")]
            S::Datetime(v, time_unit, time_zone) => Self::new_datetime(v, time_unit, time_zone),
            #[cfg(feature = "dtype-duration")]
            S::Duration(v, time_unit) => Self::new_duration(v, time_unit),
            #[cfg(feature = "dtype-time")]
            S::Time(v) => Self::new_time(v),
            #[cfg(feature = "dtype-array")]
            S::Array(v, width) => Self::new_array(v, width),
            #[cfg(feature = "dtype-decimal")]
            S::Decimal(v, scale) => Self::new_decimal(v, scale),

            #[cfg(feature = "dtype-categorical")]
            S::Categorical {
                value,
                name,
                namespace,
                physical,
            } => Self::new_categorical(value.as_str(), name, namespace, physical)?,
            #[cfg(feature = "dtype-categorical")]
            S::Enum { value, categories } => {
                Self::new_enum(value, categories.str()?.rechunk().downcast_as_array())?
            },
            #[cfg(feature = "dtype-struct")]
            S::Struct(scs) => {
                let (avs, fields) = scs
                    .into_iter()
                    .map(|(name, scalar)| {
                        let Scalar { dtype, value } = Scalar::try_from(scalar)?;
                        Ok((value, Field::new(name, dtype)))
                    })
                    .collect::<PolarsResult<(Vec<AnyValue<'static>>, Vec<Field>)>>()?;

                let dtype = DataType::Struct(fields.clone());
                Self::new(dtype, AnyValue::StructOwned(Box::new((avs, fields))))
            },
        })
    }
}
