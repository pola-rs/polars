use std::borrow::Cow;

use arrow::types::PrimitiveType;
use polars_compute::cast::SerPrimitive;
use polars_error::feature_gated;
#[cfg(feature = "dtype-categorical")]
use polars_utils::sync::SyncPtr;
use polars_utils::total_ord::ToTotalOrd;

use super::*;
#[cfg(feature = "dtype-struct")]
use crate::prelude::any_value::arr_to_any_value;

#[cfg(feature = "object")]
#[derive(Debug)]
pub struct OwnedObject(pub Box<dyn PolarsObjectSafe>);

#[cfg(feature = "object")]
impl Clone for OwnedObject {
    fn clone(&self) -> Self {
        Self(self.0.to_boxed())
    }
}

#[derive(Debug, Clone, Default)]
pub enum AnyValue<'a> {
    #[default]
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    String(&'a str),
    /// An unsigned 8-bit integer number.
    UInt8(u8),
    /// An unsigned 16-bit integer number.
    UInt16(u16),
    /// An unsigned 32-bit integer number.
    UInt32(u32),
    /// An unsigned 64-bit integer number.
    UInt64(u64),
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
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    #[cfg(feature = "dtype-date")]
    Date(i32),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in nanoseconds (64 bits).
    #[cfg(feature = "dtype-datetime")]
    Datetime(i64, TimeUnit, Option<&'a TimeZone>),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in nanoseconds (64 bits).
    #[cfg(feature = "dtype-datetime")]
    DatetimeOwned(i64, TimeUnit, Option<Arc<TimeZone>>),
    /// A 64-bit integer representing difference between date-times in [`TimeUnit`]
    #[cfg(feature = "dtype-duration")]
    Duration(i64, TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    #[cfg(feature = "dtype-time")]
    Time(i64),
    // If syncptr is_null the data is in the rev-map
    // otherwise it is in the array pointer
    #[cfg(feature = "dtype-categorical")]
    Categorical(u32, &'a RevMapping, SyncPtr<Utf8ViewArray>),
    // If syncptr is_null the data is in the rev-map
    // otherwise it is in the array pointer
    #[cfg(feature = "dtype-categorical")]
    CategoricalOwned(u32, Arc<RevMapping>, SyncPtr<Utf8ViewArray>),
    #[cfg(feature = "dtype-categorical")]
    Enum(u32, &'a RevMapping, SyncPtr<Utf8ViewArray>),
    #[cfg(feature = "dtype-categorical")]
    EnumOwned(u32, Arc<RevMapping>, SyncPtr<Utf8ViewArray>),
    /// Nested type, contains arrays that are filled with one of the datatypes.
    List(Series),
    #[cfg(feature = "dtype-array")]
    Array(Series, usize),
    /// Can be used to fmt and implements Any, so can be downcasted to the proper value type.
    #[cfg(feature = "object")]
    Object(&'a dyn PolarsObjectSafe),
    #[cfg(feature = "object")]
    ObjectOwned(OwnedObject),
    // 3 pointers and thus not larger than string/vec
    // - idx in the `&StructArray`
    // - The array itself
    // - The fields
    #[cfg(feature = "dtype-struct")]
    Struct(usize, &'a StructArray, &'a [Field]),
    #[cfg(feature = "dtype-struct")]
    StructOwned(Box<(Vec<AnyValue<'a>>, Vec<Field>)>),
    /// An UTF8 encoded string type.
    StringOwned(PlSmallStr),
    Binary(&'a [u8]),
    BinaryOwned(Vec<u8>),
    /// A 128-bit fixed point decimal number with a scale.
    #[cfg(feature = "dtype-decimal")]
    Decimal(i128, usize),
}

#[cfg(feature = "serde")]
impl Serialize for AnyValue<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let name = "AnyValue";
        match self {
            AnyValue::Null => serializer.serialize_unit_variant(name, 0, "Null"),
            AnyValue::Int8(v) => serializer.serialize_newtype_variant(name, 1, "Int8", v),
            AnyValue::Int16(v) => serializer.serialize_newtype_variant(name, 2, "Int16", v),
            AnyValue::Int32(v) => serializer.serialize_newtype_variant(name, 3, "Int32", v),
            AnyValue::Int64(v) => serializer.serialize_newtype_variant(name, 4, "Int64", v),
            AnyValue::Int128(v) => serializer.serialize_newtype_variant(name, 4, "Int128", v),
            AnyValue::UInt8(v) => serializer.serialize_newtype_variant(name, 5, "UInt8", v),
            AnyValue::UInt16(v) => serializer.serialize_newtype_variant(name, 6, "UInt16", v),
            AnyValue::UInt32(v) => serializer.serialize_newtype_variant(name, 7, "UInt32", v),
            AnyValue::UInt64(v) => serializer.serialize_newtype_variant(name, 8, "UInt64", v),
            AnyValue::Float32(v) => serializer.serialize_newtype_variant(name, 9, "Float32", v),
            AnyValue::Float64(v) => serializer.serialize_newtype_variant(name, 10, "Float64", v),
            AnyValue::List(v) => serializer.serialize_newtype_variant(name, 11, "List", v),
            AnyValue::Boolean(v) => serializer.serialize_newtype_variant(name, 12, "Bool", v),
            // both string variants same number
            AnyValue::String(v) => serializer.serialize_newtype_variant(name, 13, "StringOwned", v),
            AnyValue::StringOwned(v) => {
                serializer.serialize_newtype_variant(name, 13, "StringOwned", v.as_str())
            },
            AnyValue::Binary(v) => serializer.serialize_newtype_variant(name, 14, "BinaryOwned", v),
            AnyValue::BinaryOwned(v) => {
                serializer.serialize_newtype_variant(name, 14, "BinaryOwned", v)
            },
            _ => Err(serde::ser::Error::custom(
                "Unknown data type. Cannot serialize",
            )),
        }
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for AnyValue<'static> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        #[repr(u8)]
        enum AvField {
            Null,
            Int8,
            Int16,
            Int32,
            Int64,
            Int128,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
            List,
            Bool,
            StringOwned,
            BinaryOwned,
        }
        const VARIANTS: &[&str] = &[
            "Null",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Int128",
            "Float32",
            "Float64",
            "List",
            "Boolean",
            "StringOwned",
            "BinaryOwned",
        ];
        const LAST: u8 = unsafe { std::mem::transmute::<_, u8>(AvField::BinaryOwned) };

        struct FieldVisitor;

        impl Visitor<'_> for FieldVisitor {
            type Value = AvField;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(formatter, "an integer between 0-{LAST}")
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: Error,
            {
                let field: u8 = NumCast::from(v).ok_or_else(|| {
                    serde::de::Error::invalid_value(
                        Unexpected::Signed(v),
                        &"expected value that fits into u8",
                    )
                })?;

                // SAFETY:
                // we are repr: u8 and check last value that we are in bounds
                let field = unsafe {
                    if field <= LAST {
                        std::mem::transmute::<u8, AvField>(field)
                    } else {
                        return Err(serde::de::Error::invalid_value(
                            Unexpected::Signed(v),
                            &"expected value that fits into AnyValue's number of fields",
                        ));
                    }
                };
                Ok(field)
            }

            fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
            where
                E: Error,
            {
                self.visit_bytes(v.as_bytes())
            }

            fn visit_bytes<E>(self, v: &[u8]) -> std::result::Result<Self::Value, E>
            where
                E: Error,
            {
                let field = match v {
                    b"Null" => AvField::Null,
                    b"Int8" => AvField::Int8,
                    b"Int16" => AvField::Int16,
                    b"Int32" => AvField::Int32,
                    b"Int64" => AvField::Int64,
                    b"Int128" => AvField::Int128,
                    b"UInt8" => AvField::UInt8,
                    b"UInt16" => AvField::UInt16,
                    b"UInt32" => AvField::UInt32,
                    b"UInt64" => AvField::UInt64,
                    b"Float32" => AvField::Float32,
                    b"Float64" => AvField::Float64,
                    b"List" => AvField::List,
                    b"Bool" => AvField::Bool,
                    b"StringOwned" | b"String" => AvField::StringOwned,
                    b"BinaryOwned" | b"Binary" => AvField::BinaryOwned,
                    _ => {
                        return Err(serde::de::Error::unknown_variant(
                            &String::from_utf8_lossy(v),
                            VARIANTS,
                        ))
                    },
                };
                Ok(field)
            }
        }

        impl<'a> Deserialize<'a> for AvField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'a>,
            {
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct OuterVisitor;

        impl<'b> Visitor<'b> for OuterVisitor {
            type Value = AnyValue<'static>;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(formatter, "enum AnyValue")
            }

            fn visit_enum<A>(self, data: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: EnumAccess<'b>,
            {
                let out = match data.variant()? {
                    (AvField::Null, _variant) => AnyValue::Null,
                    (AvField::Int8, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int8(value)
                    },
                    (AvField::Int16, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int16(value)
                    },
                    (AvField::Int32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int32(value)
                    },
                    (AvField::Int64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int64(value)
                    },
                    (AvField::Int128, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int128(value)
                    },
                    (AvField::UInt8, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt8(value)
                    },
                    (AvField::UInt16, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt16(value)
                    },
                    (AvField::UInt32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt32(value)
                    },
                    (AvField::UInt64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt64(value)
                    },
                    (AvField::Float32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Float32(value)
                    },
                    (AvField::Float64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Float64(value)
                    },
                    (AvField::Bool, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Boolean(value)
                    },
                    (AvField::List, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::List(value)
                    },
                    (AvField::StringOwned, variant) => {
                        let value: PlSmallStr = variant.newtype_variant()?;
                        AnyValue::StringOwned(value)
                    },
                    (AvField::BinaryOwned, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::BinaryOwned(value)
                    },
                };
                Ok(out)
            }
        }
        deserializer.deserialize_enum("AnyValue", VARIANTS, OuterVisitor)
    }
}

impl AnyValue<'static> {
    pub fn zero_sum(dtype: &DataType) -> Self {
        match dtype {
            DataType::String => AnyValue::StringOwned(PlSmallStr::EMPTY),
            DataType::Binary => AnyValue::BinaryOwned(Vec::new()),
            DataType::Boolean => (0 as IdxSize).into(),
            // SAFETY: numeric values are static, inform the compiler of this.
            d if d.is_primitive_numeric() => unsafe {
                std::mem::transmute::<AnyValue<'_>, AnyValue<'static>>(
                    AnyValue::UInt8(0).cast(dtype),
                )
            },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(unit) => AnyValue::Duration(0, *unit),
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_p, s) => {
                AnyValue::Decimal(0, s.expect("unknown scale during execution"))
            },
            _ => AnyValue::Null,
        }
    }

    /// Can the [`AnyValue`] exist as having `dtype` as its `DataType`.
    pub fn can_have_dtype(&self, dtype: &DataType) -> bool {
        matches!(self, AnyValue::Null) || dtype == &self.dtype()
    }
}

impl<'a> AnyValue<'a> {
    /// Get the matching [`DataType`] for this [`AnyValue`]`.
    ///
    /// Note: For `Categorical` and `Enum` values, the exact mapping information
    /// is not preserved in the result for performance reasons.
    pub fn dtype(&self) -> DataType {
        use AnyValue::*;
        match self {
            Null => DataType::Null,
            Boolean(_) => DataType::Boolean,
            Int8(_) => DataType::Int8,
            Int16(_) => DataType::Int16,
            Int32(_) => DataType::Int32,
            Int64(_) => DataType::Int64,
            Int128(_) => DataType::Int128,
            UInt8(_) => DataType::UInt8,
            UInt16(_) => DataType::UInt16,
            UInt32(_) => DataType::UInt32,
            UInt64(_) => DataType::UInt64,
            Float32(_) => DataType::Float32,
            Float64(_) => DataType::Float64,
            String(_) | StringOwned(_) => DataType::String,
            Binary(_) | BinaryOwned(_) => DataType::Binary,
            #[cfg(feature = "dtype-date")]
            Date(_) => DataType::Date,
            #[cfg(feature = "dtype-time")]
            Time(_) => DataType::Time,
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, tu, tz) => DataType::Datetime(*tu, (*tz).cloned()),
            #[cfg(feature = "dtype-datetime")]
            DatetimeOwned(_, tu, tz) => {
                DataType::Datetime(*tu, tz.as_ref().map(|v| v.as_ref().clone()))
            },
            #[cfg(feature = "dtype-duration")]
            Duration(_, tu) => DataType::Duration(*tu),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _, _) | CategoricalOwned(_, _, _) => {
                DataType::Categorical(None, Default::default())
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _, _) | EnumOwned(_, _, _) => DataType::Enum(None, Default::default()),
            List(s) => DataType::List(Box::new(s.dtype().clone())),
            #[cfg(feature = "dtype-array")]
            Array(s, size) => DataType::Array(Box::new(s.dtype().clone()), *size),
            #[cfg(feature = "dtype-struct")]
            Struct(_, _, fields) => DataType::Struct(fields.to_vec()),
            #[cfg(feature = "dtype-struct")]
            StructOwned(payload) => DataType::Struct(payload.1.clone()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, scale) => DataType::Decimal(None, Some(*scale)),
            #[cfg(feature = "object")]
            Object(o) => DataType::Object(o.type_name(), None),
            #[cfg(feature = "object")]
            ObjectOwned(o) => DataType::Object(o.0.type_name(), None),
        }
    }

    /// Extract a numerical value from the AnyValue
    #[doc(hidden)]
    #[inline]
    pub fn extract<T: NumCast>(&self) -> Option<T> {
        use AnyValue::*;
        match self {
            Int8(v) => NumCast::from(*v),
            Int16(v) => NumCast::from(*v),
            Int32(v) => NumCast::from(*v),
            Int64(v) => NumCast::from(*v),
            Int128(v) => NumCast::from(*v),
            UInt8(v) => NumCast::from(*v),
            UInt16(v) => NumCast::from(*v),
            UInt32(v) => NumCast::from(*v),
            UInt64(v) => NumCast::from(*v),
            Float32(v) => NumCast::from(*v),
            Float64(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-date")]
            Date(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, _, _) | DatetimeOwned(v, _, _) => NumCast::from(*v),
            #[cfg(feature = "dtype-time")]
            Time(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, _) => NumCast::from(*v),
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, scale) => {
                if *scale == 0 {
                    NumCast::from(*v)
                } else {
                    let f: Option<f64> = NumCast::from(*v);
                    NumCast::from(f? / 10f64.powi(*scale as _))
                }
            },
            Boolean(v) => NumCast::from(if *v { 1 } else { 0 }),
            String(v) => {
                if let Ok(val) = (*v).parse::<i128>() {
                    NumCast::from(val)
                } else {
                    NumCast::from((*v).parse::<f64>().ok()?)
                }
            },
            StringOwned(v) => String(v.as_str()).extract(),
            _ => None,
        }
    }

    #[inline]
    pub fn try_extract<T: NumCast>(&self) -> PolarsResult<T> {
        self.extract().ok_or_else(|| {
            polars_err!(
                ComputeError: "could not extract number from any-value of dtype: '{:?}'",
                self.dtype(),
            )
        })
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, AnyValue::Boolean(_))
    }

    pub fn is_primitive_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub fn is_float(&self) -> bool {
        matches!(self, AnyValue::Float32(_) | AnyValue::Float64(_))
    }

    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    pub fn is_signed_integer(&self) -> bool {
        matches!(
            self,
            AnyValue::Int8(_)
                | AnyValue::Int16(_)
                | AnyValue::Int32(_)
                | AnyValue::Int64(_)
                | AnyValue::Int128(_)
        )
    }

    pub fn is_unsigned_integer(&self) -> bool {
        matches!(
            self,
            AnyValue::UInt8(_) | AnyValue::UInt16(_) | AnyValue::UInt32(_) | AnyValue::UInt64(_)
        )
    }

    pub fn is_nan(&self) -> bool {
        match self {
            AnyValue::Float32(f) => f.is_nan(),
            AnyValue::Float64(f) => f.is_nan(),
            _ => false,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, AnyValue::Null)
    }

    pub fn is_nested_null(&self) -> bool {
        match self {
            AnyValue::Null => true,
            AnyValue::List(s) => s.null_count() == s.len(),
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(s, _) => s.null_count() == s.len(),
            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(_, _, _) => self._iter_struct_av().all(|av| av.is_nested_null()),
            _ => false,
        }
    }

    /// Cast `AnyValue` to the provided data type and return a new `AnyValue` with type `dtype`,
    /// if possible.
    pub fn strict_cast(&self, dtype: &'a DataType) -> Option<AnyValue<'a>> {
        let new_av = match (self, dtype) {
            // to numeric
            (av, DataType::UInt8) => AnyValue::UInt8(av.extract::<u8>()?),
            (av, DataType::UInt16) => AnyValue::UInt16(av.extract::<u16>()?),
            (av, DataType::UInt32) => AnyValue::UInt32(av.extract::<u32>()?),
            (av, DataType::UInt64) => AnyValue::UInt64(av.extract::<u64>()?),
            (av, DataType::Int8) => AnyValue::Int8(av.extract::<i8>()?),
            (av, DataType::Int16) => AnyValue::Int16(av.extract::<i16>()?),
            (av, DataType::Int32) => AnyValue::Int32(av.extract::<i32>()?),
            (av, DataType::Int64) => AnyValue::Int64(av.extract::<i64>()?),
            (av, DataType::Int128) => AnyValue::Int128(av.extract::<i128>()?),
            (av, DataType::Float32) => AnyValue::Float32(av.extract::<f32>()?),
            (av, DataType::Float64) => AnyValue::Float64(av.extract::<f64>()?),

            // to boolean
            (AnyValue::UInt8(v), DataType::Boolean) => AnyValue::Boolean(*v != u8::default()),
            (AnyValue::UInt16(v), DataType::Boolean) => AnyValue::Boolean(*v != u16::default()),
            (AnyValue::UInt32(v), DataType::Boolean) => AnyValue::Boolean(*v != u32::default()),
            (AnyValue::UInt64(v), DataType::Boolean) => AnyValue::Boolean(*v != u64::default()),
            (AnyValue::Int8(v), DataType::Boolean) => AnyValue::Boolean(*v != i8::default()),
            (AnyValue::Int16(v), DataType::Boolean) => AnyValue::Boolean(*v != i16::default()),
            (AnyValue::Int32(v), DataType::Boolean) => AnyValue::Boolean(*v != i32::default()),
            (AnyValue::Int64(v), DataType::Boolean) => AnyValue::Boolean(*v != i64::default()),
            (AnyValue::Int128(v), DataType::Boolean) => AnyValue::Boolean(*v != i128::default()),
            (AnyValue::Float32(v), DataType::Boolean) => AnyValue::Boolean(*v != f32::default()),
            (AnyValue::Float64(v), DataType::Boolean) => AnyValue::Boolean(*v != f64::default()),

            // to string
            (AnyValue::String(v), DataType::String) => AnyValue::String(v),
            (AnyValue::StringOwned(v), DataType::String) => AnyValue::StringOwned(v.clone()),

            (av, DataType::String) => {
                let mut tmp = vec![];
                if av.is_unsigned_integer() {
                    let val = av.extract::<u64>()?;
                    SerPrimitive::write(&mut tmp, val);
                } else if av.is_float() {
                    let val = av.extract::<f64>()?;
                    SerPrimitive::write(&mut tmp, val);
                } else {
                    let val = av.extract::<i64>()?;
                    SerPrimitive::write(&mut tmp, val);
                }
                AnyValue::StringOwned(PlSmallStr::from_str(std::str::from_utf8(&tmp).unwrap()))
            },

            // to binary
            (AnyValue::String(v), DataType::Binary) => AnyValue::Binary(v.as_bytes()),

            // to datetime
            #[cfg(feature = "dtype-datetime")]
            (av, DataType::Datetime(tu, tz)) if av.is_primitive_numeric() => {
                AnyValue::Datetime(av.extract::<i64>()?, *tu, tz.as_ref())
            },
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (AnyValue::Date(v), DataType::Datetime(tu, _)) => AnyValue::Datetime(
                match tu {
                    TimeUnit::Nanoseconds => (*v as i64) * NS_IN_DAY,
                    TimeUnit::Microseconds => (*v as i64) * US_IN_DAY,
                    TimeUnit::Milliseconds => (*v as i64) * MS_IN_DAY,
                },
                *tu,
                None,
            ),
            #[cfg(feature = "dtype-datetime")]
            (
                AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _),
                DataType::Datetime(tu_r, tz_r),
            ) => AnyValue::Datetime(
                match (tu, tu_r) {
                    (TimeUnit::Nanoseconds, TimeUnit::Microseconds) => *v / 1_000i64,
                    (TimeUnit::Nanoseconds, TimeUnit::Milliseconds) => *v / 1_000_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Nanoseconds) => *v * 1_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => *v / 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Microseconds) => *v * 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Nanoseconds) => *v * 1_000_000i64,
                    _ => *v,
                },
                *tu_r,
                tz_r.as_ref(),
            ),

            // to date
            #[cfg(feature = "dtype-date")]
            (av, DataType::Date) if av.is_primitive_numeric() => {
                AnyValue::Date(av.extract::<i32>()?)
            },
            #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
            (AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _), DataType::Date) => {
                AnyValue::Date(match tu {
                    TimeUnit::Nanoseconds => *v / NS_IN_DAY,
                    TimeUnit::Microseconds => *v / US_IN_DAY,
                    TimeUnit::Milliseconds => *v / MS_IN_DAY,
                } as i32)
            },

            // to time
            #[cfg(feature = "dtype-time")]
            (av, DataType::Time) if av.is_primitive_numeric() => {
                AnyValue::Time(av.extract::<i64>()?)
            },
            #[cfg(all(feature = "dtype-time", feature = "dtype-datetime"))]
            (AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _), DataType::Time) => {
                AnyValue::Time(match tu {
                    TimeUnit::Nanoseconds => *v % NS_IN_DAY,
                    TimeUnit::Microseconds => (*v % US_IN_DAY) * 1_000i64,
                    TimeUnit::Milliseconds => (*v % MS_IN_DAY) * 1_000_000i64,
                })
            },

            // to duration
            #[cfg(feature = "dtype-duration")]
            (av, DataType::Duration(tu)) if av.is_primitive_numeric() => {
                AnyValue::Duration(av.extract::<i64>()?, *tu)
            },
            #[cfg(all(feature = "dtype-duration", feature = "dtype-time"))]
            (AnyValue::Time(v), DataType::Duration(tu)) => AnyValue::Duration(
                match *tu {
                    TimeUnit::Nanoseconds => *v,
                    TimeUnit::Microseconds => *v / 1_000i64,
                    TimeUnit::Milliseconds => *v / 1_000_000i64,
                },
                *tu,
            ),
            #[cfg(feature = "dtype-duration")]
            (AnyValue::Duration(v, tu), DataType::Duration(tu_r)) => AnyValue::Duration(
                match (tu, tu_r) {
                    (_, _) if tu == tu_r => *v,
                    (TimeUnit::Nanoseconds, TimeUnit::Microseconds) => *v / 1_000i64,
                    (TimeUnit::Nanoseconds, TimeUnit::Milliseconds) => *v / 1_000_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Nanoseconds) => *v * 1_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => *v / 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Microseconds) => *v * 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Nanoseconds) => *v * 1_000_000i64,
                    _ => *v,
                },
                *tu_r,
            ),

            // to decimal
            #[cfg(feature = "dtype-decimal")]
            (av, DataType::Decimal(prec, scale)) if av.is_integer() => {
                let value = av.try_extract::<i128>().unwrap();
                let scale = scale.unwrap_or(0);
                let factor = 10_i128.pow(scale as _); // Conversion to u32 is safe, max value is 38.
                let converted = value.checked_mul(factor)?;

                // Check if the converted value fits into the specified precision
                let prec = prec.unwrap_or(38) as u32;
                let num_digits = (converted.abs() as f64).log10().ceil() as u32;
                if num_digits > prec {
                    return None;
                }

                AnyValue::Decimal(converted, scale)
            },
            #[cfg(feature = "dtype-decimal")]
            (AnyValue::Decimal(value, scale_av), DataType::Decimal(_, scale)) => {
                let Some(scale) = scale else {
                    return Some(self.clone());
                };
                // TODO: Allow lossy conversion?
                let scale_diff = scale.checked_sub(*scale_av)?;
                let factor = 10_i128.pow(scale_diff as _); // Conversion is safe, max value is 38.
                let converted = value.checked_mul(factor)?;
                AnyValue::Decimal(converted, *scale)
            },

            // to self
            (av, dtype) if av.dtype() == *dtype => self.clone(),

            _ => return None,
        };
        Some(new_av)
    }

    /// Cast `AnyValue` to the provided data type and return a new `AnyValue` with type `dtype`,
    /// if possible.
    pub fn try_strict_cast(&self, dtype: &'a DataType) -> PolarsResult<AnyValue<'a>> {
        self.strict_cast(dtype).ok_or_else(
            || polars_err!(ComputeError: "cannot cast any-value {:?} to dtype '{}'", self, dtype),
        )
    }

    pub fn cast(&self, dtype: &'a DataType) -> AnyValue<'a> {
        match self.strict_cast(dtype) {
            Some(av) => av,
            None => AnyValue::Null,
        }
    }

    pub fn idx(&self) -> IdxSize {
        match self {
            #[cfg(not(feature = "bigidx"))]
            Self::UInt32(v) => *v,
            #[cfg(feature = "bigidx")]
            Self::UInt64(v) => *v,
            _ => panic!("expected index type found {self:?}"),
        }
    }

    pub fn str_value(&self) -> Cow<'a, str> {
        match self {
            Self::String(s) => Cow::Borrowed(s),
            Self::StringOwned(s) => Cow::Owned(s.to_string()),
            Self::Null => Cow::Borrowed("null"),
            #[cfg(feature = "dtype-categorical")]
            Self::Categorical(idx, rev, arr) | AnyValue::Enum(idx, rev, arr) => {
                if arr.is_null() {
                    Cow::Borrowed(rev.get(*idx))
                } else {
                    unsafe { Cow::Borrowed(arr.deref_unchecked().value(*idx as usize)) }
                }
            },
            #[cfg(feature = "dtype-categorical")]
            Self::CategoricalOwned(idx, rev, arr) | AnyValue::EnumOwned(idx, rev, arr) => {
                if arr.is_null() {
                    Cow::Owned(rev.get(*idx).to_string())
                } else {
                    unsafe { Cow::Borrowed(arr.deref_unchecked().value(*idx as usize)) }
                }
            },
            av => Cow::Owned(av.to_string()),
        }
    }
}

impl From<AnyValue<'_>> for DataType {
    fn from(value: AnyValue<'_>) -> Self {
        value.dtype()
    }
}

impl<'a> From<&AnyValue<'a>> for DataType {
    fn from(value: &AnyValue<'a>) -> Self {
        value.dtype()
    }
}

impl AnyValue<'_> {
    pub fn hash_impl<H: Hasher>(&self, state: &mut H, cheap: bool) {
        use AnyValue::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Int8(v) => v.hash(state),
            Int16(v) => v.hash(state),
            Int32(v) => v.hash(state),
            Int64(v) => v.hash(state),
            Int128(v) => feature_gated!("dtype-i128", v.hash(state)),
            UInt8(v) => v.hash(state),
            UInt16(v) => v.hash(state),
            UInt32(v) => v.hash(state),
            UInt64(v) => v.hash(state),
            String(v) => v.hash(state),
            StringOwned(v) => v.hash(state),
            Float32(v) => v.to_ne_bytes().hash(state),
            Float64(v) => v.to_ne_bytes().hash(state),
            Binary(v) => v.hash(state),
            BinaryOwned(v) => v.hash(state),
            Boolean(v) => v.hash(state),
            List(v) => {
                if !cheap {
                    Hash::hash(&Wrap(v.clone()), state)
                }
            },
            #[cfg(feature = "dtype-array")]
            Array(v, width) => {
                if !cheap {
                    Hash::hash(&Wrap(v.clone()), state)
                }
                width.hash(state)
            },
            #[cfg(feature = "dtype-date")]
            Date(v) => v.hash(state),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, tu, tz) => {
                v.hash(state);
                tu.hash(state);
                tz.hash(state);
            },
            #[cfg(feature = "dtype-datetime")]
            DatetimeOwned(v, tu, tz) => {
                v.hash(state);
                tu.hash(state);
                tz.hash(state);
            },
            #[cfg(feature = "dtype-duration")]
            Duration(v, tz) => {
                v.hash(state);
                tz.hash(state);
            },
            #[cfg(feature = "dtype-time")]
            Time(v) => v.hash(state),
            #[cfg(feature = "dtype-categorical")]
            Categorical(v, _, _)
            | CategoricalOwned(v, _, _)
            | Enum(v, _, _)
            | EnumOwned(v, _, _) => v.hash(state),
            #[cfg(feature = "object")]
            Object(_) => {},
            #[cfg(feature = "object")]
            ObjectOwned(_) => {},
            #[cfg(feature = "dtype-struct")]
            Struct(_, _, _) => {
                if !cheap {
                    let mut buf = vec![];
                    self._materialize_struct_av(&mut buf);
                    buf.hash(state)
                }
            },
            #[cfg(feature = "dtype-struct")]
            StructOwned(v) => v.0.hash(state),
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, k) => {
                v.hash(state);
                k.hash(state);
            },
            Null => {},
        }
    }
}

impl Hash for AnyValue<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_impl(state, false)
    }
}

impl Eq for AnyValue<'_> {}

impl<'a, T> From<Option<T>> for AnyValue<'a>
where
    T: Into<AnyValue<'a>>,
{
    #[inline]
    fn from(a: Option<T>) -> Self {
        match a {
            None => AnyValue::Null,
            Some(v) => v.into(),
        }
    }
}

impl<'a> AnyValue<'a> {
    #[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
    pub(crate) fn as_date(&self) -> AnyValue<'static> {
        match self {
            #[cfg(feature = "dtype-date")]
            AnyValue::Int32(v) => AnyValue::Date(*v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }
    #[cfg(feature = "dtype-datetime")]
    pub(crate) fn as_datetime(&self, tu: TimeUnit, tz: Option<&'a TimeZone>) -> AnyValue<'a> {
        match self {
            AnyValue::Int64(v) => AnyValue::Datetime(*v, tu, tz),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[cfg(feature = "dtype-duration")]
    pub(crate) fn as_duration(&self, tu: TimeUnit) -> AnyValue<'static> {
        match self {
            AnyValue::Int64(v) => AnyValue::Duration(*v, tu),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[cfg(feature = "dtype-time")]
    pub(crate) fn as_time(&self) -> AnyValue<'static> {
        match self {
            AnyValue::Int64(v) => AnyValue::Time(*v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    pub(crate) fn to_i128(&self) -> Option<i128> {
        match self {
            AnyValue::UInt8(v) => Some((*v).into()),
            AnyValue::UInt16(v) => Some((*v).into()),
            AnyValue::UInt32(v) => Some((*v).into()),
            AnyValue::UInt64(v) => Some((*v).into()),
            AnyValue::Int8(v) => Some((*v).into()),
            AnyValue::Int16(v) => Some((*v).into()),
            AnyValue::Int32(v) => Some((*v).into()),
            AnyValue::Int64(v) => Some((*v).into()),
            AnyValue::Int128(v) => Some(*v),
            _ => None,
        }
    }

    pub(crate) fn to_f64(&self) -> Option<f64> {
        match self {
            AnyValue::Float32(v) => Some((*v).into()),
            AnyValue::Float64(v) => Some(*v),
            _ => None,
        }
    }

    #[must_use]
    pub fn add(&self, rhs: &AnyValue) -> AnyValue<'static> {
        use AnyValue::*;
        match (self, rhs) {
            (Null, r) => r.clone().into_static(),
            (l, Null) => l.clone().into_static(),
            (Int32(l), Int32(r)) => Int32(l + r),
            (Int64(l), Int64(r)) => Int64(l + r),
            (UInt32(l), UInt32(r)) => UInt32(l + r),
            (UInt64(l), UInt64(r)) => UInt64(l + r),
            (Float32(l), Float32(r)) => Float32(l + r),
            (Float64(l), Float64(r)) => Float64(l + r),
            #[cfg(feature = "dtype-duration")]
            (Duration(l, lu), Duration(r, ru)) => {
                if lu != ru {
                    unimplemented!("adding durations with different units is not supported here");
                }

                Duration(l + r, *lu)
            },
            #[cfg(feature = "dtype-decimal")]
            (Decimal(l, ls), Decimal(r, rs)) => {
                if ls != rs {
                    unimplemented!("adding decimals with different scales is not supported here");
                }

                Decimal(l + r, *ls)
            },
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub fn as_borrowed(&self) -> AnyValue<'_> {
        match self {
            AnyValue::BinaryOwned(data) => AnyValue::Binary(data),
            AnyValue::StringOwned(data) => AnyValue::String(data.as_str()),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::DatetimeOwned(v, tu, tz) => {
                AnyValue::Datetime(*v, *tu, tz.as_ref().map(AsRef::as_ref))
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::CategoricalOwned(v, rev, arr) => {
                AnyValue::Categorical(*v, rev.as_ref(), *arr)
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::EnumOwned(v, rev, arr) => AnyValue::Enum(*v, rev.as_ref(), *arr),
            av => av.clone(),
        }
    }

    /// Try to coerce to an AnyValue with static lifetime.
    /// This can be done if it does not borrow any values.
    #[inline]
    pub fn into_static(self) -> AnyValue<'static> {
        use AnyValue::*;
        match self {
            Null => Null,
            Int8(v) => Int8(v),
            Int16(v) => Int16(v),
            Int32(v) => Int32(v),
            Int64(v) => Int64(v),
            Int128(v) => Int128(v),
            UInt8(v) => UInt8(v),
            UInt16(v) => UInt16(v),
            UInt32(v) => UInt32(v),
            UInt64(v) => UInt64(v),
            Boolean(v) => Boolean(v),
            Float32(v) => Float32(v),
            Float64(v) => Float64(v),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, tu, tz) => DatetimeOwned(v, tu, tz.map(|v| Arc::new(v.clone()))),
            #[cfg(feature = "dtype-datetime")]
            DatetimeOwned(v, tu, tz) => DatetimeOwned(v, tu, tz),
            #[cfg(feature = "dtype-date")]
            Date(v) => Date(v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, tu) => Duration(v, tu),
            #[cfg(feature = "dtype-time")]
            Time(v) => Time(v),
            List(v) => List(v),
            #[cfg(feature = "dtype-array")]
            Array(s, size) => Array(s, size),
            String(v) => StringOwned(PlSmallStr::from_str(v)),
            StringOwned(v) => StringOwned(v),
            Binary(v) => BinaryOwned(v.to_vec()),
            BinaryOwned(v) => BinaryOwned(v),
            #[cfg(feature = "object")]
            Object(v) => ObjectOwned(OwnedObject(v.to_boxed())),
            #[cfg(feature = "dtype-struct")]
            Struct(idx, arr, fields) => {
                let avs = struct_to_avs_static(idx, arr, fields);
                StructOwned(Box::new((avs, fields.to_vec())))
            },
            #[cfg(feature = "dtype-struct")]
            StructOwned(payload) => {
                let av = StructOwned(payload);
                // SAFETY: owned is already static
                unsafe { std::mem::transmute::<AnyValue<'a>, AnyValue<'static>>(av) }
            },
            #[cfg(feature = "object")]
            ObjectOwned(payload) => {
                let av = ObjectOwned(payload);
                // SAFETY: owned is already static
                unsafe { std::mem::transmute::<AnyValue<'a>, AnyValue<'static>>(av) }
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(val, scale) => Decimal(val, scale),
            #[cfg(feature = "dtype-categorical")]
            Categorical(v, rev, arr) => CategoricalOwned(v, Arc::new(rev.clone()), arr),
            #[cfg(feature = "dtype-categorical")]
            CategoricalOwned(v, rev, arr) => CategoricalOwned(v, rev, arr),
            #[cfg(feature = "dtype-categorical")]
            Enum(v, rev, arr) => EnumOwned(v, Arc::new(rev.clone()), arr),
            #[cfg(feature = "dtype-categorical")]
            EnumOwned(v, rev, arr) => EnumOwned(v, rev, arr),
        }
    }

    /// Get a reference to the `&str` contained within [`AnyValue`].
    pub fn get_str(&self) -> Option<&str> {
        match self {
            AnyValue::String(s) => Some(s),
            AnyValue::StringOwned(s) => Some(s.as_str()),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(idx, rev, arr) | AnyValue::Enum(idx, rev, arr) => {
                let s = if arr.is_null() {
                    rev.get(*idx)
                } else {
                    unsafe { arr.deref_unchecked().value(*idx as usize) }
                };
                Some(s)
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::CategoricalOwned(idx, rev, arr) | AnyValue::EnumOwned(idx, rev, arr) => {
                let s = if arr.is_null() {
                    rev.get(*idx)
                } else {
                    unsafe { arr.deref_unchecked().value(*idx as usize) }
                };
                Some(s)
            },
            _ => None,
        }
    }
}

impl<'a> From<AnyValue<'a>> for Option<i64> {
    fn from(val: AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => None,
            Int32(v) => Some(v as i64),
            Int64(v) => Some(v),
            UInt32(v) => Some(v as i64),
            _ => todo!(),
        }
    }
}

impl AnyValue<'_> {
    #[inline]
    pub fn eq_missing(&self, other: &Self, null_equal: bool) -> bool {
        fn struct_owned_value_iter<'a>(
            v: &'a (Vec<AnyValue<'_>>, Vec<Field>),
        ) -> impl ExactSizeIterator<Item = AnyValue<'a>> {
            v.0.iter().map(|v| v.as_borrowed())
        }
        fn struct_value_iter(
            idx: usize,
            arr: &StructArray,
        ) -> impl ExactSizeIterator<Item = AnyValue<'_>> {
            assert!(idx < arr.len());

            arr.values().iter().map(move |field_arr| unsafe {
                // SAFETY: We asserted before that idx is smaller than the array length. Since it
                // is an invariant of StructArray that all fields have the same length this is fine
                // to do.
                field_arr.get_unchecked(idx)
            })
        }

        fn struct_eq_missing<'a>(
            l: impl ExactSizeIterator<Item = AnyValue<'a>>,
            r: impl ExactSizeIterator<Item = AnyValue<'a>>,
            null_equal: bool,
        ) -> bool {
            if l.len() != r.len() {
                return false;
            }

            l.zip(r).all(|(lv, rv)| lv.eq_missing(&rv, null_equal))
        }

        use AnyValue::*;
        match (self, other) {
            // Map to borrowed.
            (StringOwned(l), r) => AnyValue::String(l.as_str()) == *r,
            (BinaryOwned(l), r) => AnyValue::Binary(l.as_slice()) == *r,
            #[cfg(feature = "object")]
            (ObjectOwned(l), r) => AnyValue::Object(&*l.0) == *r,
            (l, StringOwned(r)) => *l == AnyValue::String(r.as_str()),
            (l, BinaryOwned(r)) => *l == AnyValue::Binary(r.as_slice()),
            #[cfg(feature = "object")]
            (l, ObjectOwned(r)) => *l == AnyValue::Object(&*r.0),
            #[cfg(feature = "dtype-datetime")]
            (DatetimeOwned(lv, ltu, ltz), r) => {
                Datetime(*lv, *ltu, ltz.as_ref().map(|v| v.as_ref())) == *r
            },
            #[cfg(feature = "dtype-datetime")]
            (l, DatetimeOwned(rv, rtu, rtz)) => {
                *l == Datetime(*rv, *rtu, rtz.as_ref().map(|v| v.as_ref()))
            },
            #[cfg(feature = "dtype-categorical")]
            (CategoricalOwned(lv, lrev, larr), r) => Categorical(*lv, lrev.as_ref(), *larr) == *r,
            #[cfg(feature = "dtype-categorical")]
            (l, CategoricalOwned(rv, rrev, rarr)) => *l == Categorical(*rv, rrev.as_ref(), *rarr),
            #[cfg(feature = "dtype-categorical")]
            (EnumOwned(lv, lrev, larr), r) => Enum(*lv, lrev.as_ref(), *larr) == *r,
            #[cfg(feature = "dtype-categorical")]
            (l, EnumOwned(rv, rrev, rarr)) => *l == Enum(*rv, rrev.as_ref(), *rarr),

            // Comparison with null.
            (Null, Null) => null_equal,
            (Null, _) => false,
            (_, Null) => false,

            // Equality between equal types.
            (Boolean(l), Boolean(r)) => *l == *r,
            (UInt8(l), UInt8(r)) => *l == *r,
            (UInt16(l), UInt16(r)) => *l == *r,
            (UInt32(l), UInt32(r)) => *l == *r,
            (UInt64(l), UInt64(r)) => *l == *r,
            (Int8(l), Int8(r)) => *l == *r,
            (Int16(l), Int16(r)) => *l == *r,
            (Int32(l), Int32(r)) => *l == *r,
            (Int64(l), Int64(r)) => *l == *r,
            (Int128(l), Int128(r)) => *l == *r,
            (Float32(l), Float32(r)) => l.to_total_ord() == r.to_total_ord(),
            (Float64(l), Float64(r)) => l.to_total_ord() == r.to_total_ord(),
            (String(l), String(r)) => l == r,
            (Binary(l), Binary(r)) => l == r,
            #[cfg(feature = "dtype-time")]
            (Time(l), Time(r)) => *l == *r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Date(l), Date(r)) => *l == *r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Datetime(l, tul, tzl), Datetime(r, tur, tzr)) => {
                *l == *r && *tul == *tur && tzl == tzr
            },
            (List(l), List(r)) => l == r,
            #[cfg(feature = "dtype-categorical")]
            (Categorical(idx_l, rev_l, ptr_l), Categorical(idx_r, rev_r, ptr_r)) => {
                if !same_revmap(rev_l, *ptr_l, rev_r, *ptr_r) {
                    // We can't support this because our Hash impl directly hashes the index. If you
                    // add support for this we must change the Hash impl.
                    unimplemented!(
                        "comparing categoricals with different revmaps is not supported"
                    );
                }

                idx_l == idx_r
            },
            #[cfg(feature = "dtype-categorical")]
            (Enum(idx_l, rev_l, ptr_l), Enum(idx_r, rev_r, ptr_r)) => {
                // We can't support this because our Hash impl directly hashes the index. If you
                // add support for this we must change the Hash impl.
                if !same_revmap(rev_l, *ptr_l, rev_r, *ptr_r) {
                    unimplemented!("comparing enums with different revmaps is not supported");
                }

                idx_l == idx_r
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(l, tu_l), Duration(r, tu_r)) => l == r && tu_l == tu_r,

            #[cfg(feature = "dtype-struct")]
            (StructOwned(l), StructOwned(r)) => struct_eq_missing(
                struct_owned_value_iter(l.as_ref()),
                struct_owned_value_iter(r.as_ref()),
                null_equal,
            ),
            #[cfg(feature = "dtype-struct")]
            (StructOwned(l), Struct(idx, arr, _)) => struct_eq_missing(
                struct_owned_value_iter(l.as_ref()),
                struct_value_iter(*idx, arr),
                null_equal,
            ),
            #[cfg(feature = "dtype-struct")]
            (Struct(idx, arr, _), StructOwned(r)) => struct_eq_missing(
                struct_value_iter(*idx, arr),
                struct_owned_value_iter(r.as_ref()),
                null_equal,
            ),
            #[cfg(feature = "dtype-struct")]
            (Struct(l_idx, l_arr, _), Struct(r_idx, r_arr, _)) => struct_eq_missing(
                struct_value_iter(*l_idx, l_arr),
                struct_value_iter(*r_idx, r_arr),
                null_equal,
            ),
            #[cfg(feature = "dtype-decimal")]
            (Decimal(l_v, l_s), Decimal(r_v, r_s)) => {
                // l_v / 10**l_s == r_v / 10**r_s
                if l_s == r_s && l_v == r_v || *l_v == 0 && *r_v == 0 {
                    true
                } else if l_s < r_s {
                    // l_v * 10**(r_s - l_s) == r_v
                    if let Some(lhs) = (|| {
                        let exp = i128::checked_pow(10, (r_s - l_s).try_into().ok()?)?;
                        l_v.checked_mul(exp)
                    })() {
                        lhs == *r_v
                    } else {
                        false
                    }
                } else {
                    // l_v == r_v * 10**(l_s - r_s)
                    if let Some(rhs) = (|| {
                        let exp = i128::checked_pow(10, (l_s - r_s).try_into().ok()?)?;
                        r_v.checked_mul(exp)
                    })() {
                        *l_v == rhs
                    } else {
                        false
                    }
                }
            },
            #[cfg(feature = "object")]
            (Object(l), Object(r)) => l == r,
            #[cfg(feature = "dtype-array")]
            (Array(l_values, l_size), Array(r_values, r_size)) => {
                if l_size != r_size {
                    return false;
                }

                debug_assert_eq!(l_values.len(), *l_size);
                debug_assert_eq!(r_values.len(), *r_size);

                let mut is_equal = true;
                for i in 0..*l_size {
                    let l = unsafe { l_values.get_unchecked(i) };
                    let r = unsafe { r_values.get_unchecked(i) };

                    is_equal &= l.eq_missing(&r, null_equal);
                }
                is_equal
            },

            (l, r) if l.to_i128().is_some() && r.to_i128().is_some() => l.to_i128() == r.to_i128(),
            (l, r) if l.to_f64().is_some() && r.to_f64().is_some() => {
                l.to_f64().unwrap().to_total_ord() == r.to_f64().unwrap().to_total_ord()
            },

            (_, _) => {
                unimplemented!(
                    "scalar eq_missing for mixed dtypes {self:?} and {other:?} is not supported"
                )
            },
        }
    }
}

impl PartialEq for AnyValue<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.eq_missing(other, true)
    }
}

impl PartialOrd for AnyValue<'_> {
    /// Only implemented for the same types and physical types!
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use AnyValue::*;
        match (self, &other) {
            // Map to borrowed.
            (StringOwned(l), r) => AnyValue::String(l.as_str()).partial_cmp(r),
            (BinaryOwned(l), r) => AnyValue::Binary(l.as_slice()).partial_cmp(r),
            #[cfg(feature = "object")]
            (ObjectOwned(l), r) => AnyValue::Object(&*l.0).partial_cmp(r),
            (l, StringOwned(r)) => l.partial_cmp(&AnyValue::String(r.as_str())),
            (l, BinaryOwned(r)) => l.partial_cmp(&AnyValue::Binary(r.as_slice())),
            #[cfg(feature = "object")]
            (l, ObjectOwned(r)) => l.partial_cmp(&AnyValue::Object(&*r.0)),
            #[cfg(feature = "dtype-datetime")]
            (DatetimeOwned(lv, ltu, ltz), r) => {
                Datetime(*lv, *ltu, ltz.as_ref().map(|v| v.as_ref())).partial_cmp(r)
            },
            #[cfg(feature = "dtype-datetime")]
            (l, DatetimeOwned(rv, rtu, rtz)) => {
                l.partial_cmp(&Datetime(*rv, *rtu, rtz.as_ref().map(|v| v.as_ref())))
            },
            #[cfg(feature = "dtype-categorical")]
            (CategoricalOwned(lv, lrev, larr), r) => {
                Categorical(*lv, lrev.as_ref(), *larr).partial_cmp(r)
            },
            #[cfg(feature = "dtype-categorical")]
            (l, CategoricalOwned(rv, rrev, rarr)) => {
                l.partial_cmp(&Categorical(*rv, rrev.as_ref(), *rarr))
            },
            #[cfg(feature = "dtype-categorical")]
            (EnumOwned(lv, lrev, larr), r) => Enum(*lv, lrev.as_ref(), *larr).partial_cmp(r),
            #[cfg(feature = "dtype-categorical")]
            (l, EnumOwned(rv, rrev, rarr)) => l.partial_cmp(&Enum(*rv, rrev.as_ref(), *rarr)),

            // Comparison with null.
            (Null, Null) => Some(Ordering::Equal),
            (Null, _) => Some(Ordering::Less),
            (_, Null) => Some(Ordering::Greater),

            // Comparison between equal types.
            (Boolean(l), Boolean(r)) => l.partial_cmp(r),
            (UInt8(l), UInt8(r)) => l.partial_cmp(r),
            (UInt16(l), UInt16(r)) => l.partial_cmp(r),
            (UInt32(l), UInt32(r)) => l.partial_cmp(r),
            (UInt64(l), UInt64(r)) => l.partial_cmp(r),
            (Int8(l), Int8(r)) => l.partial_cmp(r),
            (Int16(l), Int16(r)) => l.partial_cmp(r),
            (Int32(l), Int32(r)) => l.partial_cmp(r),
            (Int64(l), Int64(r)) => l.partial_cmp(r),
            (Int128(l), Int128(r)) => l.partial_cmp(r),
            (Float32(l), Float32(r)) => Some(l.tot_cmp(r)),
            (Float64(l), Float64(r)) => Some(l.tot_cmp(r)),
            (String(l), String(r)) => l.partial_cmp(r),
            (Binary(l), Binary(r)) => l.partial_cmp(r),
            #[cfg(feature = "dtype-date")]
            (Date(l), Date(r)) => l.partial_cmp(r),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(lt, lu, lz), Datetime(rt, ru, rz)) => {
                if lu != ru || lz != rz {
                    unimplemented!(
                        "comparing datetimes with different units or timezones is not supported"
                    );
                }

                lt.partial_cmp(rt)
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(lt, lu), Duration(rt, ru)) => {
                if lu != ru {
                    unimplemented!("comparing durations with different units is not supported");
                }

                lt.partial_cmp(rt)
            },
            #[cfg(feature = "dtype-time")]
            (Time(l), Time(r)) => l.partial_cmp(r),
            #[cfg(feature = "dtype-categorical")]
            (Categorical(..), Categorical(..)) => {
                unimplemented!(
                    "can't order categoricals as AnyValues, dtype for ordering is needed"
                )
            },
            #[cfg(feature = "dtype-categorical")]
            (Enum(..), Enum(..)) => {
                unimplemented!("can't order enums as AnyValues, dtype for ordering is needed")
            },
            (List(_), List(_)) => {
                unimplemented!("ordering for List dtype is not supported")
            },
            #[cfg(feature = "dtype-array")]
            (Array(..), Array(..)) => {
                unimplemented!("ordering for Array dtype is not supported")
            },
            #[cfg(feature = "object")]
            (Object(_), Object(_)) => {
                unimplemented!("ordering for Object dtype is not supported")
            },
            #[cfg(feature = "dtype-struct")]
            (StructOwned(_), StructOwned(_))
            | (StructOwned(_), Struct(..))
            | (Struct(..), StructOwned(_))
            | (Struct(..), Struct(..)) => {
                unimplemented!("ordering for Struct dtype is not supported")
            },
            #[cfg(feature = "dtype-decimal")]
            (Decimal(l_v, l_s), Decimal(r_v, r_s)) => {
                // l_v / 10**l_s <=> r_v / 10**r_s
                if l_s == r_s && l_v == r_v || *l_v == 0 && *r_v == 0 {
                    Some(Ordering::Equal)
                } else if l_s < r_s {
                    // l_v * 10**(r_s - l_s) <=> r_v
                    if let Some(lhs) = (|| {
                        let exp = i128::checked_pow(10, (r_s - l_s).try_into().ok()?)?;
                        l_v.checked_mul(exp)
                    })() {
                        lhs.partial_cmp(r_v)
                    } else {
                        Some(Ordering::Greater)
                    }
                } else {
                    // l_v <=> r_v * 10**(l_s - r_s)
                    if let Some(rhs) = (|| {
                        let exp = i128::checked_pow(10, (l_s - r_s).try_into().ok()?)?;
                        r_v.checked_mul(exp)
                    })() {
                        l_v.partial_cmp(&rhs)
                    } else {
                        Some(Ordering::Less)
                    }
                }
            },

            (_, _) => {
                unimplemented!(
                    "scalar ordering for mixed dtypes {self:?} and {other:?} is not supported"
                )
            },
        }
    }
}

impl TotalEq for AnyValue<'_> {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self.eq_missing(other, true)
    }
}

#[cfg(feature = "dtype-struct")]
fn struct_to_avs_static(idx: usize, arr: &StructArray, fields: &[Field]) -> Vec<AnyValue<'static>> {
    assert!(idx < arr.len());

    let arrs = arr.values();

    debug_assert_eq!(arrs.len(), fields.len());

    arrs.iter()
        .zip(fields)
        .map(|(arr, field)| {
            // SAFETY: We asserted above that the length of StructArray is larger than `idx`. Since
            // StructArray has the invariant that each array is the same length. This is okay to do
            // now.
            unsafe { arr_to_any_value(arr.as_ref(), idx, &field.dtype) }.into_static()
        })
        .collect()
}

#[cfg(feature = "dtype-categorical")]
fn same_revmap(
    rev_l: &RevMapping,
    ptr_l: SyncPtr<Utf8ViewArray>,
    rev_r: &RevMapping,
    ptr_r: SyncPtr<Utf8ViewArray>,
) -> bool {
    if ptr_l.is_null() && ptr_r.is_null() {
        match (rev_l, rev_r) {
            (RevMapping::Global(_, _, id_l), RevMapping::Global(_, _, id_r)) => id_l == id_r,
            (RevMapping::Local(_, id_l), RevMapping::Local(_, id_r)) => id_l == id_r,
            _ => false,
        }
    } else {
        ptr_l == ptr_r
    }
}

pub trait GetAnyValue {
    /// # Safety
    ///
    /// Get an value without doing bound checks.
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue;
}

impl GetAnyValue for ArrayRef {
    // Should only be called with physical types
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        match self.dtype() {
            ArrowDataType::Int8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i8>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int8(v),
                }
            },
            ArrowDataType::Int16 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i16>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int16(v),
                }
            },
            ArrowDataType::Int32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i32>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int32(v),
                }
            },
            ArrowDataType::Int64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i64>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int64(v),
                }
            },
            ArrowDataType::Int128 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i128>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int128(v),
                }
            },
            ArrowDataType::UInt8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u8>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt8(v),
                }
            },
            ArrowDataType::UInt16 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u16>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt16(v),
                }
            },
            ArrowDataType::UInt32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u32>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt32(v),
                }
            },
            ArrowDataType::UInt64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u64>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt64(v),
                }
            },
            ArrowDataType::Float32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Float32(v),
                }
            },
            ArrowDataType::Float64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Float64(v),
                }
            },
            ArrowDataType::Boolean => {
                let arr = self
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Boolean(v),
                }
            },
            ArrowDataType::LargeUtf8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::String(v),
                }
            },
            _ => unimplemented!(),
        }
    }
}

impl<K: NumericNative> From<K> for AnyValue<'static> {
    fn from(value: K) -> Self {
        unsafe {
            match K::PRIMITIVE {
                PrimitiveType::Int8 => AnyValue::Int8(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int16 => AnyValue::Int16(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int32 => AnyValue::Int32(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int64 => AnyValue::Int64(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int128 => AnyValue::Int128(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt8 => AnyValue::UInt8(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt16 => AnyValue::UInt16(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt32 => AnyValue::UInt32(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt64 => AnyValue::UInt64(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Float32 => {
                    AnyValue::Float32(NumCast::from(value).unwrap_unchecked())
                },
                PrimitiveType::Float64 => {
                    AnyValue::Float64(NumCast::from(value).unwrap_unchecked())
                },
                // not supported by polars
                _ => unreachable!(),
            }
        }
    }
}

impl<'a> From<&'a [u8]> for AnyValue<'a> {
    fn from(value: &'a [u8]) -> Self {
        AnyValue::Binary(value)
    }
}

impl<'a> From<&'a str> for AnyValue<'a> {
    fn from(value: &'a str) -> Self {
        AnyValue::String(value)
    }
}

impl From<bool> for AnyValue<'static> {
    fn from(value: bool) -> Self {
        AnyValue::Boolean(value)
    }
}

#[cfg(test)]
mod test {
    #[cfg(feature = "dtype-categorical")]
    use super::*;

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_arrow_dtypes_to_polars() {
        let dtypes = [
            (
                ArrowDataType::Duration(ArrowTimeUnit::Nanosecond),
                DataType::Duration(TimeUnit::Nanoseconds),
            ),
            (
                ArrowDataType::Duration(ArrowTimeUnit::Millisecond),
                DataType::Duration(TimeUnit::Milliseconds),
            ),
            (
                ArrowDataType::Date64,
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                DataType::Datetime(TimeUnit::Nanoseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, None),
                DataType::Datetime(TimeUnit::Microseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Millisecond, None),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, None),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, Some(PlSmallStr::EMPTY)),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (ArrowDataType::LargeUtf8, DataType::String),
            (ArrowDataType::Utf8, DataType::String),
            (ArrowDataType::LargeBinary, DataType::Binary),
            (ArrowDataType::Binary, DataType::Binary),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Nanosecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Millisecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Microsecond),
                DataType::Time,
            ),
            (ArrowDataType::Time64(ArrowTimeUnit::Second), DataType::Time),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Nanosecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Millisecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Microsecond),
                DataType::Time,
            ),
            (ArrowDataType::Time32(ArrowTimeUnit::Second), DataType::Time),
            (
                ArrowDataType::List(Box::new(ArrowField::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
            (
                ArrowDataType::LargeList(Box::new(ArrowField::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
        ];

        for (dt_a, dt_p) in dtypes {
            let dt = DataType::from_arrow_dtype(&dt_a);

            assert_eq!(dt_p, dt);
        }
    }
}
