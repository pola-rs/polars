//! # Data types supported by Polars.
//!
//! At the moment Polars doesn't include all data types available by Arrow. The goal is to
//! incrementally support more data types and prioritize these by usability.
//!
//! [See the AnyValue variants](enum.AnyValue.html#variants) for the data types that
//! are currently supported.
//!
#[cfg(feature = "serde")]
mod _serde;
mod dtype;
mod field;

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub, SubAssign};

use ahash::RandomState;
use arrow::compute::arithmetics::basic::NativeArithmetics;
use arrow::compute::comparison::Simd8;
#[cfg(feature = "dtype-categorical")]
use arrow::datatypes::IntegerType;
pub use arrow::datatypes::{DataType as ArrowDataType, TimeUnit as ArrowTimeUnit};
use arrow::types::simd::Simd;
use arrow::types::NativeType;
pub use dtype::*;
pub use field::*;
use num::{Bounded, FromPrimitive, Num, NumCast, Zero};
use polars_arrow::data_types::IsFloat;
#[cfg(feature = "serde")]
use serde::de::{EnumAccess, Error, Unexpected, VariantAccess, Visitor};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use serde::{Deserializer, Serializer};

pub use crate::chunked_array::logical::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::prelude::*;
use crate::utils::Wrap;

pub struct Utf8Type {}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ListType {}

pub trait PolarsDataType: Send + Sync {
    fn get_dtype() -> DataType
    where
        Self: Sized;
}

macro_rules! impl_polars_datatype {
    ($ca:ident, $variant:ident, $physical:ty) => {
        pub struct $ca {}

        impl PolarsDataType for $ca {
            fn get_dtype() -> DataType {
                DataType::$variant
            }
        }
    };
}

impl_polars_datatype!(UInt8Type, UInt8, u8);
impl_polars_datatype!(UInt16Type, UInt16, u16);
impl_polars_datatype!(UInt32Type, UInt32, u32);
impl_polars_datatype!(UInt64Type, UInt64, u64);
impl_polars_datatype!(Int8Type, Int8, i8);
impl_polars_datatype!(Int16Type, Int16, i16);
impl_polars_datatype!(Int32Type, Int32, i32);
impl_polars_datatype!(Int64Type, Int64, i64);
impl_polars_datatype!(Float32Type, Float32, f32);
impl_polars_datatype!(Float64Type, Float64, f64);
impl_polars_datatype!(DateType, Date, i32);
impl_polars_datatype!(DatetimeType, Unknown, i64);
impl_polars_datatype!(DurationType, Unknown, i64);
impl_polars_datatype!(CategoricalType, Unknown, u32);
impl_polars_datatype!(TimeType, Time, i64);

impl PolarsDataType for Utf8Type {
    fn get_dtype() -> DataType {
        DataType::Utf8
    }
}

pub struct BooleanType {}

impl PolarsDataType for BooleanType {
    fn get_dtype() -> DataType {
        DataType::Boolean
    }
}

impl PolarsDataType for ListType {
    fn get_dtype() -> DataType {
        // null as we cannot no anything without self.
        DataType::List(Box::new(DataType::Null))
    }
}

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
pub struct ObjectType<T>(T);
#[cfg(feature = "object")]
pub type ObjectChunked<T> = ChunkedArray<ObjectType<T>>;

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T: PolarsObject> PolarsDataType for ObjectType<T> {
    fn get_dtype() -> DataType {
        DataType::Object(T::type_name())
    }
}

/// Any type that is not nested
pub trait PolarsSingleType: PolarsDataType {}

impl<T> PolarsSingleType for T where T: NativeType + PolarsDataType {}

impl PolarsSingleType for Utf8Type {}

pub type ListChunked = ChunkedArray<ListType>;
pub type BooleanChunked = ChunkedArray<BooleanType>;
pub type UInt8Chunked = ChunkedArray<UInt8Type>;
pub type UInt16Chunked = ChunkedArray<UInt16Type>;
pub type UInt32Chunked = ChunkedArray<UInt32Type>;
pub type UInt64Chunked = ChunkedArray<UInt64Type>;
pub type Int8Chunked = ChunkedArray<Int8Type>;
pub type Int16Chunked = ChunkedArray<Int16Type>;
pub type Int32Chunked = ChunkedArray<Int32Type>;
pub type Int64Chunked = ChunkedArray<Int64Type>;
pub type Float32Chunked = ChunkedArray<Float32Type>;
pub type Float64Chunked = ChunkedArray<Float64Type>;
pub type Utf8Chunked = ChunkedArray<Utf8Type>;

pub trait NumericNative:
    PartialOrd
    + NativeType
    + Num
    + NumCast
    + Zero
    + Simd
    + Simd8
    + std::iter::Sum<Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + Bounded
    + FromPrimitive
    + IsFloat
    + NativeArithmetics
{
    type POLARSTYPE;
}
impl NumericNative for i8 {
    type POLARSTYPE = Int8Type;
}
impl NumericNative for i16 {
    type POLARSTYPE = Int16Type;
}
impl NumericNative for i32 {
    type POLARSTYPE = Int32Type;
}
impl NumericNative for i64 {
    type POLARSTYPE = Int64Type;
}
impl NumericNative for u8 {
    type POLARSTYPE = UInt8Type;
}
impl NumericNative for u16 {
    type POLARSTYPE = UInt16Type;
}
impl NumericNative for u32 {
    type POLARSTYPE = UInt32Type;
}
impl NumericNative for u64 {
    type POLARSTYPE = UInt64Type;
}
impl NumericNative for f32 {
    type POLARSTYPE = Float32Type;
}
impl NumericNative for f64 {
    type POLARSTYPE = Float64Type;
}

pub trait PolarsNumericType: Send + Sync + PolarsDataType + 'static {
    type Native: NumericNative;
}
impl PolarsNumericType for UInt8Type {
    type Native = u8;
}
impl PolarsNumericType for UInt16Type {
    type Native = u16;
}
impl PolarsNumericType for UInt32Type {
    type Native = u32;
}
impl PolarsNumericType for UInt64Type {
    type Native = u64;
}
impl PolarsNumericType for Int8Type {
    type Native = i8;
}
impl PolarsNumericType for Int16Type {
    type Native = i16;
}
impl PolarsNumericType for Int32Type {
    type Native = i32;
}
impl PolarsNumericType for Int64Type {
    type Native = i64;
}
impl PolarsNumericType for Float32Type {
    type Native = f32;
}
impl PolarsNumericType for Float64Type {
    type Native = f64;
}

pub trait PolarsIntegerType: PolarsNumericType {}
impl PolarsIntegerType for UInt8Type {}
impl PolarsIntegerType for UInt16Type {}
impl PolarsIntegerType for UInt32Type {}
impl PolarsIntegerType for UInt64Type {}
impl PolarsIntegerType for Int8Type {}
impl PolarsIntegerType for Int16Type {}
impl PolarsIntegerType for Int32Type {}
impl PolarsIntegerType for Int64Type {}

pub trait PolarsFloatType: PolarsNumericType {}
impl PolarsFloatType for Float32Type {}
impl PolarsFloatType for Float64Type {}

#[derive(Debug, Clone)]
pub enum AnyValue<'a> {
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    Utf8(&'a str),
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
    Datetime(i64, TimeUnit, &'a Option<TimeZone>),
    // A 64-bit integer representing difference between date-times in [`TimeUnit`]
    #[cfg(feature = "dtype-duration")]
    Duration(i64, TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    #[cfg(feature = "dtype-time")]
    Time(i64),
    #[cfg(feature = "dtype-categorical")]
    Categorical(u32, &'a RevMapping),
    /// Nested type, contains arrays that are filled with one of the datetypes.
    List(Series),
    #[cfg(feature = "object")]
    /// Can be used to fmt and implements Any, so can be downcasted to the proper value type.
    Object(&'a dyn PolarsObjectSafe),
    #[cfg(feature = "dtype-struct")]
    Struct(Vec<AnyValue<'a>>, &'a [Field]),
    #[cfg(feature = "dtype-struct")]
    StructOwned(Box<(Vec<AnyValue<'a>>, Vec<Field>)>),
    /// A UTF8 encoded string type.
    Utf8Owned(String),
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
            AnyValue::UInt8(v) => serializer.serialize_newtype_variant(name, 5, "UInt8", v),
            AnyValue::UInt16(v) => serializer.serialize_newtype_variant(name, 6, "UInt16", v),
            AnyValue::UInt32(v) => serializer.serialize_newtype_variant(name, 7, "UInt32", v),
            AnyValue::UInt64(v) => serializer.serialize_newtype_variant(name, 8, "UInt64", v),
            AnyValue::Float32(v) => serializer.serialize_newtype_variant(name, 9, "Float32", v),
            AnyValue::Float64(v) => serializer.serialize_newtype_variant(name, 10, "Float64", v),
            AnyValue::List(v) => serializer.serialize_newtype_variant(name, 11, "List", v),
            AnyValue::Boolean(v) => serializer.serialize_newtype_variant(name, 12, "Bool", v),
            // both utf8 variants same number
            AnyValue::Utf8(v) => serializer.serialize_newtype_variant(name, 13, "Utf8Owned", v),
            AnyValue::Utf8Owned(v) => {
                serializer.serialize_newtype_variant(name, 13, "Utf8Owned", v)
            }
            _ => todo!(),
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
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
            List,
            Bool,
            Utf8Owned,
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
            "Float32",
            "Float64",
            "List",
            "Boolean",
            "Utf8Owned",
        ];
        const LAST: u8 = unsafe { std::mem::transmute::<_, u8>(AvField::Utf8Owned) };

        struct FieldVisitor;

        impl Visitor<'_> for FieldVisitor {
            type Value = AvField;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(formatter, "an integer between 0-{}", LAST)
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

                // safety:
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
                    b"UInt8" => AvField::UInt8,
                    b"UInt16" => AvField::UInt16,
                    b"UInt32" => AvField::UInt32,
                    b"UInt64" => AvField::UInt64,
                    b"Float32" => AvField::Float32,
                    b"Float64" => AvField::Float64,
                    b"List" => AvField::List,
                    b"Bool" => AvField::Bool,
                    b"Utf8Owned" | b"Utf8" => AvField::Utf8Owned,
                    _ => {
                        return Err(serde::de::Error::unknown_variant(
                            &String::from_utf8_lossy(v),
                            VARIANTS,
                        ))
                    }
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
                    }
                    (AvField::Int16, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int16(value)
                    }
                    (AvField::Int32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int32(value)
                    }
                    (AvField::Int64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int64(value)
                    }
                    (AvField::UInt8, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt8(value)
                    }
                    (AvField::UInt16, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt16(value)
                    }
                    (AvField::UInt32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt32(value)
                    }
                    (AvField::UInt64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt64(value)
                    }
                    (AvField::Float32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Float32(value)
                    }
                    (AvField::Float64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Float64(value)
                    }
                    (AvField::Bool, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Boolean(value)
                    }
                    (AvField::List, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::List(value)
                    }
                    (AvField::Utf8Owned, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Utf8Owned(value)
                    }
                };
                Ok(out)
            }
        }
        deserializer.deserialize_enum("AnyValue", VARIANTS, OuterVisitor)
    }
}

impl<'a> AnyValue<'a> {
    /// Extract a numerical value from the AnyValue
    #[doc(hidden)]
    #[cfg(feature = "private")]
    pub fn extract<T: NumCast>(&self) -> Option<T> {
        use AnyValue::*;
        match self {
            Null => None,
            Int8(v) => NumCast::from(*v),
            Int16(v) => NumCast::from(*v),
            Int32(v) => NumCast::from(*v),
            Int64(v) => NumCast::from(*v),
            UInt8(v) => NumCast::from(*v),
            UInt16(v) => NumCast::from(*v),
            UInt32(v) => NumCast::from(*v),
            UInt64(v) => NumCast::from(*v),
            Float32(v) => NumCast::from(*v),
            Float64(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-date")]
            Date(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, _, _) => NumCast::from(*v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, _) => NumCast::from(*v),
            Boolean(v) => {
                if *v {
                    NumCast::from(1)
                } else {
                    NumCast::from(0)
                }
            }
            _ => unimplemented!(),
        }
    }
}

impl<'a> Hash for AnyValue<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use AnyValue::*;
        match self {
            Null => state.write_u64(u64::MAX / 2 + 135123),
            Int8(v) => state.write_i8(*v),
            Int16(v) => state.write_i16(*v),
            Int32(v) => state.write_i32(*v),
            Int64(v) => state.write_i64(*v),
            UInt8(v) => state.write_u8(*v),
            UInt16(v) => state.write_u16(*v),
            UInt32(v) => state.write_u32(*v),
            UInt64(v) => state.write_u64(*v),
            Utf8(s) => state.write(s.as_bytes()),
            Boolean(v) => state.write_u8(*v as u8),
            List(v) => Hash::hash(&Wrap(v.clone()), state),
            _ => unimplemented!(),
        }
    }
}

impl<'a> Eq for AnyValue<'a> {}

impl From<f64> for AnyValue<'_> {
    fn from(a: f64) -> Self {
        AnyValue::Float64(a)
    }
}
impl From<f32> for AnyValue<'_> {
    fn from(a: f32) -> Self {
        AnyValue::Float32(a)
    }
}
impl From<u32> for AnyValue<'_> {
    fn from(a: u32) -> Self {
        AnyValue::UInt32(a)
    }
}
impl From<u64> for AnyValue<'_> {
    fn from(a: u64) -> Self {
        AnyValue::UInt64(a)
    }
}
impl From<i64> for AnyValue<'_> {
    fn from(a: i64) -> Self {
        AnyValue::Int64(a)
    }
}
impl From<i32> for AnyValue<'_> {
    fn from(a: i32) -> Self {
        AnyValue::Int32(a)
    }
}
impl From<i16> for AnyValue<'_> {
    fn from(a: i16) -> Self {
        AnyValue::Int16(a)
    }
}
impl From<u16> for AnyValue<'_> {
    fn from(a: u16) -> Self {
        AnyValue::UInt16(a)
    }
}

impl From<i8> for AnyValue<'_> {
    fn from(a: i8) -> Self {
        AnyValue::Int8(a)
    }
}
impl From<u8> for AnyValue<'_> {
    fn from(a: u8) -> Self {
        AnyValue::UInt8(a)
    }
}

impl<'a, T> From<Option<T>> for AnyValue<'a>
where
    T: Into<AnyValue<'a>>,
{
    fn from(a: Option<T>) -> Self {
        match a {
            None => AnyValue::Null,
            Some(v) => v.into(),
        }
    }
}

impl<'a> AnyValue<'a> {
    #[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
    pub(crate) fn into_date(self) -> Self {
        match self {
            #[cfg(feature = "dtype-date")]
            AnyValue::Int32(v) => AnyValue::Date(v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {}", dt),
        }
    }
    #[cfg(feature = "dtype-datetime")]
    pub(crate) fn into_datetime(self, tu: TimeUnit, tz: &'a Option<TimeZone>) -> Self {
        match self {
            AnyValue::Int64(v) => AnyValue::Datetime(v, tu, tz),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {}", dt),
        }
    }

    #[cfg(feature = "dtype-duration")]
    pub(crate) fn into_duration(self, tu: TimeUnit) -> Self {
        match self {
            AnyValue::Int64(v) => AnyValue::Duration(v, tu),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {}", dt),
        }
    }

    #[cfg(feature = "dtype-time")]
    pub(crate) fn into_time(self) -> Self {
        match self {
            AnyValue::Int64(v) => AnyValue::Time(v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {}", dt),
        }
    }

    #[must_use]
    pub fn add<'b>(&self, rhs: &AnyValue<'b>) -> Self {
        use AnyValue::*;
        match (self, rhs) {
            (Null, _) => Null,
            (_, Null) => Null,
            (Int32(l), Int32(r)) => Int32(l + r),
            (Int64(l), Int64(r)) => Int64(l + r),
            (UInt32(l), UInt32(r)) => UInt32(l + r),
            (UInt64(l), UInt64(r)) => UInt64(l + r),
            (Float32(l), Float32(r)) => Float32(l + r),
            (Float64(l), Float64(r)) => Float64(l + r),
            _ => todo!(),
        }
    }

    /// Try to coerce to an AnyValue with static lifetime.
    /// This can be done if it does not borrow any values.
    pub fn into_static(self) -> PolarsResult<AnyValue<'static>> {
        use AnyValue::*;
        let av = match self {
            Null => AnyValue::Null,
            Int8(v) => AnyValue::Int8(v),
            Int16(v) => AnyValue::Int16(v),
            Int32(v) => AnyValue::Int32(v),
            Int64(v) => AnyValue::Int64(v),
            UInt8(v) => AnyValue::UInt8(v),
            UInt16(v) => AnyValue::UInt16(v),
            UInt32(v) => AnyValue::UInt32(v),
            UInt64(v) => AnyValue::UInt64(v),
            Boolean(v) => AnyValue::Boolean(v),
            Float32(v) => AnyValue::Float32(v),
            Float64(v) => AnyValue::Float64(v),
            #[cfg(feature = "dtype-date")]
            Date(v) => AnyValue::Date(v),
            #[cfg(feature = "dtype-time")]
            Time(v) => AnyValue::Time(v),
            List(v) => AnyValue::List(v),
            Utf8(s) => AnyValue::Utf8Owned(s.to_string()),
            dt => {
                return Err(PolarsError::ComputeError(
                    format!("cannot get static AnyValue from {}", dt).into(),
                ))
            }
        };
        Ok(av)
    }
}

impl<'a> From<AnyValue<'a>> for Option<i64> {
    fn from(val: AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => None,
            Int32(v) => Some(v as i64),
            Int64(v) => Some(v as i64),
            UInt32(v) => Some(v as i64),
            _ => todo!(),
        }
    }
}

impl PartialEq for AnyValue<'_> {
    // Everything of Any is slow. Don't use.
    fn eq(&self, other: &Self) -> bool {
        use AnyValue::*;
        match (self, other) {
            (Utf8(l), Utf8(r)) => l == r,
            (UInt8(l), UInt8(r)) => l == r,
            (UInt16(l), UInt16(r)) => l == r,
            (UInt32(l), UInt32(r)) => l == r,
            (UInt64(l), UInt64(r)) => l == r,
            (Int8(l), Int8(r)) => l == r,
            (Int16(l), Int16(r)) => l == r,
            (Int32(l), Int32(r)) => l == r,
            (Int64(l), Int64(r)) => l == r,
            (Float32(l), Float32(r)) => l == r,
            (Float64(l), Float64(r)) => l == r,
            #[cfg(feature = "dtype-time")]
            (Time(l), Time(r)) => l == r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Date(l), Date(r)) => l == r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Datetime(l, tul, tzl), Datetime(r, tur, tzr)) => l == r && tul == tur && tzl == tzr,
            (Boolean(l), Boolean(r)) => l == r,
            (List(l), List(r)) => l == r,
            #[cfg(feature = "object")]
            (Object(_), Object(_)) => panic!("eq between object not supported"),
            // should it?
            (Null, Null) => true,
            #[cfg(feature = "dtype-categorical")]
            (Categorical(idx_l, rev_l), Categorical(idx_r, rev_r)) => match (rev_l, rev_r) {
                (RevMapping::Global(_, _, id_l), RevMapping::Global(_, _, id_r)) => {
                    id_l == id_r && idx_l == idx_r
                }
                (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => {
                    std::ptr::eq(arr_l, arr_r) && idx_l == idx_r
                }
                _ => false,
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(l, tu_l), Duration(r, tu_r)) => l == r && tu_l == tu_r,
            _ => false,
        }
    }
}

impl PartialOrd for AnyValue<'_> {
    /// Only implemented for the same types and physical types!
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use AnyValue::*;
        match (self, other) {
            (UInt8(l), UInt8(r)) => l.partial_cmp(r),
            (UInt16(l), UInt16(r)) => l.partial_cmp(r),
            (UInt32(l), UInt32(r)) => l.partial_cmp(r),
            (UInt64(l), UInt64(r)) => l.partial_cmp(r),
            (Int8(l), Int8(r)) => l.partial_cmp(r),
            (Int16(l), Int16(r)) => l.partial_cmp(r),
            (Int32(l), Int32(r)) => l.partial_cmp(r),
            (Int64(l), Int64(r)) => l.partial_cmp(r),
            (Float32(l), Float32(r)) => l.partial_cmp(r),
            (Float64(l), Float64(r)) => l.partial_cmp(r),
            (Utf8(l), Utf8(r)) => l.partial_cmp(r),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde-lazy", feature = "serde"),
    derive(Serialize, Deserialize)
)]
pub enum TimeUnit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
}

impl From<&ArrowTimeUnit> for TimeUnit {
    fn from(tu: &ArrowTimeUnit) -> Self {
        match tu {
            ArrowTimeUnit::Nanosecond => TimeUnit::Nanoseconds,
            ArrowTimeUnit::Microsecond => TimeUnit::Microseconds,
            ArrowTimeUnit::Millisecond => TimeUnit::Milliseconds,
            // will be cast
            ArrowTimeUnit::Second => TimeUnit::Milliseconds,
        }
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeUnit::Nanoseconds => {
                write!(f, "ns")
            }
            TimeUnit::Microseconds => {
                write!(f, "μs")
            }
            TimeUnit::Milliseconds => {
                write!(f, "ms")
            }
        }
    }
}

impl TimeUnit {
    pub fn to_ascii(self) -> &'static str {
        use TimeUnit::*;
        match self {
            Nanoseconds => "ns",
            Microseconds => "us",
            Milliseconds => "ms",
        }
    }

    pub fn to_arrow(self) -> ArrowTimeUnit {
        match self {
            TimeUnit::Nanoseconds => ArrowTimeUnit::Nanosecond,
            TimeUnit::Microseconds => ArrowTimeUnit::Microsecond,
            TimeUnit::Milliseconds => ArrowTimeUnit::Millisecond,
        }
    }
}

#[cfg(feature = "private")]
pub type PlHashMap<K, V> = hashbrown::HashMap<K, V, RandomState>;
#[cfg(feature = "private")]
pub type PlHashSet<V> = hashbrown::HashSet<V, RandomState>;
#[cfg(feature = "private")]
pub type PlIndexMap<K, V> = indexmap::IndexMap<K, V, RandomState>;
#[cfg(feature = "private")]
pub type PlIndexSet<K> = indexmap::IndexSet<K, RandomState>;

#[cfg(not(feature = "bigidx"))]
pub type IdxCa = UInt32Chunked;
#[cfg(feature = "bigidx")]
pub type IdxCa = UInt64Chunked;
pub use polars_arrow::index::{IdxArr, IdxSize};

#[cfg(not(feature = "bigidx"))]
pub const IDX_DTYPE: DataType = DataType::UInt32;
#[cfg(feature = "bigidx")]
pub const IDX_DTYPE: DataType = DataType::UInt64;

#[cfg(not(feature = "bigidx"))]
pub type IdxType = UInt32Type;
#[cfg(feature = "bigidx")]
pub type IdxType = UInt64Type;

pub const NULL_DTYPE: DataType = DataType::Int32;

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
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, Some("".to_string())),
                DataType::Datetime(TimeUnit::Milliseconds, Some("".to_string())),
            ),
            (ArrowDataType::LargeUtf8, DataType::Utf8),
            (ArrowDataType::Utf8, DataType::Utf8),
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
                    "item",
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
            (
                ArrowDataType::LargeList(Box::new(ArrowField::new(
                    "item",
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
            (
                ArrowDataType::Dictionary(IntegerType::UInt32, ArrowDataType::Utf8.into(), false),
                DataType::Categorical(None),
            ),
            (
                ArrowDataType::Dictionary(
                    IntegerType::UInt32,
                    ArrowDataType::LargeUtf8.into(),
                    false,
                ),
                DataType::Categorical(None),
            ),
            (
                ArrowDataType::Dictionary(
                    IntegerType::UInt64,
                    ArrowDataType::LargeUtf8.into(),
                    false,
                ),
                DataType::Categorical(None),
            ),
        ];

        for (dt_a, dt_p) in dtypes {
            let dt: DataType = (&dt_a).into();

            assert_eq!(dt_p, dt);
        }
    }
}
