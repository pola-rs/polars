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
mod aliases;
mod any_value;
mod dtype;
mod field;
mod time_unit;

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub, SubAssign};

use ahash::RandomState;
pub use aliases::*;
pub use any_value::*;
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
pub use time_unit::*;

use crate::chunked_array::arithmetic::ArrayArithmetics;
pub use crate::chunked_array::logical::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::prelude::*;
use crate::utils::Wrap;

pub struct Utf8Type {}

#[cfg(feature = "dtype-binary")]
pub struct BinaryType {}

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
            #[inline]
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
#[cfg(feature = "dtype-i128")]
impl_polars_datatype!(DecimalType, Unknown, i128);
impl_polars_datatype!(DatetimeType, Unknown, i64);
impl_polars_datatype!(DurationType, Unknown, i64);
impl_polars_datatype!(CategoricalType, Unknown, u32);
impl_polars_datatype!(TimeType, Time, i64);

impl PolarsDataType for Utf8Type {
    fn get_dtype() -> DataType {
        DataType::Utf8
    }
}

#[cfg(feature = "dtype-binary")]
impl PolarsDataType for BinaryType {
    fn get_dtype() -> DataType {
        DataType::Binary
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

#[cfg(feature = "dtype-i128")]
pub struct Int128Type {}

#[cfg(feature = "dtype-i128")]
impl PolarsDataType for Int128Type {
    fn get_dtype() -> DataType {
        // we cannot know precision/scale statically
        DataType::Decimal128(None)
    }
}

#[cfg(feature = "object")]
pub struct ObjectType<T>(T);
#[cfg(feature = "object")]
pub type ObjectChunked<T> = ChunkedArray<ObjectType<T>>;

#[cfg(feature = "object")]
impl<T: PolarsObject> PolarsDataType for ObjectType<T> {
    fn get_dtype() -> DataType {
        DataType::Object(T::type_name())
    }
}

/// Any type that is not nested
pub trait PolarsSingleType: PolarsDataType {}

impl<T> PolarsSingleType for T where T: NativeType + PolarsDataType {}

impl PolarsSingleType for Utf8Type {}

#[cfg(feature = "dtype-binary")]
impl PolarsSingleType for BinaryType {}

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
#[cfg(feature = "dtype-i128")]
pub type Int128Chunked = ChunkedArray<Int128Type>;
pub type Float32Chunked = ChunkedArray<Float32Type>;
pub type Float64Chunked = ChunkedArray<Float64Type>;
pub type Utf8Chunked = ChunkedArray<Utf8Type>;
#[cfg(feature = "dtype-binary")]
pub type BinaryChunked = ChunkedArray<BinaryType>;

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
    + ArrayArithmetics
{
    type POLARSTYPE: PolarsNumericType;
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
#[cfg(feature = "dtype-i128")]
impl NumericNative for i128 {
    type POLARSTYPE = Int128Type;
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
#[cfg(feature = "dtype-i128")]
impl PolarsNumericType for Int128Type {
    type Native = i128;
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

// Provide options to cloud providers (credentials, region).
pub type CloudOptions = PlHashMap<String, String>;
