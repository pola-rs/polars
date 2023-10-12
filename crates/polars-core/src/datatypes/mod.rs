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
mod static_array;
mod static_array_collect;
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
use arrow::legacy::data_types::IsFloat;
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use bytemuck::Zeroable;
pub use dtype::*;
pub use field::*;
use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Zero};
use polars_arrow::data_types::IsFloat;
use polars_utils::abs_diff::AbsDiff;
#[cfg(feature = "serde")]
use serde::de::{EnumAccess, Error, Unexpected, VariantAccess, Visitor};
#[cfg(any(feature = "serde", feature = "serde-lazy"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "serde", feature = "serde-lazy"))]
use serde::{Deserializer, Serializer};
pub use static_array::StaticArray;
pub use static_array_collect::{ArrayCollectIterExt, ArrayFromIter, ArrayFromIterDtype};
pub use time_unit::*;

use crate::chunked_array::arithmetic::ArrayArithmetics;
pub use crate::chunked_array::logical::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::prelude::*;
use crate::utils::Wrap;

pub struct Nested;
pub struct Flat;

/// # Safety
///
/// The StaticArray and dtype return must be correct.
pub unsafe trait PolarsDataType: Send + Sync + Sized {
    type Physical<'a>;
    type ZeroablePhysical<'a>: Zeroable + From<Self::Physical<'a>>;
    type Array: for<'a> StaticArray<
        ValueT<'a> = Self::Physical<'a>,
        ZeroableValueT<'a> = Self::ZeroablePhysical<'a>,
    >;
    type Structure;

    fn get_dtype() -> DataType
    where
        Self: Sized;
}

pub trait PolarsNumericType: 'static
where
    Self: for<'a> PolarsDataType<
        Physical<'a> = Self::Native,
        ZeroablePhysical<'a> = Self::Native,
        Array = PrimitiveArray<Self::Native>,
        Structure = Flat,
    >,
{
    type Native: NumericNative;
}

pub trait PolarsIntegerType: PolarsNumericType {}
pub trait PolarsFloatType: PolarsNumericType {}

macro_rules! impl_polars_num_datatype {
    ($trait: ident, $ca:ident, $variant:ident, $physical:ty) => {
        #[derive(Clone, Copy)]
        pub struct $ca {}

        unsafe impl PolarsDataType for $ca {
            type Physical<'a> = $physical;
            type ZeroablePhysical<'a> = $physical;
            type Array = PrimitiveArray<$physical>;
            type Structure = Flat;

            #[inline]
            fn get_dtype() -> DataType {
                DataType::$variant
            }
        }

        impl PolarsNumericType for $ca {
            type Native = $physical;
        }

        impl $trait for $ca {}
    };
}

macro_rules! impl_polars_datatype {
    ($ca:ident, $variant:ident, $arr:ty, $lt:lifetime, $phys:ty, $zerophys:ty) => {
        #[derive(Clone, Copy)]
        pub struct $ca {}

        unsafe impl PolarsDataType for $ca {
            type Physical<$lt> = $phys;
            type ZeroablePhysical<$lt> = $zerophys;
            type Array = $arr;
            type Structure = Flat;

            #[inline]
            fn get_dtype() -> DataType {
                DataType::$variant
            }
        }
    };
}

impl_polars_num_datatype!(PolarsIntegerType, UInt8Type, UInt8, u8);
impl_polars_num_datatype!(PolarsIntegerType, UInt16Type, UInt16, u16);
impl_polars_num_datatype!(PolarsIntegerType, UInt32Type, UInt32, u32);
impl_polars_num_datatype!(PolarsIntegerType, UInt64Type, UInt64, u64);
impl_polars_num_datatype!(PolarsIntegerType, Int8Type, Int8, i8);
impl_polars_num_datatype!(PolarsIntegerType, Int16Type, Int16, i16);
impl_polars_num_datatype!(PolarsIntegerType, Int32Type, Int32, i32);
impl_polars_num_datatype!(PolarsIntegerType, Int64Type, Int64, i64);
impl_polars_num_datatype!(PolarsFloatType, Float32Type, Float32, f32);
impl_polars_num_datatype!(PolarsFloatType, Float64Type, Float64, f64);
impl_polars_datatype!(DateType, Date, PrimitiveArray<i32>, 'a, i32, i32);
#[cfg(feature = "dtype-decimal")]
impl_polars_datatype!(DecimalType, Unknown, PrimitiveArray<i128>, 'a, i128, i128);
impl_polars_datatype!(DatetimeType, Unknown, PrimitiveArray<i64>, 'a, i64, i64);
impl_polars_datatype!(DurationType, Unknown, PrimitiveArray<i64>, 'a, i64, i64);
impl_polars_datatype!(CategoricalType, Unknown, PrimitiveArray<u32>, 'a, u32, u32);
impl_polars_datatype!(TimeType, Time, PrimitiveArray<i64>, 'a, i64, i64);
impl_polars_datatype!(Utf8Type, Utf8, Utf8Array<i64>, 'a, &'a str, Option<&'a str>);
impl_polars_datatype!(BinaryType, Binary, BinaryArray<i64>, 'a, &'a [u8], Option<&'a [u8]>);
impl_polars_datatype!(BooleanType, Boolean, BooleanArray, 'a, bool, bool);

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ListType {}
unsafe impl PolarsDataType for ListType {
    type Physical<'a> = Box<dyn Array>;
    type ZeroablePhysical<'a> = Option<Box<dyn Array>>;
    type Array = ListArray<i64>;
    type Structure = Nested;

    fn get_dtype() -> DataType {
        // Null as we cannot know anything without self.
        DataType::List(Box::new(DataType::Null))
    }
}

#[cfg(feature = "dtype-array")]
pub struct FixedSizeListType {}
#[cfg(feature = "dtype-array")]
unsafe impl PolarsDataType for FixedSizeListType {
    type Physical<'a> = Box<dyn Array>;
    type ZeroablePhysical<'a> = Option<Box<dyn Array>>;
    type Array = FixedSizeListArray;
    type Structure = Nested;

    fn get_dtype() -> DataType {
        // Null as we cannot know anything without self.
        DataType::Array(Box::new(DataType::Null), 0)
    }
}
#[cfg(feature = "dtype-decimal")]
pub struct Int128Type {}
#[cfg(feature = "dtype-decimal")]
unsafe impl PolarsDataType for Int128Type {
    type Physical<'a> = i128;
    type ZeroablePhysical<'a> = i128;
    type Array = PrimitiveArray<i128>;
    type Structure = Flat;

    fn get_dtype() -> DataType {
        // Scale is not None to allow for get_any_value() to work.
        DataType::Decimal(None, Some(0))
    }
}
#[cfg(feature = "dtype-decimal")]
impl PolarsNumericType for Int128Type {
    type Native = i128;
}
#[cfg(feature = "dtype-decimal")]
impl PolarsIntegerType for Int128Type {}
#[cfg(feature = "object")]
pub struct ObjectType<T>(T);
#[cfg(feature = "object")]
unsafe impl<T: PolarsObject> PolarsDataType for ObjectType<T> {
    type Physical<'a> = &'a T;
    type ZeroablePhysical<'a> = Option<&'a T>;
    type Array = ObjectArray<T>;
    type Structure = Nested;

    fn get_dtype() -> DataType {
        DataType::Object(T::type_name())
    }
}

#[cfg(feature = "dtype-array")]
pub type ArrayChunked = ChunkedArray<FixedSizeListType>;
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
#[cfg(feature = "dtype-decimal")]
pub type Int128Chunked = ChunkedArray<Int128Type>;
pub type Float32Chunked = ChunkedArray<Float32Type>;
pub type Float64Chunked = ChunkedArray<Float64Type>;
pub type Utf8Chunked = ChunkedArray<Utf8Type>;
pub type BinaryChunked = ChunkedArray<BinaryType>;
#[cfg(feature = "object")]
pub type ObjectChunked<T> = ChunkedArray<ObjectType<T>>;

pub trait NumericNative:
    TotalOrd
    + PartialOrd
    + NativeType
    + Num
    + NumCast
    + Zero
    + One
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
    + AbsDiff
    + Bounded
    + FromPrimitive
    + IsFloat
    + ArrayArithmetics
{
    type PolarsType: PolarsNumericType;
}

impl NumericNative for i8 {
    type PolarsType = Int8Type;
}
impl NumericNative for i16 {
    type PolarsType = Int16Type;
}
impl NumericNative for i32 {
    type PolarsType = Int32Type;
}
impl NumericNative for i64 {
    type PolarsType = Int64Type;
}
impl NumericNative for u8 {
    type PolarsType = UInt8Type;
}
impl NumericNative for u16 {
    type PolarsType = UInt16Type;
}
impl NumericNative for u32 {
    type PolarsType = UInt32Type;
}
impl NumericNative for u64 {
    type PolarsType = UInt64Type;
}
#[cfg(feature = "dtype-decimal")]
impl NumericNative for i128 {
    type PolarsType = Int128Type;
}
impl NumericNative for f32 {
    type PolarsType = Float32Type;
}
impl NumericNative for f64 {
    type PolarsType = Float64Type;
}
