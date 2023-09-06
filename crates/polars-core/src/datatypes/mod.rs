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
mod from_values;
mod static_array;
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
pub use from_values::ArrayFromElementIter;
use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Zero};
use polars_arrow::data_types::IsFloat;
#[cfg(feature = "serde")]
use serde::de::{EnumAccess, Error, Unexpected, VariantAccess, Visitor};
#[cfg(any(feature = "serde", feature = "serde-lazy"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "serde", feature = "serde-lazy"))]
use serde::{Deserializer, Serializer};
pub use static_array::StaticArray;
pub use time_unit::*;

use crate::chunked_array::arithmetic::ArrayArithmetics;
pub use crate::chunked_array::logical::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::prelude::*;
use crate::utils::Wrap;

pub trait PolarsDataType: Send + Sync + Sized + HasArrayType + HasLogicalType {}

// Important: PolarsNumericType implements PolarsDataType and HasArrayT
// using a blanket implementation. If we added the bounds to the trait itself
// Rust ignores the blanket implementation for type checking and will complain
// about a generic ArrayT given by HasArrayT instead of PrimitiveArray<Native>.
pub trait PolarsNumericType: Send + Sync + Sized + HasLogicalType + 'static {
    type Native: NumericNative;
}
impl<T: PolarsNumericType> PolarsDataType for T {}
unsafe impl<T: PolarsNumericType> HasArrayType for T {
    type Array = PrimitiveArray<T::Native>;
}

pub trait HasLogicalType {
    fn get_dtype() -> DataType
    where
        Self: Sized;
}

/// Gives the underlying array type for a particular data type.
#[doc(hidden)]
pub unsafe trait HasArrayType {
    type Array: StaticArray;
}

/// Gets the physical type associated with a PolarsDataType. Same as T::Native for
/// PolarsNumericTypes.
pub type ArrayT<T> = <T as HasArrayType>::Array;
pub type PhysicalT<'a, T> = <<T as HasArrayType>::Array as StaticArray>::ValueT<'a>;

pub trait PolarsIntegerType: PolarsNumericType {}
pub trait PolarsFloatType: PolarsNumericType {}

macro_rules! impl_polars_num_datatype {
    ($trait: ident, $ca:ident, $variant:ident, $physical:ty) => {
        #[derive(Clone, Copy)]
        pub struct $ca {}

        impl HasLogicalType for $ca {
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
    ($ca:ident, $variant:ident, $arr:ty) => {
        #[derive(Clone, Copy)]
        pub struct $ca {}

        impl HasLogicalType for $ca {
            #[inline]
            fn get_dtype() -> DataType {
                DataType::$variant
            }
        }

        impl PolarsDataType for $ca {}

        unsafe impl HasArrayType for $ca {
            type Array = $arr;
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
impl_polars_datatype!(DateType, Date, PrimitiveArray<i32>);
#[cfg(feature = "dtype-decimal")]
impl_polars_datatype!(DecimalType, Unknown, PrimitiveArray<i128>);
impl_polars_datatype!(DatetimeType, Unknown, PrimitiveArray<i64>);
impl_polars_datatype!(DurationType, Unknown, PrimitiveArray<i64>);
impl_polars_datatype!(CategoricalType, Unknown, PrimitiveArray<u32>);
impl_polars_datatype!(TimeType, Time, PrimitiveArray<i64>);
impl_polars_datatype!(Utf8Type, Utf8, Utf8Array<i64>);
impl_polars_datatype!(BinaryType, Binary, BinaryArray<i64>);
impl_polars_datatype!(BooleanType, Boolean, BooleanArray);

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ListType {}
impl PolarsDataType for ListType {}
impl HasLogicalType for ListType {
    fn get_dtype() -> DataType {
        // Mull as we cannot know anything without self.
        DataType::List(Box::new(DataType::Null))
    }
}
unsafe impl HasArrayType for ListType {
    type Array = ListArray<i64>;
}

#[cfg(feature = "dtype-array")]
pub struct FixedSizeListType {}
#[cfg(feature = "dtype-array")]
impl HasLogicalType for FixedSizeListType {
    fn get_dtype() -> DataType {
        // Null as we cannot know anything without self.
        DataType::Array(Box::new(DataType::Null), 0)
    }
}
#[cfg(feature = "dtype-array")]
impl PolarsDataType for FixedSizeListType {}
#[cfg(feature = "dtype-array")]
unsafe impl HasArrayType for FixedSizeListType {
    type Array = FixedSizeListArray;
}

#[cfg(feature = "dtype-decimal")]
pub struct Int128Type {}
#[cfg(feature = "dtype-decimal")]
impl PolarsNumericType for Int128Type {
    type Native = i128;
}
#[cfg(feature = "dtype-decimal")]
impl PolarsIntegerType for Int128Type {}
#[cfg(feature = "dtype-decimal")]
impl HasLogicalType for Int128Type {
    fn get_dtype() -> DataType {
        DataType::Decimal(None, Some(0)) // scale is not None to allow for get_any_value() to work
    }
}

#[cfg(feature = "object")]
pub struct ObjectType<T>(T);
#[cfg(feature = "object")]
impl<T: PolarsObject> PolarsDataType for ObjectType<T> {}
#[cfg(feature = "object")]
impl<T: PolarsObject> HasLogicalType for ObjectType<T> {
    fn get_dtype() -> DataType {
        DataType::Object(T::type_name())
    }
}
#[cfg(feature = "object")]
unsafe impl<T: PolarsObject> HasArrayType for ObjectType<T> {
    type Array = crate::chunked_array::object::ObjectArray<T>;
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
    PartialOrd
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

// Provide options to cloud providers (credentials, region).
pub type CloudOptions = PlHashMap<String, String>;
