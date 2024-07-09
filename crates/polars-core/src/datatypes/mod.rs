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
mod into_scalar;
#[cfg(feature = "object")]
mod static_array_collect;
mod time_unit;

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub, SubAssign};

pub use aliases::*;
pub use any_value::*;
pub use arrow::array::{ArrayCollectIterExt, ArrayFromIter, ArrayFromIterDtype, StaticArray};
#[cfg(feature = "dtype-categorical")]
use arrow::datatypes::IntegerType;
pub use arrow::datatypes::{ArrowDataType, TimeUnit as ArrowTimeUnit};
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use bytemuck::Zeroable;
pub use dtype::*;
pub use field::*;
pub use into_scalar::*;
use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Zero};
use polars_compute::arithmetic::HasPrimitiveArithmeticKernel;
use polars_compute::float_sum::FloatSum;
use polars_utils::abs_diff::AbsDiff;
use polars_utils::float::IsFloat;
use polars_utils::min_max::MinMax;
use polars_utils::nulls::IsNull;
#[cfg(feature = "serde")]
use serde::de::{EnumAccess, Error, Unexpected, VariantAccess, Visitor};
#[cfg(any(feature = "serde", feature = "serde-lazy"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "serde", feature = "serde-lazy"))]
use serde::{Deserializer, Serializer};
pub use time_unit::*;

pub use crate::chunked_array::logical::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::prelude::*;
use crate::utils::Wrap;

pub struct TrueT;
pub struct FalseT;

/// # Safety
///
/// The StaticArray and dtype return must be correct.
pub unsafe trait PolarsDataType: Send + Sync + Sized {
    type Physical<'a>: std::fmt::Debug + Clone;
    type OwnedPhysical: std::fmt::Debug + Send + Sync + Clone + PartialEq;
    type ZeroablePhysical<'a>: Zeroable + From<Self::Physical<'a>>;
    type Array: for<'a> StaticArray<
        ValueT<'a> = Self::Physical<'a>,
        ZeroableValueT<'a> = Self::ZeroablePhysical<'a>,
    >;
    type IsNested;
    type HasViews;
    type IsStruct;
    type IsObject;

    fn get_dtype() -> DataType
    where
        Self: Sized;
}

pub trait PolarsNumericType: 'static
where
    Self: for<'a> PolarsDataType<
        OwnedPhysical = Self::Native,
        Physical<'a> = Self::Native,
        ZeroablePhysical<'a> = Self::Native,
        Array = PrimitiveArray<Self::Native>,
        IsNested = FalseT,
        HasViews = FalseT,
        IsStruct = FalseT,
        IsObject = FalseT,
    >,
{
    type Native: NumericNative;
}

pub trait PolarsIntegerType: PolarsNumericType {}
pub trait PolarsFloatType: PolarsNumericType {}

macro_rules! impl_polars_num_datatype {
    ($trait: ident, $ca:ident, $variant:ident, $physical:ty, $owned_phys:ty) => {
        #[derive(Clone, Copy)]
        pub struct $ca {}

        unsafe impl PolarsDataType for $ca {
            type Physical<'a> = $physical;
            type OwnedPhysical = $owned_phys;
            type ZeroablePhysical<'a> = $physical;
            type Array = PrimitiveArray<$physical>;
            type IsNested = FalseT;
            type HasViews = FalseT;
            type IsStruct = FalseT;
            type IsObject = FalseT;

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

macro_rules! impl_polars_datatype_pass_dtype {
    ($ca:ident, $dtype:expr, $arr:ty, $lt:lifetime, $phys:ty, $zerophys:ty, $owned_phys:ty, $has_views:ident) => {
        #[derive(Clone, Copy)]
        pub struct $ca {}

        unsafe impl PolarsDataType for $ca {
            type Physical<$lt> = $phys;
            type OwnedPhysical = $owned_phys;
            type ZeroablePhysical<$lt> = $zerophys;
            type Array = $arr;
            type IsNested = FalseT;
            type HasViews = $has_views;
            type IsStruct = FalseT;
            type IsObject = FalseT;

            #[inline]
            fn get_dtype() -> DataType {
                $dtype
            }
        }
    };
}
macro_rules! impl_polars_binview_datatype {
    ($ca:ident, $variant:ident, $arr:ty, $lt:lifetime, $phys:ty, $zerophys:ty, $owned_phys:ty) => {
        impl_polars_datatype_pass_dtype!(
            $ca,
            DataType::$variant,
            $arr,
            $lt,
            $phys,
            $zerophys,
            $owned_phys,
            TrueT
        );
    };
}

macro_rules! impl_polars_datatype {
    ($ca:ident, $variant:ident, $arr:ty, $lt:lifetime, $phys:ty, $zerophys:ty, $owned_phys:ty) => {
        impl_polars_datatype_pass_dtype!(
            $ca,
            DataType::$variant,
            $arr,
            $lt,
            $phys,
            $zerophys,
            $owned_phys,
            FalseT
        );
    };
}

impl_polars_num_datatype!(PolarsIntegerType, UInt8Type, UInt8, u8, u8);
impl_polars_num_datatype!(PolarsIntegerType, UInt16Type, UInt16, u16, u16);
impl_polars_num_datatype!(PolarsIntegerType, UInt32Type, UInt32, u32, u32);
impl_polars_num_datatype!(PolarsIntegerType, UInt64Type, UInt64, u64, u64);
impl_polars_num_datatype!(PolarsIntegerType, Int8Type, Int8, i8, i8);
impl_polars_num_datatype!(PolarsIntegerType, Int16Type, Int16, i16, i16);
impl_polars_num_datatype!(PolarsIntegerType, Int32Type, Int32, i32, i32);
impl_polars_num_datatype!(PolarsIntegerType, Int64Type, Int64, i64, i64);
impl_polars_num_datatype!(PolarsFloatType, Float32Type, Float32, f32, f32);
impl_polars_num_datatype!(PolarsFloatType, Float64Type, Float64, f64, f64);
impl_polars_datatype!(DateType, Date, PrimitiveArray<i32>, 'a, i32, i32, i32);
impl_polars_datatype!(TimeType, Time, PrimitiveArray<i64>, 'a, i64, i64, i64);
impl_polars_binview_datatype!(StringType, String, Utf8ViewArray, 'a, &'a str, Option<&'a str>, String);
impl_polars_binview_datatype!(BinaryType, Binary, BinaryViewArray, 'a, &'a [u8], Option<&'a [u8]>, Box<[u8]>);
impl_polars_datatype!(BinaryOffsetType, BinaryOffset, BinaryArray<i64>, 'a, &'a [u8], Option<&'a [u8]>, Box<[u8]>);
impl_polars_datatype!(BooleanType, Boolean, BooleanArray, 'a, bool, bool, bool);

#[cfg(feature = "dtype-decimal")]
impl_polars_datatype_pass_dtype!(DecimalType, DataType::Unknown(UnknownKind::Any), PrimitiveArray<i128>, 'a, i128, i128, i128, FalseT);
impl_polars_datatype_pass_dtype!(DatetimeType, DataType::Unknown(UnknownKind::Any), PrimitiveArray<i64>, 'a, i64, i64, i64, FalseT);
impl_polars_datatype_pass_dtype!(DurationType, DataType::Unknown(UnknownKind::Any), PrimitiveArray<i64>, 'a, i64, i64, i64, FalseT);
impl_polars_datatype_pass_dtype!(CategoricalType, DataType::Unknown(UnknownKind::Any), PrimitiveArray<u32>, 'a, u32, u32, u32, FalseT);

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ListType {}
unsafe impl PolarsDataType for ListType {
    type Physical<'a> = Box<dyn Array>;
    type OwnedPhysical = Box<dyn Array>;
    type ZeroablePhysical<'a> = Option<Box<dyn Array>>;
    type Array = ListArray<i64>;
    type IsNested = TrueT;
    type HasViews = FalseT;
    type IsStruct = FalseT;
    type IsObject = FalseT;

    fn get_dtype() -> DataType {
        // Null as we cannot know anything without self.
        DataType::List(Box::new(DataType::Null))
    }
}

#[cfg(feature = "dtype-struct")]
pub struct StructType {}
#[cfg(feature = "dtype-struct")]
unsafe impl PolarsDataType for StructType {
    // The physical types are invalid.
    // We don't want these to be used as that would be
    // very expensive. We use const asserts to ensure
    // traits/methods using the physical types are
    // not called for structs.
    type Physical<'a> = ();
    type OwnedPhysical = ();
    type ZeroablePhysical<'a> = ();
    type Array = StructArray;
    type IsNested = TrueT;
    type HasViews = FalseT;
    type IsStruct = TrueT;
    type IsObject = FalseT;

    fn get_dtype() -> DataType
    where
        Self: Sized,
    {
        DataType::Struct(vec![])
    }
}

#[cfg(feature = "dtype-array")]
pub struct FixedSizeListType {}
#[cfg(feature = "dtype-array")]
unsafe impl PolarsDataType for FixedSizeListType {
    type Physical<'a> = Box<dyn Array>;
    type OwnedPhysical = Box<dyn Array>;
    type ZeroablePhysical<'a> = Option<Box<dyn Array>>;
    type Array = FixedSizeListArray;
    type IsNested = TrueT;
    type HasViews = FalseT;
    type IsStruct = FalseT;
    type IsObject = FalseT;

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
    type OwnedPhysical = i128;
    type ZeroablePhysical<'a> = i128;
    type Array = PrimitiveArray<i128>;
    type IsNested = FalseT;
    type HasViews = FalseT;
    type IsStruct = FalseT;
    type IsObject = FalseT;

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
    type OwnedPhysical = T;
    type ZeroablePhysical<'a> = Option<&'a T>;
    type Array = ObjectArray<T>;
    type IsNested = TrueT;
    type HasViews = FalseT;
    type IsStruct = FalseT;
    type IsObject = TrueT;

    fn get_dtype() -> DataType {
        DataType::Object(T::type_name(), None)
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
pub type StringChunked = ChunkedArray<StringType>;
pub type BinaryChunked = ChunkedArray<BinaryType>;
pub type BinaryOffsetChunked = ChunkedArray<BinaryOffsetType>;
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
    // + Simd8
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
    + HasPrimitiveArithmeticKernel<TrueDivT=<Self::TrueDivPolarsType as PolarsNumericType>::Native>
    + FloatSum<f64>
    + MinMax
    + IsNull
{
    type PolarsType: PolarsNumericType;
    type TrueDivPolarsType: PolarsNumericType;
}

impl NumericNative for i8 {
    type PolarsType = Int8Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for i16 {
    type PolarsType = Int16Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for i32 {
    type PolarsType = Int32Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for i64 {
    type PolarsType = Int64Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for u8 {
    type PolarsType = UInt8Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for u16 {
    type PolarsType = UInt16Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for u32 {
    type PolarsType = UInt32Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for u64 {
    type PolarsType = UInt64Type;
    type TrueDivPolarsType = Float64Type;
}
#[cfg(feature = "dtype-decimal")]
impl NumericNative for i128 {
    type PolarsType = Int128Type;
    type TrueDivPolarsType = Float64Type;
}
impl NumericNative for f32 {
    type PolarsType = Float32Type;
    type TrueDivPolarsType = Float32Type;
}
impl NumericNative for f64 {
    type PolarsType = Float64Type;
    type TrueDivPolarsType = Float64Type;
}
