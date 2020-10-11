//! # Data types supported by Polars.
//!
//! At the moment Polars doesn't include all data types available by Arrow. The goal is to
//! incrementally support more data types and prioritize these by usability.
//!
//! [See the AnyType variants](enum.AnyType.html#variants) for the data types that
//! are currently supported.
//!
use crate::chunked_array::ChunkedArray;
use crate::series::Series;
pub use arrow::datatypes::DataType as ArrowDataType;
pub use arrow::datatypes::{
    ArrowNumericType, ArrowPrimitiveType, BooleanType, Date32Type, Date64Type, DateUnit,
    DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType, DurationSecondType,
    Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, IntervalDayTimeType,
    IntervalUnit, IntervalYearMonthType, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};

#[cfg(feature = "simd")]
use arrow::datatypes::ToByteSlice;
#[cfg(feature = "simd")]
use packed_simd_2::*;
#[cfg(feature = "simd")]
use std::ops::{Add, Div, Mul, Sub};

pub struct Utf8Type {}

pub struct LargeListType {}

pub trait PolarsDataType {
    fn get_data_type() -> ArrowDataType;
}

impl<T> PolarsDataType for T
where
    T: ArrowPrimitiveType,
{
    fn get_data_type() -> ArrowDataType {
        T::get_data_type()
    }
}

impl PolarsDataType for Utf8Type {
    fn get_data_type() -> ArrowDataType {
        ArrowDataType::Utf8
    }
}

impl PolarsDataType for LargeListType {
    fn get_data_type() -> ArrowDataType {
        // null as we cannot no anything without self.
        ArrowDataType::LargeList(Box::new(ArrowDataType::Null))
    }
}

/// Any type that is not nested
pub trait PolarsSingleType: PolarsDataType {}

impl<T> PolarsSingleType for T where T: ArrowPrimitiveType + PolarsDataType {}

impl PolarsSingleType for Utf8Type {}

pub type LargeListChunked = ChunkedArray<LargeListType>;
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
pub type Date32Chunked = ChunkedArray<Date32Type>;
pub type Date64Chunked = ChunkedArray<Date64Type>;
pub type DurationNanosecondChunked = ChunkedArray<DurationNanosecondType>;
pub type DurationMicrosecondChunked = ChunkedArray<DurationMicrosecondType>;
pub type DurationMillisecondChunked = ChunkedArray<DurationMillisecondType>;
pub type DurationSecondChunked = ChunkedArray<DurationSecondType>;

pub type Time64NanosecondChunked = ChunkedArray<Time64NanosecondType>;
pub type Time64MicrosecondChunked = ChunkedArray<Time64MicrosecondType>;
pub type Time32MillisecondChunked = ChunkedArray<Time32MillisecondType>;
pub type Time32SecondChunked = ChunkedArray<Time32SecondType>;
pub type IntervalDayTimeChunked = ChunkedArray<IntervalDayTimeType>;
pub type IntervalYearMonthChunked = ChunkedArray<IntervalYearMonthType>;

pub type TimestampNanosecondChunked = ChunkedArray<TimestampNanosecondType>;
pub type TimestampMicrosecondChunked = ChunkedArray<TimestampMicrosecondType>;
pub type TimestampMillisecondChunked = ChunkedArray<TimestampMillisecondType>;
pub type TimestampSecondChunked = ChunkedArray<TimestampSecondType>;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
pub trait PolarsNumericType: ArrowNumericType
where
    Self::Simd: Add<Output = Self::Simd>
        + Sub<Output = Self::Simd>
        + Mul<Output = Self::Simd>
        + Div<Output = Self::Simd>
        + Copy,
{
    /// Defines the SIMD type that should be used for this numeric type
    type Simd;

    /// Defines the SIMD Mask type that should be used for this numeric type
    type SimdMask;

    /// The number of SIMD lanes available
    fn lanes() -> usize;

    /// Initializes a SIMD register to a constant value
    fn init(value: Self::Native) -> Self::Simd;

    /// Loads a slice into a SIMD register
    fn load(slice: &[Self::Native]) -> Self::Simd;

    /// Creates a new SIMD mask for this SIMD type filling it with `value`
    fn mask_init(value: bool) -> Self::SimdMask;

    /// Gets the value of a single lane in a SIMD mask
    fn mask_get(mask: &Self::SimdMask, idx: usize) -> bool;

    /// Gets the bitmask for a SimdMask as a byte slice and passes it to the closure used as the action parameter
    fn bitmask<T>(mask: &Self::SimdMask, action: T)
    where
        T: FnMut(&[u8]);

    /// Sets the value of a single lane of a SIMD mask
    fn mask_set(mask: Self::SimdMask, idx: usize, value: bool) -> Self::SimdMask;

    /// Selects elements of `a` and `b` using `mask`
    fn mask_select(mask: Self::SimdMask, a: Self::Simd, b: Self::Simd) -> Self::Simd;

    /// Returns `true` if any of the lanes in the mask are `true`
    fn mask_any(mask: Self::SimdMask) -> bool;

    /// Performs a SIMD binary operation
    fn bin_op<F: Fn(Self::Simd, Self::Simd) -> Self::Simd>(
        left: Self::Simd,
        right: Self::Simd,
        op: F,
    ) -> Self::Simd;

    /// SIMD version of equal
    fn eq(left: Self::Simd, right: Self::Simd) -> Self::SimdMask;

    /// SIMD version of not equal
    fn ne(left: Self::Simd, right: Self::Simd) -> Self::SimdMask;

    /// SIMD version of less than
    fn lt(left: Self::Simd, right: Self::Simd) -> Self::SimdMask;

    /// SIMD version of less than or equal to
    fn le(left: Self::Simd, right: Self::Simd) -> Self::SimdMask;

    /// SIMD version of greater than
    fn gt(left: Self::Simd, right: Self::Simd) -> Self::SimdMask;

    /// SIMD version of greater than or equal to
    fn ge(left: Self::Simd, right: Self::Simd) -> Self::SimdMask;

    /// Writes a SIMD result back to a slice
    fn write(simd_result: Self::Simd, slice: &mut [Self::Native]);
}

#[cfg(any(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    not(feature = "simd")
))]
pub trait PolarsNumericType: ArrowNumericType {}

macro_rules! make_numeric_type {
    ($impl_ty:ty, $native_ty:ty, $simd_ty:ident, $simd_mask_ty:ident) => {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        impl PolarsNumericType for $impl_ty {
            type Simd = $simd_ty;

            type SimdMask = $simd_mask_ty;

            fn lanes() -> usize {
                Self::Simd::lanes()
            }

            fn init(value: Self::Native) -> Self::Simd {
                Self::Simd::splat(value)
            }

            fn load(slice: &[Self::Native]) -> Self::Simd {
                unsafe { Self::Simd::from_slice_unaligned_unchecked(slice) }
            }

            fn mask_init(value: bool) -> Self::SimdMask {
                Self::SimdMask::splat(value)
            }

            fn mask_get(mask: &Self::SimdMask, idx: usize) -> bool {
                unsafe { mask.extract_unchecked(idx) }
            }

            fn bitmask<T>(mask: &Self::SimdMask, mut action: T)
            where
                T: FnMut(&[u8]),
            {
                action(mask.bitmask().to_byte_slice());
            }

            fn mask_set(mask: Self::SimdMask, idx: usize, value: bool) -> Self::SimdMask {
                unsafe { mask.replace_unchecked(idx, value) }
            }

            /// Selects elements of `a` and `b` using `mask`
            fn mask_select(mask: Self::SimdMask, a: Self::Simd, b: Self::Simd) -> Self::Simd {
                mask.select(a, b)
            }

            fn mask_any(mask: Self::SimdMask) -> bool {
                mask.any()
            }

            fn bin_op<F: Fn(Self::Simd, Self::Simd) -> Self::Simd>(
                left: Self::Simd,
                right: Self::Simd,
                op: F,
            ) -> Self::Simd {
                op(left, right)
            }

            fn eq(left: Self::Simd, right: Self::Simd) -> Self::SimdMask {
                left.eq(right)
            }

            fn ne(left: Self::Simd, right: Self::Simd) -> Self::SimdMask {
                left.ne(right)
            }

            fn lt(left: Self::Simd, right: Self::Simd) -> Self::SimdMask {
                left.lt(right)
            }

            fn le(left: Self::Simd, right: Self::Simd) -> Self::SimdMask {
                left.le(right)
            }

            fn gt(left: Self::Simd, right: Self::Simd) -> Self::SimdMask {
                left.gt(right)
            }

            fn ge(left: Self::Simd, right: Self::Simd) -> Self::SimdMask {
                left.ge(right)
            }

            fn write(simd_result: Self::Simd, slice: &mut [Self::Native]) {
                unsafe { simd_result.write_to_slice_unaligned_unchecked(slice) };
            }
        }
        #[cfg(any(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            not(feature = "simd")
        ))]
        impl PolarsNumericType for $impl_ty {}
    };
}

make_numeric_type!(Int8Type, i8, i8x64, m8x64);
make_numeric_type!(Int16Type, i16, i16x32, m16x32);
make_numeric_type!(Int32Type, i32, i32x16, m32x16);
make_numeric_type!(Int64Type, i64, i64x8, m64x8);
make_numeric_type!(UInt8Type, u8, u8x64, m8x64);
make_numeric_type!(UInt16Type, u16, u16x32, m16x32);
make_numeric_type!(UInt32Type, u32, u32x16, m32x16);
make_numeric_type!(UInt64Type, u64, u64x8, m64x8);
make_numeric_type!(Float32Type, f32, f32x16, m32x16);
make_numeric_type!(Float64Type, f64, f64x8, m64x8);

make_numeric_type!(TimestampSecondType, i64, i64x8, m64x8);
make_numeric_type!(TimestampMillisecondType, i64, i64x8, m64x8);
make_numeric_type!(TimestampMicrosecondType, i64, i64x8, m64x8);
make_numeric_type!(TimestampNanosecondType, i64, i64x8, m64x8);
make_numeric_type!(Date32Type, i32, i32x16, m32x16);
make_numeric_type!(Date64Type, i64, i64x8, m64x8);
make_numeric_type!(Time32SecondType, i32, i32x16, m32x16);
make_numeric_type!(Time32MillisecondType, i32, i32x16, m32x16);
make_numeric_type!(Time64MicrosecondType, i64, i64x8, m64x8);
make_numeric_type!(Time64NanosecondType, i64, i64x8, m64x8);
make_numeric_type!(IntervalYearMonthType, i32, i32x16, m32x16);
make_numeric_type!(IntervalDayTimeType, i64, i64x8, m64x8);
make_numeric_type!(DurationSecondType, i64, i64x8, m64x8);
make_numeric_type!(DurationMillisecondType, i64, i64x8, m64x8);
make_numeric_type!(DurationMicrosecondType, i64, i64x8, m64x8);
make_numeric_type!(DurationNanosecondType, i64, i64x8, m64x8);

// pub trait PolarsNumericType: ArrowNumericType {}
// impl PolarsNumericType for UInt8Type {}
// impl PolarsNumericType for UInt16Type {}
// impl PolarsNumericType for UInt32Type {}
// impl PolarsNumericType for UInt64Type {}
// impl PolarsNumericType for Int8Type {}
// impl PolarsNumericType for Int16Type {}
// impl PolarsNumericType for Int32Type {}
// impl PolarsNumericType for Int64Type {}
// impl PolarsNumericType for Float32Type {}
// impl PolarsNumericType for Float64Type {}
// impl PolarsNumericType for Date32Type {}
// impl PolarsNumericType for Date64Type {}
// impl PolarsNumericType for Time64NanosecondType {}
// impl PolarsNumericType for Time64MicrosecondType {}
// impl PolarsNumericType for Time32MillisecondType {}
// impl PolarsNumericType for Time32SecondType {}
// impl PolarsNumericType for DurationNanosecondType {}
// impl PolarsNumericType for DurationMicrosecondType {}
// impl PolarsNumericType for DurationMillisecondType {}
// impl PolarsNumericType for DurationSecondType {}
// impl PolarsNumericType for IntervalYearMonthType {}
// impl PolarsNumericType for IntervalDayTimeType {}
// impl PolarsNumericType for TimestampNanosecondType {}
// impl PolarsNumericType for TimestampMicrosecondType {}
// impl PolarsNumericType for TimestampMillisecondType {}
// impl PolarsNumericType for TimestampSecondType {}
//
pub trait PolarsIntegerType: PolarsNumericType {}
impl PolarsIntegerType for UInt8Type {}
impl PolarsIntegerType for UInt16Type {}
impl PolarsIntegerType for UInt32Type {}
impl PolarsIntegerType for UInt64Type {}
impl PolarsIntegerType for Int8Type {}
impl PolarsIntegerType for Int16Type {}
impl PolarsIntegerType for Int32Type {}
impl PolarsIntegerType for Int64Type {}
impl PolarsIntegerType for Date32Type {}
impl PolarsIntegerType for Date64Type {}
impl PolarsIntegerType for Time64NanosecondType {}
impl PolarsIntegerType for Time64MicrosecondType {}
impl PolarsIntegerType for Time32MillisecondType {}
impl PolarsIntegerType for Time32SecondType {}
impl PolarsIntegerType for DurationNanosecondType {}
impl PolarsIntegerType for DurationMicrosecondType {}
impl PolarsIntegerType for DurationMillisecondType {}
impl PolarsIntegerType for DurationSecondType {}
impl PolarsIntegerType for IntervalYearMonthType {}
impl PolarsIntegerType for IntervalDayTimeType {}
impl PolarsIntegerType for TimestampNanosecondType {}
impl PolarsIntegerType for TimestampMicrosecondType {}
impl PolarsIntegerType for TimestampMillisecondType {}
impl PolarsIntegerType for TimestampSecondType {}

#[derive(Debug)]
pub enum AnyType<'a> {
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
    Date32(i32),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in milliseconds (64 bits).
    Date64(i64),
    /// A 64-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    Time64(i64, TimeUnit),
    /// A 32-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    Time32(i32, TimeUnit),
    /// Measure of elapsed time in either seconds, milliseconds, microseconds or nanoseconds.
    Duration(i64, TimeUnit),
    /// Naive Time elapsed from the Unix epoch, 00:00:00.000 on 1 January 1970, excluding leap seconds, as a 64-bit integer.
    /// Note that UNIX time does not include leap seconds.
    TimeStamp(i64, TimeUnit),
    /// A "calendar" interval which models types that don't necessarily have a precise duration without the context of a base timestamp
    /// (e.g. days can differ in length during day light savings time transitions).
    IntervalDayTime(i64),
    IntervalYearMonth(i32),
    LargeList(Series),
}

pub trait ToStr {
    fn to_str(&self) -> String;
}

impl ToStr for ArrowDataType {
    fn to_str(&self) -> String {
        // TODO: add types here
        let s = match self {
            ArrowDataType::Null => "null",
            ArrowDataType::Boolean => "bool",
            ArrowDataType::UInt8 => "u8",
            ArrowDataType::UInt16 => "u16",
            ArrowDataType::UInt32 => "u32",
            ArrowDataType::UInt64 => "u64",
            ArrowDataType::Int8 => "i8",
            ArrowDataType::Int16 => "i16",
            ArrowDataType::Int32 => "i32",
            ArrowDataType::Int64 => "i64",
            ArrowDataType::Float32 => "f32",
            ArrowDataType::Float64 => "f64",
            ArrowDataType::Utf8 => "str",
            ArrowDataType::Date32(DateUnit::Day) => "date32(days)",
            ArrowDataType::Date64(DateUnit::Millisecond) => "date64(ms)",
            ArrowDataType::Time32(TimeUnit::Second) => "time64(s)",
            ArrowDataType::Time32(TimeUnit::Millisecond) => "time64(ms)",
            ArrowDataType::Time64(TimeUnit::Nanosecond) => "time64(ns)",
            ArrowDataType::Time64(TimeUnit::Microsecond) => "time64(μs)",
            // Note: Polars doesn't support the optional TimeZone in the timestamps.
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => "timestamp(ns)",
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => "timestamp(μs)",
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => "timestamp(ms)",
            ArrowDataType::Timestamp(TimeUnit::Second, _) => "timestamp(s)",
            ArrowDataType::Duration(TimeUnit::Nanosecond) => "duration(ns)",
            ArrowDataType::Duration(TimeUnit::Microsecond) => "duration(μs)",
            ArrowDataType::Duration(TimeUnit::Millisecond) => "duration(ms)",
            ArrowDataType::Duration(TimeUnit::Second) => "duration(s)",
            ArrowDataType::Interval(IntervalUnit::DayTime) => "interval(daytime)",
            ArrowDataType::Interval(IntervalUnit::YearMonth) => "interval(year-month)",
            ArrowDataType::LargeList(tp) => return format!("list [{}]", tp.to_str()),
            _ => panic!(format!("{:?} not implemented", self)),
        };
        s.into()
    }
}

impl<'a> PartialEq for AnyType<'a> {
    // Everything of Any is slow. Don't use.
    fn eq(&self, other: &Self) -> bool {
        format!("{}", self) == format!("{}", other)
    }
}
