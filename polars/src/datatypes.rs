//! # Data types supported by Polars.
//!
//! At the moment Polars doesn't include all data types available by Arrow. The goal is to
//! incrementally support more data types and prioritize these by usability.
//!
//! [See the AnyType variants](enum.AnyType.html#variants) for the data types that
//! are currently supported.
//!
use crate::prelude::*;
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

pub struct Utf8Type {}

pub struct ListType {}

pub trait PolarsDataType: Send + Sync {
    fn get_dtype() -> ArrowDataType;
}

impl<T> PolarsDataType for T
where
    T: PolarsPrimitiveType + Sync + Send,
{
    fn get_dtype() -> ArrowDataType {
        T::get_data_type()
    }
}

impl PolarsDataType for Utf8Type {
    fn get_dtype() -> ArrowDataType {
        ArrowDataType::Utf8
    }
}

impl PolarsDataType for ListType {
    fn get_dtype() -> ArrowDataType {
        // null as we cannot no anything without self.
        ArrowDataType::List(Box::new(ArrowDataType::Null))
    }
}

#[cfg(feature = "object")]
#[doc(cfg(feature = "object"))]
pub struct ObjectType<T>(T);
#[cfg(feature = "object")]
pub type ObjectChunked<T> = ChunkedArray<ObjectType<T>>;

#[cfg(feature = "object")]
#[doc(cfg(feature = "object"))]
impl<T: Send + Sync> PolarsDataType for ObjectType<T> {
    fn get_dtype() -> ArrowDataType {
        // the best fit?
        ArrowDataType::Binary
    }
}

/// Any type that is not nested
pub trait PolarsSingleType: PolarsDataType {}

impl<T> PolarsSingleType for T where T: ArrowPrimitiveType + PolarsDataType {}

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
pub type Date32Chunked = ChunkedArray<Date32Type>;
pub type Date64Chunked = ChunkedArray<Date64Type>;
pub type DurationNanosecondChunked = ChunkedArray<DurationNanosecondType>;
pub type DurationMillisecondChunked = ChunkedArray<DurationMillisecondType>;
pub type Time64NanosecondChunked = ChunkedArray<Time64NanosecondType>;

pub trait PolarsPrimitiveType: ArrowPrimitiveType + Send + Sync {}
impl PolarsPrimitiveType for BooleanType {}
impl PolarsPrimitiveType for UInt8Type {}
impl PolarsPrimitiveType for UInt16Type {}
impl PolarsPrimitiveType for UInt32Type {}
impl PolarsPrimitiveType for UInt64Type {}
impl PolarsPrimitiveType for Int8Type {}
impl PolarsPrimitiveType for Int16Type {}
impl PolarsPrimitiveType for Int32Type {}
impl PolarsPrimitiveType for Int64Type {}
impl PolarsPrimitiveType for Float32Type {}
impl PolarsPrimitiveType for Float64Type {}
impl PolarsPrimitiveType for Date32Type {}
impl PolarsPrimitiveType for Date64Type {}
impl PolarsPrimitiveType for Time64NanosecondType {}
impl PolarsPrimitiveType for DurationNanosecondType {}
impl PolarsPrimitiveType for DurationMillisecondType {}

pub trait PolarsNumericType: PolarsPrimitiveType + ArrowNumericType {}
impl PolarsNumericType for UInt8Type {}
impl PolarsNumericType for UInt16Type {}
impl PolarsNumericType for UInt32Type {}
impl PolarsNumericType for UInt64Type {}
impl PolarsNumericType for Int8Type {}
impl PolarsNumericType for Int16Type {}
impl PolarsNumericType for Int32Type {}
impl PolarsNumericType for Int64Type {}
impl PolarsNumericType for Float32Type {}
impl PolarsNumericType for Float64Type {}
impl PolarsNumericType for Date32Type {}
impl PolarsNumericType for Date64Type {}
impl PolarsNumericType for Time64NanosecondType {}
impl PolarsNumericType for DurationNanosecondType {}
impl PolarsNumericType for DurationMillisecondType {}

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
impl PolarsIntegerType for DurationNanosecondType {}
impl PolarsIntegerType for DurationMillisecondType {}

pub trait PolarsFloatType: PolarsNumericType {}
impl PolarsFloatType for Float32Type {}
impl PolarsFloatType for Float64Type {}

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
    Duration(i64, TimeUnit),
    /// Naive Time elapsed from the Unix epoch, 00:00:00.000 on 1 January 1970, excluding leap seconds, as a 64-bit integer.
    /// Note that UNIX time does not include leap seconds.
    List(Series),
    #[cfg(feature = "object")]
    /// Use as_any to get a dyn Any
    Object(&'a str),
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
            ArrowDataType::List(tp) => return format!("list [{}]", tp.to_str()),
            #[cfg(feature = "object")]
            ArrowDataType::Binary => "object",
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
