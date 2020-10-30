use polars::datatypes::{ArrowDataType, IntervalUnit, TimeUnit};

// Don't change the order of these!
#[repr(u8)]
pub enum DataType {
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
    Bool,
    Utf8,
    List,
    Date32,
    Date64,
    Time32Millisecond,
    Time32Second,
    Time64Nanosecond,
    Time64Microsecond,
    DurationNanosecond,
    DurationMicrosecond,
    DurationMillisecond,
    DurationSecond,
    IntervalDayTime,
    IntervalYearMonth,
    TimestampNanosecond,
    TimestampMicrosecond,
    TimestampMillisecond,
    TimestampSecond,
}

impl From<&ArrowDataType> for DataType {
    fn from(dt: &ArrowDataType) -> Self {
        use DataType::*;
        match dt {
            ArrowDataType::Int8 => Int8,
            ArrowDataType::Int16 => Int16,
            ArrowDataType::Int32 => Int32,
            ArrowDataType::Int64 => Int64,
            ArrowDataType::UInt8 => UInt8,
            ArrowDataType::UInt16 => UInt16,
            ArrowDataType::UInt32 => UInt32,
            ArrowDataType::UInt64 => UInt64,
            ArrowDataType::Float32 => Float32,
            ArrowDataType::Float64 => Float64,
            ArrowDataType::Boolean => Bool,
            ArrowDataType::Utf8 => Utf8,
            ArrowDataType::List(_) => List,
            ArrowDataType::Date32(_) => Date32,
            ArrowDataType::Date64(_) => Date64,
            ArrowDataType::Time32(TimeUnit::Millisecond) => Time32Millisecond,
            ArrowDataType::Time32(TimeUnit::Second) => Time32Second,
            ArrowDataType::Time64(TimeUnit::Nanosecond) => Time64Nanosecond,
            ArrowDataType::Time64(TimeUnit::Microsecond) => Time64Microsecond,
            ArrowDataType::Interval(IntervalUnit::DayTime) => IntervalDayTime,
            ArrowDataType::Interval(IntervalUnit::YearMonth) => IntervalYearMonth,
            ArrowDataType::Duration(TimeUnit::Nanosecond) => DurationNanosecond,
            ArrowDataType::Duration(TimeUnit::Microsecond) => DurationMicrosecond,
            ArrowDataType::Duration(TimeUnit::Millisecond) => DurationMillisecond,
            ArrowDataType::Duration(TimeUnit::Second) => DurationSecond,
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => TimestampNanosecond,
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecond,
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => TimestampMillisecond,
            ArrowDataType::Timestamp(TimeUnit::Second, _) => TimestampSecond,
            dt => panic!(format!("datatype: {:?} not supported", dt)),
        }
    }
}
