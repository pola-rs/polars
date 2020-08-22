//! Comparison operations on Series.

use super::Series;
use crate::apply_method_numeric_series;
use crate::prelude::*;

fn fill_bool(val: bool, len: usize) -> BooleanChunked {
    std::iter::repeat(val).take(len).collect()
}

macro_rules! compare {
    ($variant:path, $lhs:ident, $rhs:ident, $cmp_method:ident) => {{
        if let $variant(rhs_) = $rhs {
            $lhs.$cmp_method(rhs_)
        } else {
            fill_bool(false, $lhs.len())
        }
    }};
}

impl ChunkCompare<&Series> for Series {
    /// Create a boolean mask by checking for equality.
    fn eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::Bool(a) => compare!(Series::Bool, a, rhs, eq),
            Series::UInt8(a) => compare!(Series::UInt8, a, rhs, eq),
            Series::UInt16(a) => compare!(Series::UInt16, a, rhs, eq),
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, eq),
            Series::UInt64(a) => compare!(Series::UInt64, a, rhs, eq),
            Series::Int8(a) => compare!(Series::Int8, a, rhs, eq),
            Series::Int16(a) => compare!(Series::Int16, a, rhs, eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, eq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, eq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, eq),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, rhs, eq),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, rhs, eq),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, rhs, eq),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, rhs, eq),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, rhs, eq),
            Series::DurationMicrosecond(a) => compare!(Series::DurationMicrosecond, a, rhs, eq),
            Series::DurationMillisecond(a) => compare!(Series::DurationMillisecond, a, rhs, eq),
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, rhs, eq),
            Series::TimestampNanosecond(a) => compare!(Series::TimestampNanosecond, a, rhs, eq),
            Series::TimestampMicrosecond(a) => compare!(Series::TimestampMicrosecond, a, rhs, eq),
            Series::TimestampMillisecond(a) => compare!(Series::TimestampMillisecond, a, rhs, eq),
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, rhs, eq),
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, rhs, eq),
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, rhs, eq),
            Series::LargeList(_a) => fill_bool(true, self.len()),
        }
    }

    /// Create a boolean mask by checking for inequality.
    fn neq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::Bool(a) => compare!(Series::Bool, a, rhs, neq),
            Series::UInt8(a) => compare!(Series::UInt8, a, rhs, neq),
            Series::UInt16(a) => compare!(Series::UInt16, a, rhs, neq),
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, neq),
            Series::UInt64(a) => compare!(Series::UInt64, a, rhs, neq),
            Series::Int8(a) => compare!(Series::Int8, a, rhs, neq),
            Series::Int16(a) => compare!(Series::Int16, a, rhs, neq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, neq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, neq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, neq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, neq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, neq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, neq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, neq),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, rhs, neq),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, rhs, neq),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, rhs, neq),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, rhs, neq),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, rhs, neq),
            Series::DurationMicrosecond(a) => compare!(Series::DurationMicrosecond, a, rhs, neq),
            Series::DurationMillisecond(a) => compare!(Series::DurationMillisecond, a, rhs, neq),
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, rhs, neq),
            Series::TimestampNanosecond(a) => compare!(Series::TimestampNanosecond, a, rhs, neq),
            Series::TimestampMicrosecond(a) => compare!(Series::TimestampMicrosecond, a, rhs, neq),
            Series::TimestampMillisecond(a) => compare!(Series::TimestampMillisecond, a, rhs, neq),
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, rhs, neq),
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, rhs, neq),
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, rhs, neq),
            Series::LargeList(_a) => fill_bool(true, self.len()),
        }
    }

    /// Create a boolean mask by checking if lhs > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::Bool(a) => compare!(Series::Bool, a, rhs, gt),
            Series::UInt8(a) => compare!(Series::UInt8, a, rhs, gt),
            Series::UInt16(a) => compare!(Series::UInt16, a, rhs, gt),
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, gt),
            Series::UInt64(a) => compare!(Series::UInt64, a, rhs, gt),
            Series::Int8(a) => compare!(Series::Int8, a, rhs, gt),
            Series::Int16(a) => compare!(Series::Int16, a, rhs, gt),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, gt),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, gt),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, gt),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, gt),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, gt),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, gt),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, gt),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, rhs, gt),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, rhs, gt),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, rhs, gt),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, rhs, gt),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, rhs, gt),
            Series::DurationMicrosecond(a) => compare!(Series::DurationMicrosecond, a, rhs, gt),
            Series::DurationMillisecond(a) => compare!(Series::DurationMillisecond, a, rhs, gt),
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, rhs, gt),
            Series::TimestampNanosecond(a) => compare!(Series::TimestampNanosecond, a, rhs, gt),
            Series::TimestampMicrosecond(a) => compare!(Series::TimestampMicrosecond, a, rhs, gt),
            Series::TimestampMillisecond(a) => compare!(Series::TimestampMillisecond, a, rhs, gt),
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, rhs, gt),
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, rhs, gt),
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, rhs, gt),
            Series::LargeList(_a) => fill_bool(true, self.len()),
        }
    }

    /// Create a boolean mask by checking if lhs >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::Bool(a) => compare!(Series::Bool, a, rhs, gt_eq),
            Series::UInt8(a) => compare!(Series::UInt8, a, rhs, gt_eq),
            Series::UInt16(a) => compare!(Series::UInt16, a, rhs, gt_eq),
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, gt_eq),
            Series::UInt64(a) => compare!(Series::UInt64, a, rhs, gt_eq),
            Series::Int8(a) => compare!(Series::Int8, a, rhs, gt_eq),
            Series::Int16(a) => compare!(Series::Int16, a, rhs, gt_eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, gt_eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, gt_eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, gt_eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, gt_eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, gt_eq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, gt_eq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, gt_eq),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, rhs, gt_eq),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, rhs, gt_eq),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, rhs, gt_eq),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, rhs, gt_eq),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, rhs, gt_eq),
            Series::DurationMicrosecond(a) => compare!(Series::DurationMicrosecond, a, rhs, gt_eq),
            Series::DurationMillisecond(a) => compare!(Series::DurationMillisecond, a, rhs, gt_eq),
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, rhs, gt_eq),
            Series::TimestampNanosecond(a) => compare!(Series::TimestampNanosecond, a, rhs, gt_eq),
            Series::TimestampMicrosecond(a) => {
                compare!(Series::TimestampMicrosecond, a, rhs, gt_eq)
            }
            Series::TimestampMillisecond(a) => {
                compare!(Series::TimestampMillisecond, a, rhs, gt_eq)
            }
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, rhs, gt_eq),
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, rhs, gt_eq),
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, rhs, gt_eq),
            Series::LargeList(_a) => fill_bool(true, self.len()),
        }
    }

    /// Create a boolean mask by checking if lhs < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::Bool(a) => compare!(Series::Bool, a, rhs, lt),
            Series::UInt8(a) => compare!(Series::UInt8, a, rhs, lt),
            Series::UInt16(a) => compare!(Series::UInt16, a, rhs, lt),
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, lt),
            Series::UInt64(a) => compare!(Series::UInt64, a, rhs, lt),
            Series::Int8(a) => compare!(Series::Int8, a, rhs, lt),
            Series::Int16(a) => compare!(Series::Int16, a, rhs, lt),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, lt),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, lt),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, lt),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, lt),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, lt),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, lt),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, lt),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, rhs, lt),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, rhs, lt),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, rhs, lt),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, rhs, lt),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, rhs, lt),
            Series::DurationMicrosecond(a) => compare!(Series::DurationMicrosecond, a, rhs, lt),
            Series::DurationMillisecond(a) => compare!(Series::DurationMillisecond, a, rhs, lt),
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, rhs, lt),
            Series::TimestampNanosecond(a) => compare!(Series::TimestampNanosecond, a, rhs, lt),
            Series::TimestampMicrosecond(a) => compare!(Series::TimestampMicrosecond, a, rhs, lt),
            Series::TimestampMillisecond(a) => compare!(Series::TimestampMillisecond, a, rhs, lt),
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, rhs, lt),
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, rhs, lt),
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, rhs, lt),
            Series::LargeList(_a) => fill_bool(true, self.len()),
        }
    }

    /// Create a boolean mask by checking if lhs <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::Bool(a) => compare!(Series::Bool, a, rhs, lt_eq),
            Series::UInt8(a) => compare!(Series::UInt8, a, rhs, lt_eq),
            Series::UInt16(a) => compare!(Series::UInt16, a, rhs, lt_eq),
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, lt_eq),
            Series::UInt64(a) => compare!(Series::UInt64, a, rhs, lt_eq),
            Series::Int8(a) => compare!(Series::Int8, a, rhs, lt_eq),
            Series::Int16(a) => compare!(Series::Int16, a, rhs, lt_eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, lt_eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, lt_eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, lt_eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, lt_eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, lt_eq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, lt_eq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, lt_eq),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, rhs, lt_eq),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, rhs, lt_eq),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, rhs, lt_eq),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, rhs, lt_eq),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, rhs, lt_eq),
            Series::DurationMicrosecond(a) => compare!(Series::DurationMicrosecond, a, rhs, lt_eq),
            Series::DurationMillisecond(a) => compare!(Series::DurationMillisecond, a, rhs, lt_eq),
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, rhs, lt_eq),
            Series::TimestampNanosecond(a) => compare!(Series::TimestampNanosecond, a, rhs, lt_eq),
            Series::TimestampMicrosecond(a) => {
                compare!(Series::TimestampMicrosecond, a, rhs, lt_eq)
            }
            Series::TimestampMillisecond(a) => {
                compare!(Series::TimestampMillisecond, a, rhs, lt_eq)
            }
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, rhs, lt_eq),
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, rhs, lt_eq),
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, rhs, lt_eq),
            Series::LargeList(_a) => fill_bool(true, self.len()),
        }
    }
}

impl<Rhs> ChunkCompare<Rhs> for Series
where
    Rhs: NumComp,
{
    fn eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, eq, rhs)
    }

    fn neq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, neq, rhs)
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, gt, rhs)
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, gt_eq, rhs)
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, lt, rhs)
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_numeric_series!(self, lt_eq, rhs)
    }
}

impl ChunkCompare<&str> for Series {
    fn eq(&self, rhs: &str) -> BooleanChunked {
        match self {
            Series::Utf8(a) => a.eq(rhs),
            _ => std::iter::repeat(false).take(self.len()).collect(),
        }
    }

    fn neq(&self, rhs: &str) -> BooleanChunked {
        match self {
            Series::Utf8(a) => a.neq(rhs),
            _ => std::iter::repeat(false).take(self.len()).collect(),
        }
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        match self {
            Series::Utf8(a) => a.gt(rhs),
            _ => std::iter::repeat(false).take(self.len()).collect(),
        }
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        match self {
            Series::Utf8(a) => a.gt_eq(rhs),
            _ => std::iter::repeat(false).take(self.len()).collect(),
        }
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        match self {
            Series::Utf8(a) => a.lt(rhs),
            _ => std::iter::repeat(false).take(self.len()).collect(),
        }
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        match self {
            Series::Utf8(a) => a.lt_eq(rhs),
            _ => std::iter::repeat(false).take(self.len()).collect(),
        }
    }
}
