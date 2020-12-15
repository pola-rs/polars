//! Comparison operations on Series.

use super::Series;
use crate::apply_method_numeric_series;
use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;

fn fill_bool(val: bool, len: usize) -> BooleanChunked {
    std::iter::repeat(val).take(len).collect()
}

macro_rules! compare {
    ($variant:path, $lhs:expr, $rhs:expr, $cmp_method:ident) => {{
        if let $variant(rhs_) = $rhs {
            $lhs.$cmp_method(rhs_)
        } else {
            fill_bool(false, $lhs.len())
        }
    }};
}

macro_rules! impl_compare {
    ($self:expr, $rhs:expr, $method:ident) => {{
        match $self {
            Series::Bool(a) => compare!(Series::Bool, a, $rhs, $method),
            Series::UInt8(a) => compare!(Series::UInt8, a, $rhs, $method),
            Series::UInt16(a) => compare!(Series::UInt16, a, $rhs, $method),
            Series::UInt32(a) => compare!(Series::UInt32, a, $rhs, $method),
            Series::UInt64(a) => compare!(Series::UInt64, a, $rhs, $method),
            Series::Int8(a) => compare!(Series::Int8, a, $rhs, $method),
            Series::Int16(a) => compare!(Series::Int16, a, $rhs, $method),
            Series::Int32(a) => compare!(Series::Int32, a, $rhs, $method),
            Series::Int64(a) => compare!(Series::Int64, a, $rhs, $method),
            Series::Float32(a) => compare!(Series::Float32, a, $rhs, $method),
            Series::Float64(a) => compare!(Series::Float64, a, $rhs, $method),
            Series::Utf8(a) => compare!(Series::Utf8, a, $rhs, $method),
            Series::Date32(a) => compare!(Series::Date32, a, $rhs, $method),
            Series::Date64(a) => compare!(Series::Date64, a, $rhs, $method),
            Series::Time32Millisecond(a) => compare!(Series::Time32Millisecond, a, $rhs, $method),
            Series::Time32Second(a) => compare!(Series::Time32Second, a, $rhs, $method),
            Series::Time64Nanosecond(a) => compare!(Series::Time64Nanosecond, a, $rhs, $method),
            Series::Time64Microsecond(a) => compare!(Series::Time64Microsecond, a, $rhs, $method),
            Series::DurationNanosecond(a) => compare!(Series::DurationNanosecond, a, $rhs, $method),
            Series::DurationMicrosecond(a) => {
                compare!(Series::DurationMicrosecond, a, $rhs, $method)
            }
            Series::DurationMillisecond(a) => {
                compare!(Series::DurationMillisecond, a, $rhs, $method)
            }
            Series::DurationSecond(a) => compare!(Series::DurationSecond, a, $rhs, $method),
            Series::TimestampNanosecond(a) => {
                compare!(Series::TimestampNanosecond, a, $rhs, $method)
            }
            Series::TimestampMicrosecond(a) => {
                compare!(Series::TimestampMicrosecond, a, $rhs, $method)
            }
            Series::TimestampMillisecond(a) => {
                compare!(Series::TimestampMillisecond, a, $rhs, $method)
            }
            Series::TimestampSecond(a) => compare!(Series::TimestampSecond, a, $rhs, $method),
            #[cfg(feature = "dtype-interval")]
            Series::IntervalDayTime(a) => compare!(Series::IntervalDayTime, a, $rhs, $method),
            #[cfg(feature = "dtype-interval")]
            Series::IntervalYearMonth(a) => compare!(Series::IntervalYearMonth, a, $rhs, $method),
            Series::List(a) => compare!(Series::List, a, $rhs, $method),
            Series::Object(_) => fill_bool(false, $self.len()),
        }
    }};
}

impl ChunkCompare<&Series> for Series {
    fn eq_missing(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), eq_missing)
    }

    /// Create a boolean mask by checking for equality.
    fn eq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), eq)
    }

    /// Create a boolean mask by checking for inequality.
    fn neq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), neq)
    }

    /// Create a boolean mask by checking if lhs > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), gt)
    }

    /// Create a boolean mask by checking if lhs >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), gt_eq)
    }

    /// Create a boolean mask by checking if lhs < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), lt)
    }

    /// Create a boolean mask by checking if lhs <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        impl_compare!(lhs.as_ref(), rhs.as_ref(), lt_eq)
    }
}

impl<Rhs> ChunkCompare<Rhs> for Series
where
    Rhs: NumComp,
{
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.eq(rhs)
    }

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
    fn eq_missing(&self, rhs: &str) -> BooleanChunked {
        self.eq(rhs)
    }

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
