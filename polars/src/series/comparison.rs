//! Comparison operations on Series.

use super::Series;
use crate::apply_method_numeric_series;
use crate::prelude::*;

macro_rules! compare {
    ($variant:path, $lhs:ident, $rhs:ident, $cmp_method:ident) => {{
        if let $variant(rhs_) = $rhs {
            $lhs.$cmp_method(rhs_)
        } else {
            std::iter::repeat(false).take($lhs.len()).collect()
        }
    }};
}

impl CmpOps<&Series> for Series {
    /// Create a boolean mask by checking for equality.
    fn eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
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
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, eq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, eq),
            Series::Bool(a) => compare!(Series::Bool, a, rhs, eq),
        }
    }

    /// Create a boolean mask by checking for inequality.
    fn neq(&self, rhs: &Series) -> BooleanChunked {
        match self {
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
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, neq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, neq),
            Series::Bool(a) => compare!(Series::Bool, a, rhs, neq),
        }
    }

    /// Create a boolean mask by checking if lhs > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        match self {
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
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, gt),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, gt),
            Series::Bool(a) => compare!(Series::Bool, a, rhs, gt),
        }
    }

    /// Create a boolean mask by checking if lhs >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
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
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, gt_eq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, gt_eq),
            Series::Bool(a) => compare!(Series::Bool, a, rhs, gt_eq),
        }
    }

    /// Create a boolean mask by checking if lhs < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        match self {
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
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, lt),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, lt),
            Series::Bool(a) => compare!(Series::Bool, a, rhs, lt),
        }
    }

    /// Create a boolean mask by checking if lhs <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
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
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, lt_eq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, lt_eq),
            Series::Bool(a) => compare!(Series::Bool, a, rhs, lt_eq),
        }
    }
}

impl<Rhs> CmpOps<Rhs> for Series
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

impl CmpOps<&str> for Series {
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
