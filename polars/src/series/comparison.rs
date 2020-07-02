//! Comparison operations on Series.

use super::Series;
use crate::apply_method_arrowprimitive_series;
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
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, eq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, eq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, eq),
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, eq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, eq),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    /// Create a boolean mask by checking for inequality.
    fn neq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, neq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, neq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, neq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, neq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, neq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, neq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, neq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, neq),
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, neq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, neq),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    /// Create a boolean mask by checking if lhs > rhs.
    fn gt(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, gt),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, gt),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, gt),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, gt),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, gt),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, gt),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, gt),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, gt),
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, gt),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, gt),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    /// Create a boolean mask by checking if lhs >= rhs.
    fn gt_eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, gt_eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, gt_eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, gt_eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, gt_eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, gt_eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, gt_eq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, gt_eq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, gt_eq),
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, gt_eq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, gt_eq),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    /// Create a boolean mask by checking if lhs < rhs.
    fn lt(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, lt),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, lt),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, lt),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, lt),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, lt),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, lt),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, lt),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, lt),
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, lt),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, lt),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    /// Create a boolean mask by checking if lhs <= rhs.
    fn lt_eq(&self, rhs: &Series) -> BooleanChunked {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, lt_eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, lt_eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, lt_eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, lt_eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, lt_eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, lt_eq),
            Series::Date32(a) => compare!(Series::Date32, a, rhs, lt_eq),
            Series::Date64(a) => compare!(Series::Date64, a, rhs, lt_eq),
            Series::Time64Ns(a) => compare!(Series::Time64Ns, a, rhs, lt_eq),
            Series::DurationNs(a) => compare!(Series::DurationNs, a, rhs, lt_eq),
            Series::Bool(_a) => unimplemented!(),
        }
    }
}

impl<Rhs> CmpOps<Rhs> for Series
where
    Rhs: NumComp,
{
    fn eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_arrowprimitive_series!(self, eq, rhs)
    }

    fn neq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_arrowprimitive_series!(self, neq, rhs)
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_arrowprimitive_series!(self, gt, rhs)
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_arrowprimitive_series!(self, gt_eq, rhs)
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_arrowprimitive_series!(self, lt, rhs)
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        apply_method_arrowprimitive_series!(self, lt_eq, rhs)
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
