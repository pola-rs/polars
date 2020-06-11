use super::series::Series;
use crate::{
    datatypes,
    datatypes::BooleanChunked,
    error::{PolarsError, Result},
    series::chunked_array::{comparison::CmpOps, ChunkedArray},
};
use num::traits::{Num, NumCast};

macro_rules! compare {
    ($variant:path, $lhs:ident, $rhs:ident, $cmp_method:ident) => {{
        if let $variant(rhs_) = $rhs {
            Ok($lhs.$cmp_method(rhs_)?)
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }};
}

impl CmpOps<&Series> for Series {
    fn eq(&self, rhs: &Series) -> Result<BooleanChunked> {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, eq),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    fn neq(&self, rhs: &Series) -> Result<BooleanChunked> {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, neq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, neq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, neq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, neq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, neq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, neq),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    fn gt(&self, rhs: &Series) -> Result<BooleanChunked> {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, gt),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, gt),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, gt),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, gt),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, gt),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, gt),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    fn gt_eq(&self, rhs: &Series) -> Result<BooleanChunked> {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, gt_eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, gt_eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, gt_eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, gt_eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, gt_eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, gt_eq),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    fn lt(&self, rhs: &Series) -> Result<BooleanChunked> {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, lt),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, lt),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, lt),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, lt),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, lt),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, lt),
            Series::Bool(_a) => unimplemented!(),
        }
    }

    fn lt_eq(&self, rhs: &Series) -> Result<BooleanChunked> {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, lt_eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, lt_eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, lt_eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, lt_eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, lt_eq),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, lt_eq),
            Series::Bool(_a) => unimplemented!(),
        }
    }
}
