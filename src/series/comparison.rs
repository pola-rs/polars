use super::series::Series;
use crate::{
    datatypes,
    error::{PolarsError, Result},
    series::chunked_array::{comparison::CmpOpsChunkedArray, ChunkedArray},
};

type CompareResult = Result<ChunkedArray<datatypes::BooleanType>>;

macro_rules! compare {
    ($variant:path, $lhs:ident, $rhs:ident, $cmp_method:ident) => {{
        if let $variant(rhs_) = $rhs {
            Ok($lhs.$cmp_method(&rhs_))
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }};
}

pub trait CmpOps<T> {
    fn eq(&self, rhs: T) -> CompareResult;
}

impl CmpOps<Series> for Series {
    fn eq(&self, rhs: Series) -> CompareResult {
        match self {
            Series::UInt32(a) => compare!(Series::UInt32, a, rhs, eq),
            Series::Int32(a) => compare!(Series::Int32, a, rhs, eq),
            Series::Int64(a) => compare!(Series::Int64, a, rhs, eq),
            Series::Float32(a) => compare!(Series::Float32, a, rhs, eq),
            Series::Float64(a) => compare!(Series::Float64, a, rhs, eq),
            Series::Bool(_a) => unimplemented!(),
            Series::Utf8(a) => compare!(Series::Utf8, a, rhs, eq),
        };
        unimplemented!()
    }
}
