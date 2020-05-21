use super::chunked_array::ChunkedArray;
use crate::{datatypes, error::Result};
use arrow::array::ArrayRef;
use arrow::datatypes::{ArrowPrimitiveType, DataType};

pub enum Series {
    Int32(ChunkedArray<datatypes::Int32Type>),
    Int64(ChunkedArray<datatypes::Int64Type>),
    Float32(ChunkedArray<datatypes::Float32Type>),
    Float64(ChunkedArray<datatypes::Float64Type>),
    Utf8(ChunkedArray<datatypes::Utf8Type>),
}

impl Series {
    pub fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        match self {
            Series::Int32(a) => a.append_array(other),
            Series::Int64(a) => a.append_array(other),
            Series::Float32(a) => a.append_array(other),
            Series::Float64(a) => a.append_array(other),
            Series::Utf8(a) => a.append_array(other),
            _ => unimplemented!(),
        }
    }
}
