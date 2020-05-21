use super::primitive::ChunkedArray;
use crate::error::Result;
use arrow::array::ArrayRef;
use arrow::datatypes::Field;
use arrow::{
    datatypes,
    datatypes::{ArrowPrimitiveType, DataType},
};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub enum Series {
    Int32(ChunkedArray<datatypes::Int32Type>),
    Int64(ChunkedArray<datatypes::Int64Type>),
    Float32(ChunkedArray<datatypes::Float32Type>),
    Float64(ChunkedArray<datatypes::Float64Type>),
}

impl Series {
    pub fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        match self {
            Series::Int32(a) => a.append_array(other),
            Series::Int64(a) => a.append_array(other),
            Series::Float32(a) => a.append_array(other),
            Series::Float64(a) => a.append_array(other),
        }
    }
}
