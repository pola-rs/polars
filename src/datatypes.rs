pub use arrow::datatypes::{
    BooleanType, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type,
};

use crate::series::chunked_array::ChunkedArray;
pub use arrow::datatypes::DataType as ArrowDataType;
use arrow::datatypes::{ArrowNumericType, ArrowPrimitiveType};
use std::ops::{Deref, DerefMut};

pub struct Utf8Type {
    data: String,
}

impl Deref for Utf8Type {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Utf8Type {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

pub trait PolarsDataType {
    fn get_data_type() -> ArrowDataType;
}

impl<T> PolarsDataType for T
where
    T: ArrowPrimitiveType,
{
    fn get_data_type() -> ArrowDataType {
        T::get_data_type()
    }
}

impl<'a> PolarsDataType for Utf8Type {
    fn get_data_type() -> ArrowDataType {
        ArrowDataType::Utf8
    }
}

pub type BooleanChunked = ChunkedArray<BooleanType>;
pub type UInt32Chunked = ChunkedArray<UInt32Type>;
pub type Int32Chunked = ChunkedArray<Int32Type>;
pub type Int64Chunked = ChunkedArray<Int64Type>;
pub type Float32Chunked = ChunkedArray<Float32Type>;
pub type Float64Chunked = ChunkedArray<Float64Type>;
pub type Utf8Chunked = ChunkedArray<Utf8Type>;

pub trait PolarNumericType: ArrowNumericType {}

impl PolarNumericType for Int32Type {}
impl PolarNumericType for Int64Type {}
impl PolarNumericType for Float32Type {}
impl PolarNumericType for Float64Type {}
impl PolarNumericType for UInt32Type {}
