pub use arrow::datatypes::{
    BooleanType, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type,
};

use arrow::datatypes::ArrowPrimitiveType;
pub use arrow::datatypes::DataType as ArrowDataType;
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
