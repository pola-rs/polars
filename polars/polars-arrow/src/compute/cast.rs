use arrow::array::Array;
use arrow::datatypes::DataType;
use arrow::error::Result;

pub fn cast(array: &dyn Array, to_type: &DataType) -> Result<Box<dyn Array>> {
    arrow::compute::cast::cast(array, to_type, Default::default())
}
