use arrow::array::StructArray;
use arrow::chunk::Chunk;
use arrow::datatypes::{DataType, Field};

use crate::prelude::*;

pub fn chunk_to_struct(chunk: Chunk<ArrayRef>, fields: Vec<Field>) -> StructArray {
    let dtype = DataType::Struct(fields);
    StructArray::from_data(dtype, chunk.into_arrays(), None)
}
