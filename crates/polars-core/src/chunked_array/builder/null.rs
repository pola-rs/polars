use polars_array::PlNullArrayBuilder;
use polars_array::builder::StaticArrayBuilder;

use super::*;
use crate::series::implementations::null::NullChunked;

#[derive(Clone)]
pub struct NullChunkedBuilder {
    array_builder: PlNullArrayBuilder,
    pub(crate) field: Field,
}

impl NullChunkedBuilder {
    pub fn new(name: PlSmallStr, len: usize) -> Self {
        let mut array_builder = PlNullArrayBuilder::new();
        array_builder.extend_nulls(len);

        NullChunkedBuilder {
            array_builder,
            field: Field::new(name, DataType::Null),
        }
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.array_builder.push_null()
    }

    pub fn finish(self) -> NullChunked {
        NullChunked::new(self.field.name().clone(), self.array_builder.len())
    }

    /// Does nothing: a null builder holds a length and no allocation to give back.
    pub fn shrink_to_fit(&mut self) {}
}
