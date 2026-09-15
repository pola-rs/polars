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
        // `len` is how many nulls the builder starts out holding, not room for that many: a null
        // array is nothing but its length, so there is no allocation to reserve.
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
        // A null array holds no values, so the length the builder counted is the whole of it.
        NullChunked::new(self.field.name().clone(), self.array_builder.len())
    }

    /// Does nothing: a null builder holds a length and no allocation to give back.
    pub fn shrink_to_fit(&mut self) {}
}
