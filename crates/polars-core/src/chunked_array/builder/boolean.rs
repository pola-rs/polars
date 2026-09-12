use polars_array::PlBooleanArrayBuilder;

use super::*;

#[derive(Clone)]
pub struct BooleanChunkedBuilder {
    pub(crate) array_builder: PlBooleanArrayBuilder,
    pub(crate) field: Field,
}

impl ChunkedBuilder<bool, BooleanType> for BooleanChunkedBuilder {
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: bool) {
        self.array_builder.push_value(v);
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.push_null();
    }

    fn finish(self) -> BooleanChunked {
        let arr = self.array_builder.freeze().into_boxed();
        ChunkedArray::new_with_compute_len(Arc::new(self.field), vec![arr])
    }
}

impl BooleanChunkedBuilder {
    pub fn new(name: PlSmallStr, capacity: usize) -> Self {
        BooleanChunkedBuilder {
            array_builder: PlBooleanArrayBuilder::with_capacity(capacity),
            field: Field::new(name, DataType::Boolean),
        }
    }
}
