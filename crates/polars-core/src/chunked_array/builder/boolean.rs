use super::*;

#[derive(Clone)]
pub struct BooleanChunkedBuilder {
    pub(crate) array_builder: MutableBooleanArray,
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

    fn finish(mut self) -> BooleanChunked {
        let arr = self.array_builder.as_box();
        ChunkedArray::new_with_compute_len(Arc::new(self.field), vec![arr])
    }

    fn shrink_to_fit(&mut self) {
        self.array_builder.shrink_to_fit()
    }
}

impl BooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        BooleanChunkedBuilder {
            array_builder: MutableBooleanArray::with_capacity(capacity),
            field: Field::new(name, DataType::Boolean),
        }
    }
}
