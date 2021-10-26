use super::*;

pub struct BooleanChunkedBuilder {
    array_builder: MutableBooleanArray,
    field: Field,
}

impl ChunkedBuilder<bool, BooleanType> for BooleanChunkedBuilder {
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: bool) {
        self.array_builder.push(Some(v));
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.push(None);
    }

    fn finish(self) -> BooleanChunked {
        let arr: BooleanArray = self.array_builder.into();
        let arr = Arc::new(arr) as ArrayRef;

        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        }
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
