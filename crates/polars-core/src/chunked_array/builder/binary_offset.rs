use super::*;

pub struct BinaryOffsetChunkedBuilder {
    pub(crate) chunk_builder: MutableBinaryArray<i64>,
    pub(crate) field: FieldRef,
}

impl Clone for BinaryOffsetChunkedBuilder {
    fn clone(&self) -> Self {
        Self {
            chunk_builder: self.chunk_builder.clone(),
            field: self.field.clone(),
        }
    }
}

impl BinaryOffsetChunkedBuilder {
    /// Create a new [`BinaryOffsetChunkedBuilder`]
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    pub fn new(name: PlSmallStr, capacity: usize) -> Self {
        Self {
            chunk_builder: MutableBinaryArray::with_capacity(capacity),
            field: Arc::new(Field::new(name, DataType::BinaryOffset)),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value(&mut self, v: &[u8]) {
        self.chunk_builder.push(Some(v));
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.chunk_builder.push_null()
    }

    #[inline]
    pub fn append_option(&mut self, opt: Option<&[u8]>) {
        self.chunk_builder.push(opt);
    }

    pub fn finish(mut self) -> BinaryOffsetChunked {
        let arr = self.chunk_builder.as_box();
        ChunkedArray::new_with_compute_len(self.field, vec![arr])
    }
}
