use super::*;

pub struct BinViewChunkedBuilder<T: ViewType + ?Sized> {
    pub(crate) chunk_builder: MutableBinaryViewArray<T>,
    pub(crate) field: FieldRef,
}

impl<T: ViewType + ?Sized> Clone for BinViewChunkedBuilder<T> {
    fn clone(&self) -> Self {
        Self {
            chunk_builder: self.chunk_builder.clone(),
            field: self.field.clone(),
        }
    }
}

pub type StringChunkedBuilder = BinViewChunkedBuilder<str>;
pub type BinaryChunkedBuilder = BinViewChunkedBuilder<[u8]>;

impl<T: ViewType + ?Sized> BinViewChunkedBuilder<T> {
    /// Create a new StringChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    /// * `bytes_capacity` - Number of bytes needed to store the string values.
    pub fn new(name: &str, capacity: usize) -> Self {
        Self {
            chunk_builder: MutableBinaryViewArray::with_capacity(capacity),
            field: Arc::new(Field::new(name, DataType::from(&T::DATA_TYPE))),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value<S: AsRef<T>>(&mut self, v: S) {
        self.chunk_builder.push_value(v.as_ref());
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.chunk_builder.push_null()
    }

    #[inline]
    pub fn append_option<S: AsRef<T>>(&mut self, opt: Option<S>) {
        self.chunk_builder.push(opt);
    }
}

impl StringChunkedBuilder {
    pub fn finish(mut self) -> StringChunked {
        let arr = self.chunk_builder.as_box();
        ChunkedArray::new_with_compute_len(self.field, vec![arr])
    }
}
impl BinaryChunkedBuilder {
    pub fn finish(mut self) -> BinaryChunked {
        let arr = self.chunk_builder.as_box();
        ChunkedArray::new_with_compute_len(self.field, vec![arr])
    }
}
