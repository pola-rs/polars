use super::*;

pub struct BinaryChunkedBuilder {
    pub(crate) builder: MutableBinaryArray<i64>,
    pub capacity: usize,
    field: Field,
}

impl BinaryChunkedBuilder {
    /// Create a new UtfChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    /// * `bytes_capacity` - Number of bytes needed to store the string values.
    pub fn new(name: &str, capacity: usize, bytes_capacity: usize) -> Self {
        BinaryChunkedBuilder {
            builder: MutableBinaryArray::<i64>::with_capacities(capacity, bytes_capacity),
            capacity,
            field: Field::new(name, DataType::Binary),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value<S: AsRef<[u8]>>(&mut self, v: S) {
        self.builder.push(Some(v.as_ref()));
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.builder.push::<&[u8]>(None);
    }

    #[inline]
    pub fn append_option<S: AsRef<[u8]>>(&mut self, opt: Option<S>) {
        self.builder.push(opt);
    }

    pub fn finish(mut self) -> BinaryChunked {
        let arr = self.builder.as_box();
        let length = arr.len() as IdxSize;

        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            bit_settings: Default::default(),
            length,
        }
    }

    fn shrink_to_fit(&mut self) {
        self.builder.shrink_to_fit()
    }
}

pub struct BinaryChunkedBuilderCow {
    builder: BinaryChunkedBuilder,
}

impl BinaryChunkedBuilderCow {
    pub fn new(name: &str, capacity: usize) -> Self {
        BinaryChunkedBuilderCow {
            builder: BinaryChunkedBuilder::new(name, capacity, capacity),
        }
    }
}

impl ChunkedBuilder<Cow<'_, [u8]>, BinaryType> for BinaryChunkedBuilderCow {
    #[inline]
    fn append_value(&mut self, val: Cow<'_, [u8]>) {
        self.builder.append_value(val.as_ref())
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(self) -> ChunkedArray<BinaryType> {
        self.builder.finish()
    }

    fn shrink_to_fit(&mut self) {
        self.builder.shrink_to_fit()
    }
}
