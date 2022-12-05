use super::*;

pub struct Utf8ChunkedBuilder {
    pub(crate) builder: MutableUtf8Array<i64>,
    pub capacity: usize,
    field: Field,
}

impl Utf8ChunkedBuilder {
    /// Create a new UtfChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    /// * `bytes_capacity` - Number of bytes needed to store the string values.
    pub fn new(name: &str, capacity: usize, bytes_capacity: usize) -> Self {
        Utf8ChunkedBuilder {
            builder: MutableUtf8Array::<i64>::with_capacities(capacity, bytes_capacity),
            capacity,
            field: Field::new(name, DataType::Utf8),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value<S: AsRef<str>>(&mut self, v: S) {
        self.builder.push(Some(v.as_ref()));
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.builder.push::<&str>(None);
    }

    #[inline]
    pub fn append_option<S: AsRef<str>>(&mut self, opt: Option<S>) {
        self.builder.push(opt);
    }

    pub fn finish(mut self) -> Utf8Chunked {
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

pub struct Utf8ChunkedBuilderCow {
    builder: Utf8ChunkedBuilder,
}

impl Utf8ChunkedBuilderCow {
    pub fn new(name: &str, capacity: usize) -> Self {
        Utf8ChunkedBuilderCow {
            builder: Utf8ChunkedBuilder::new(name, capacity, capacity),
        }
    }
}

impl ChunkedBuilder<Cow<'_, str>, Utf8Type> for Utf8ChunkedBuilderCow {
    #[inline]
    fn append_value(&mut self, val: Cow<'_, str>) {
        self.builder.append_value(val.as_ref())
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(self) -> ChunkedArray<Utf8Type> {
        self.builder.finish()
    }

    fn shrink_to_fit(&mut self) {
        self.builder.shrink_to_fit()
    }
}
