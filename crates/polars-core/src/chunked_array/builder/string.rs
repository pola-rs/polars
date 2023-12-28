use super::*;

#[derive(Clone)]
pub struct StringChunkedBuilder {
    pub(crate) builder: MutableUtf8Array<i64>,
    pub capacity: usize,
    pub(crate) field: Field,
}

impl StringChunkedBuilder {
    /// Create a new StringChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    /// * `bytes_capacity` - Number of bytes needed to store the string values.
    pub fn new(name: &str, capacity: usize, bytes_capacity: usize) -> Self {
        StringChunkedBuilder {
            builder: MutableUtf8Array::<i64>::with_capacities(capacity, bytes_capacity),
            capacity,
            field: Field::new(name, DataType::String),
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

    pub fn finish(mut self) -> StringChunked {
        let arr = self.builder.as_box();

        let mut ca = ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
            null_count: 0,
        };
        ca.compute_len();
        ca
    }

    fn shrink_to_fit(&mut self) {
        self.builder.shrink_to_fit()
    }
}

pub struct StringChunkedBuilderCow {
    builder: StringChunkedBuilder,
}

impl StringChunkedBuilderCow {
    pub fn new(name: &str, capacity: usize) -> Self {
        StringChunkedBuilderCow {
            builder: StringChunkedBuilder::new(name, capacity, capacity),
        }
    }
}

impl ChunkedBuilder<Cow<'_, str>, StringType> for StringChunkedBuilderCow {
    #[inline]
    fn append_value(&mut self, val: Cow<'_, str>) {
        self.builder.append_value(val.as_ref())
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(self) -> ChunkedArray<StringType> {
        self.builder.finish()
    }

    fn shrink_to_fit(&mut self) {
        self.builder.shrink_to_fit()
    }
}
