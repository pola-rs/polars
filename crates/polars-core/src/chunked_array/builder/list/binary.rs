use super::*;

pub struct ListUtf8ChunkedBuilder {
    builder: LargeListUtf8Builder,
    field: Field,
    fast_explode: bool,
}

impl ListUtf8ChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableUtf8Array::<i64>::with_capacity(values_capacity);
        let builder = LargeListUtf8Builder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Utf8)));

        ListUtf8ChunkedBuilder {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_trusted_len_iter<'a, I: Iterator<Item = Option<&'a str>> + TrustedLen>(
        &mut self,
        iter: I,
    ) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub fn append_values_iter<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        values.extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub(crate) fn append(&mut self, ca: &Utf8Chunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        let value_builder = self.builder.mut_values();
        value_builder.try_extend(ca).unwrap();
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListUtf8ChunkedBuilder {
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let ca = s.utf8()?;
        self.append(ca);
        Ok(())
    }

    fn field(&self) -> &Field {
        &self.field
    }

    fn inner_array(&mut self) -> ArrayRef {
        self.builder.as_box()
    }

    fn fast_explode(&self) -> bool {
        self.fast_explode
    }
}

pub struct ListBinaryChunkedBuilder {
    builder: LargeListBinaryBuilder,
    field: Field,
    fast_explode: bool,
}

impl ListBinaryChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableBinaryArray::<i64>::with_capacity(values_capacity);
        let builder = LargeListBinaryBuilder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Binary)));

        ListBinaryChunkedBuilder {
            builder,
            field,
            fast_explode: true,
        }
    }

    pub fn append_trusted_len_iter<'a, I: Iterator<Item = Option<&'a [u8]>> + TrustedLen>(
        &mut self,
        iter: I,
    ) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    pub fn append_values_iter<'a, I: Iterator<Item = &'a [u8]>>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        values.extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    pub(crate) fn append(&mut self, ca: &BinaryChunked) {
        let value_builder = self.builder.mut_values();
        value_builder.try_extend(ca).unwrap();
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListBinaryChunkedBuilder {
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let ca = s.binary()?;
        self.append(ca);
        Ok(())
    }

    fn field(&self) -> &Field {
        &self.field
    }

    fn inner_array(&mut self) -> ArrayRef {
        self.builder.as_box()
    }

    fn fast_explode(&self) -> bool {
        self.fast_explode
    }
}
