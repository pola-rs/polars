use super::*;

pub struct ListStringChunkedBuilder {
    builder: LargeListBinViewBuilder<str>,
    field: Field,
    fast_explode: bool,
}

impl ListStringChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableBinaryViewArray::with_capacity(values_capacity);
        let builder = LargeListBinViewBuilder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::String)));

        ListStringChunkedBuilder {
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
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // SAFETY:
        // trusted len, trust the type system
        self.builder.mut_values().extend_trusted_len(iter);
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub fn append_values_iter<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        self.builder.mut_values().extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub(crate) fn append(&mut self, ca: &StringChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        for arr in ca.downcast_iter() {
            if arr.null_count() == 0 {
                self.builder
                    .mut_values()
                    .extend_values(arr.non_null_values_iter());
            } else {
                self.builder.mut_values().extend_trusted_len(arr.iter())
            }
        }
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListStringChunkedBuilder {
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
        let ca = s.str()?;
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
    builder: LargeListBinViewBuilder<[u8]>,
    field: Field,
    fast_explode: bool,
}

impl ListBinaryChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutablePlBinary::with_capacity(values_capacity);
        let builder = LargeListBinViewBuilder::new_with_capacity(values, capacity);
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
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // SAFETY:
        // trusted len, trust the type system
        self.builder.mut_values().extend_trusted_len(iter);
        self.builder.try_push_valid().unwrap();
    }

    pub fn append_values_iter<'a, I: Iterator<Item = &'a [u8]>>(&mut self, iter: I) {
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        self.builder.mut_values().extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    pub(crate) fn append(&mut self, ca: &BinaryChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        for arr in ca.downcast_iter() {
            if arr.null_count() == 0 {
                self.builder
                    .mut_values()
                    .extend_values(arr.non_null_values_iter());
            } else {
                self.builder.mut_values().extend_trusted_len(arr.iter())
            }
        }
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
