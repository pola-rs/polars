use super::*;

pub struct ListStringChunkedBuilder {
    builder: LargeListStringBuilder,
    field: Field,
    fast_explode: bool,
}

impl ListStringChunkedBuilder {
    pub fn new(name: PlSmallStr, capacity: usize, values_capacity: usize) -> Self {
        let values = PlUtf8ViewArrayBuilder::with_capacity(values_capacity);
        let builder = LargeListStringBuilder::with_capacity(values, capacity);
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
        let values = self.builder.values_mut();
        for value in iter {
            values.push(value);
        }
        self.builder.finish_row();
    }

    #[inline]
    pub fn append_values_iter<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        let values = self.builder.values_mut();
        for value in iter {
            values.push_value(value);
        }
        self.builder.finish_row();
    }

    #[inline]
    pub(crate) fn append(&mut self, ca: &StringChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        // The chunks are appended whole, which leaves each of them in whatever representation it
        // is in rather than reading it an element at a time, and shares their byte buffers rather
        // than copying the bytes out.
        let values = self.builder.values_mut();
        for arr in ca.downcast_iter() {
            values.extend(arr, ShareStrategy::Always);
        }
        self.builder.finish_row();
    }
}

impl ListBuilderTrait for ListStringChunkedBuilder {
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.extend_nulls(1);
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

    fn inner_array(&mut self) -> PlArrayRef {
        Box::new(self.builder.freeze_reset())
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
    pub fn new(name: PlSmallStr, capacity: usize, values_capacity: usize) -> Self {
        let values = PlBinaryViewArrayBuilder::with_capacity(values_capacity);
        let builder = LargeListBinaryBuilder::with_capacity(values, capacity);
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
        let values = self.builder.values_mut();
        for value in iter {
            values.push(value);
        }
        self.builder.finish_row();
    }

    pub fn append_values_iter<'a, I: Iterator<Item = &'a [u8]>>(&mut self, iter: I) {
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        let values = self.builder.values_mut();
        for value in iter {
            values.push_value(value);
        }
        self.builder.finish_row();
    }

    pub(crate) fn append(&mut self, ca: &BinaryChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        // The chunks are appended whole, which leaves each of them in whatever representation it
        // is in rather than reading it an element at a time, and shares their byte buffers rather
        // than copying the bytes out.
        let values = self.builder.values_mut();
        for arr in ca.downcast_iter() {
            values.extend(arr, ShareStrategy::Always);
        }
        self.builder.finish_row();
    }
}

impl ListBuilderTrait for ListBinaryChunkedBuilder {
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.extend_nulls(1);
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

    fn inner_array(&mut self) -> PlArrayRef {
        Box::new(self.builder.freeze_reset())
    }

    fn fast_explode(&self) -> bool {
        self.fast_explode
    }
}
