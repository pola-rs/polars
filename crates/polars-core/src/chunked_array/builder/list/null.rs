use super::*;

pub struct ListNullChunkedBuilder {
    builder: LargeListNullBuilder,
    name: String,
}

impl ListNullChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        ListNullChunkedBuilder {
            builder: LargeListNullBuilder::with_capacity(capacity),
            name: name.into(),
        }
    }

    pub(crate) fn append(&mut self, s: &Series) {
        let value_builder = self.builder.mut_values();
        value_builder.extend_nulls(s.len());
        self.builder.try_push_valid().unwrap();
    }

    pub(crate) fn append_with_len(&mut self, len: usize) {
        let value_builder = self.builder.mut_values();
        value_builder.extend_nulls(len);
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListNullChunkedBuilder {
    #[inline]
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        self.append(s);
        Ok(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.push_null();
    }

    fn finish(&mut self) -> ListChunked {
        unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                &self.name,
                vec![self.builder.as_box()],
                DataType::List(Box::new(DataType::Null)),
            )
        }
    }
}
