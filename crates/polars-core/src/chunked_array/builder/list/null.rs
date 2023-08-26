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
}

impl ListBuilderTrait for ListNullChunkedBuilder {
    #[inline]
    fn append_series(&mut self, _s: &Series) -> PolarsResult<()> {
        self.builder.push_null();
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
