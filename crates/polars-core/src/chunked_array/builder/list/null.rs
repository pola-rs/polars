use super::*;

pub struct ListNullChunkedBuilder {
    builder: LargeListNullBuilder,
    name: PlSmallStr,
}

impl ListNullChunkedBuilder {
    pub fn new(name: PlSmallStr, capacity: usize) -> Self {
        ListNullChunkedBuilder {
            builder: LargeListNullBuilder::with_capacity(PlNullArrayBuilder::new(), capacity),
            name,
        }
    }

    pub(crate) fn append(&mut self, s: &Series) {
        self.append_with_len(s.len())
    }

    pub(crate) fn append_with_len(&mut self, len: usize) {
        self.builder.values_mut().extend_nulls(len);
        self.builder.finish_row();
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
        self.builder.extend_nulls(1);
    }

    fn finish(&mut self) -> ListChunked {
        unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                self.name.clone(),
                vec![Box::new(self.builder.freeze_reset())],
                DataType::List(Box::new(DataType::Null)),
            )
        }
    }
}
