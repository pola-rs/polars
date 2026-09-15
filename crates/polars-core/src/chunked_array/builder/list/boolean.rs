use super::*;

pub struct ListBooleanChunkedBuilder {
    builder: LargeListBooleanBuilder,
    field: Field,
    fast_explode: bool,
}

impl ListBooleanChunkedBuilder {
    pub fn new(name: PlSmallStr, capacity: usize, values_capacity: usize) -> Self {
        let values = PlBooleanArrayBuilder::with_capacity(values_capacity);
        let builder = LargeListBooleanBuilder::with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Boolean)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_iter<I: Iterator<Item = Option<bool>> + TrustedLen>(&mut self, iter: I) {
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
    pub(crate) fn append(&mut self, ca: &BooleanChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        // The chunks are appended whole, which leaves each of them in whatever representation it
        // is in rather than reading it an element at a time.
        let values = self.builder.values_mut();
        ca.downcast_iter()
            .for_each(|arr| values.extend(arr, ShareStrategy::Always));
        self.builder.finish_row();
    }
}

impl ListBuilderTrait for ListBooleanChunkedBuilder {
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.extend_nulls(1);
    }

    #[inline]
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let ca = s.bool()?;
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
