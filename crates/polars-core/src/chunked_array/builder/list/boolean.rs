use super::*;

pub struct ListBooleanChunkedBuilder {
    builder: LargeListBooleanBuilder,
    field: Field,
    fast_explode: bool,
}

impl ListBooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableBooleanArray::with_capacity(values_capacity);
        let builder = LargeListBooleanBuilder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Boolean)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_iter<I: Iterator<Item = Option<bool>> + TrustedLen>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // SAFETY:
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub(crate) fn append(&mut self, ca: &BooleanChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        let value_builder = self.builder.mut_values();
        value_builder.extend(ca);
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListBooleanChunkedBuilder {
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
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

    fn inner_array(&mut self) -> ArrayRef {
        self.builder.as_box()
    }

    fn fast_explode(&self) -> bool {
        self.fast_explode
    }
}
