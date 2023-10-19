use super::*;

pub(super) struct ListCategoricalChunkedBuilder {
    inner: ListPrimitiveChunkedBuilder<UInt32Type>,
    inner_dtype: RevMapMerger,
}

impl ListCategoricalChunkedBuilder {
    pub(super) fn new(
        name: &str,
        capacity: usize,
        values_capacity: usize,
        logical_type: DataType,
    ) -> Self {
        let inner =
            ListPrimitiveChunkedBuilder::new(name, capacity, values_capacity, logical_type.clone());
        let DataType::Categorical(Some(rev_map)) = logical_type else {
            panic!("expected `Categorical` type")
        };
        Self {
            inner,
            inner_dtype: RevMapMerger::new(rev_map),
        }
    }
}

impl ListBuilderTrait for ListCategoricalChunkedBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let DataType::Categorical(Some(rev_map)) = s.dtype() else {
            polars_bail!(ComputeError: "expected `Categorical` type")
        };
        self.inner_dtype.merge_map(rev_map)?;
        self.inner.append_series(s)
    }

    fn append_null(&mut self) {
        self.inner.append_null()
    }

    fn finish(&mut self) -> ListChunked {
        let rev_map = std::mem::take(&mut self.inner_dtype).finish();
        let inner_dtype = DataType::Categorical(Some(rev_map));
        let mut ca = self.inner.finish();
        unsafe { ca.set_dtype(DataType::List(Box::new(inner_dtype))) }
        ca
    }
}
