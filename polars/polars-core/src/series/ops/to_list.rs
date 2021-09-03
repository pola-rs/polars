use crate::prelude::*;

impl Series {
    /// Convert the values of this Series to a ListChunked with a length of 1,
    /// So a Series of:
    /// `[1, 2, 3]` becomes `[[1, 2, 3]]`
    pub fn to_list(&self) -> Result<ListChunked> {
        let s = self.rechunk();
        let values = &s.chunks()[0];

        let mut offsets = AlignedVec::with_capacity(2);
        offsets.push(0i64);
        offsets.push(values.len() as i64);

        let data_type = ListArray::<i64>::default_datatype(self.dtype().to_arrow());

        let arr = ListArray::from_data(data_type, offsets.into(), values.clone(), None);

        Ok(ListChunked::new_from_chunks(
            self.name(),
            vec![Arc::new(arr)],
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_to_list() -> Result<()> {
        let s = Series::new("a", &[1, 2, 3]);

        let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name());
        builder.append_series(&s);
        let expected = builder.finish();

        let out = s.to_list()?;
        assert!(expected.into_series().series_equal(&out.into_series()));

        Ok(())
    }
}
