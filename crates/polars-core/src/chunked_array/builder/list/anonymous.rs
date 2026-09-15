use super::*;

/// A list builder told the shape of its values by the first series appended to it.
pub struct AnonymousOwnedListBuilder {
    name: PlSmallStr,
    /// The builder, once the shape of the values is known.
    builder: Option<PlListArrayBuilder>,
    /// The rows appended before the builder existed: `None` a null row, `Some(n)` `n` nulls.
    pending: Vec<Option<usize>>,
    capacity: usize,
    inner_dtype: Option<DataType>,
    fast_explode: bool,
}

impl Default for AnonymousOwnedListBuilder {
    fn default() -> Self {
        Self::new(PlSmallStr::EMPTY, 0, None)
    }
}

impl AnonymousOwnedListBuilder {
    pub fn new(name: PlSmallStr, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name,
            builder: None,
            pending: Vec::new(),
            capacity,
            inner_dtype,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.append_nulls(0);
    }

    /// Appends one row covering `length` nulls, all a row can be before the shape is known.
    fn append_nulls(&mut self, length: usize) {
        match &mut self.builder {
            Some(builder) => {
                builder.values_mut().extend_nulls(length);
                builder.finish_row();
            },
            None => self.pending.push(Some(length)),
        }
    }

    /// The builder, made to build values shaped like `dtype` if it does not exist yet.
    fn builder_for(&mut self, dtype: &DataType) -> &mut PlListArrayBuilder {
        self.builder.get_or_insert_with(|| {
            // A builder is shaped like the array it builds, which an empty chunk of the physical
            // type is enough to ask for.
            let values = builder_like(&*new_empty_chunk(dtype));
            let mut builder = PlListArrayBuilder::with_capacity(values, self.capacity);

            // The rows that were appended before the shape was known go in first, so that the
            // ones appended after them still follow them.
            for row in self.pending.drain(..) {
                match row {
                    None => builder.extend_nulls(1),
                    Some(length) => {
                        builder.values_mut().extend_nulls(length);
                        builder.finish_row();
                    },
                }
            }
            builder
        })
    }
}

impl ListBuilderTrait for AnonymousOwnedListBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        match (s.dtype(), &self.inner_dtype) {
            (DataType::Null, _) => {},
            (dt, None) => self.inner_dtype = Some(dt.clone()),
            (dt, Some(set_dt)) => {
                polars_ensure!(dt == set_dt, ComputeError: "dtypes don't match, got {}, expected: {}", dt.pretty_format(), set_dt.pretty_format());
            },
        }

        if s.is_empty() {
            self.append_empty();
        } else if s.dtype().is_null() {
            // A series of nulls says nothing about the shape of the values, so its elements are
            // appended as nulls whether or not there is a builder to append them to yet.
            self.append_nulls(s.len());
        } else {
            let dtype = s.dtype().clone();
            let builder = self.builder_for(&dtype);
            // The chunks are appended whole, so each of them stays in whatever representation it
            // is in, and a series of several chunks does not have to be rechunked first.
            let values = builder.values_mut();
            for chunk in s.chunks() {
                values.extend(&**chunk, ShareStrategy::Always);
            }
            builder.finish_row();
        }
        Ok(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        match &mut self.builder {
            Some(builder) => builder.extend_nulls(1),
            None => self.pending.push(None),
        }
    }

    fn finish(&mut self) -> ListChunked {
        // Nothing said what the values are, so they are nulls: either that is what was appended,
        // or nothing was.
        let inner_dtype = std::mem::take(&mut self.inner_dtype).unwrap_or(DataType::Null);
        let arr = self.builder_for(&inner_dtype).freeze_reset();

        let mut ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        if self.fast_explode {
            ca.set_fast_explode();
        }
        ca.field = Arc::new(Field::new(
            self.name.clone(),
            DataType::List(Box::new(inner_dtype)),
        ));
        ca
    }
}
