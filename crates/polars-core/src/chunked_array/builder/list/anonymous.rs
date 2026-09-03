use super::*;

pub struct AnonymousOwnedListBuilder {
    name: PlSmallStr,
    builder: AnonymousBuilder<'static>,
    owned: Vec<Series>,
    /// The chunks handed to the builder, which is written against the Arrow arrays; they are kept
    /// here so that the references it holds stay alive until it is finished.
    owned_arrow: Vec<ArrayRef>,
    inner_dtype: Option<DataType>,
    fast_explode: bool,
}

impl Default for AnonymousOwnedListBuilder {
    fn default() -> Self {
        Self::new(PlSmallStr::EMPTY, 0, None)
    }
}

impl ListBuilderTrait for AnonymousOwnedListBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        self.append_owned_series(s.clone())
    }

    fn append_owned_series(&mut self, s: Series) -> PolarsResult<()> {
        match (s.dtype(), &self.inner_dtype) {
            (DataType::Null, _) => {},
            (dt, None) => self.inner_dtype = Some(dt.clone()),
            (dt, Some(set_dt)) => {
                polars_ensure!(dt == set_dt, ComputeError: "dtypes don't match, got {}, expected: {}", dt.pretty_format(), set_dt.pretty_format());
            },
        }
        if s.is_empty() {
            self.append_empty();
        } else {
            // The builder is the Arrow one, so the chunk crosses over — see `polars_array::arrow::bridge`. It
            // takes one array per element, so a series of several chunks is rechunked first.
            let s = if s.n_chunks() > 1 { s.rechunk() } else { s };
            let arrow = polars_array::arrow::export::to_arrow(&*s.chunks()[0]);
            // SAFETY: the array is kept alive in `owned_arrow` until the builder is finished, and
            // it lives on the heap, so growing that vector does not move it.
            let arrow_ref: &'static dyn Array = unsafe { &*(&*arrow as *const dyn Array) };
            self.owned_arrow.push(arrow);
            self.builder.push(arrow_ref);
            // This ensures that the underlying ArrayRef's are not dropped.
            self.owned.push(s);
        }
        Ok(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null()
    }

    fn finish(&mut self) -> ListChunked {
        let inner_dtype = std::mem::take(&mut self.inner_dtype);
        // Don't use self from here on out.
        let slf = std::mem::take(self);
        let inner_dtype_physical = inner_dtype
            .as_ref()
            .map(|dt| dt.to_physical().to_arrow(CompatLevel::newest()));
        let arr = slf.builder.finish(inner_dtype_physical.as_ref()).unwrap();

        let list_dtype_logical = match inner_dtype {
            None => DataType::from_arrow_dtype(arr.dtype()),
            Some(dt) => DataType::List(Box::new(dt)),
        };

        let mut ca = ListChunked::with_chunk(
            PlSmallStr::EMPTY,
            <PlListArray as ToArrow>::from_arrow(&arr),
        );
        if slf.fast_explode {
            ca.set_fast_explode();
        }
        ca.field = Arc::new(Field::new(slf.name, list_dtype_logical));
        ca
    }
}

impl AnonymousOwnedListBuilder {
    pub fn new(name: PlSmallStr, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name,
            builder: AnonymousBuilder::new(capacity),
            owned: Vec::with_capacity(capacity),
            owned_arrow: Vec::with_capacity(capacity),
            inner_dtype,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.builder.push_empty()
    }
}
