use super::*;

pub struct AnonymousListBuilder<'a> {
    name: String,
    builder: AnonymousBuilder<'a>,
    fast_explode: bool,
    inner_dtype: DtypeMerger,
}

impl Default for AnonymousListBuilder<'_> {
    fn default() -> Self {
        Self::new("", 0, None)
    }
}

impl<'a> AnonymousListBuilder<'a> {
    pub fn new(name: &str, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name: name.into(),
            builder: AnonymousBuilder::new(capacity),
            fast_explode: true,
            inner_dtype: DtypeMerger::new(inner_dtype),
        }
    }

    pub fn append_opt_series(&mut self, opt_s: Option<&'a Series>) -> PolarsResult<()> {
        match opt_s {
            Some(s) => return self.append_series(s),
            None => {
                self.append_null();
            },
        }
        Ok(())
    }

    pub fn append_opt_array(&mut self, opt_s: Option<&'a dyn Array>) {
        match opt_s {
            Some(s) => self.append_array(s),
            None => {
                self.append_null();
            },
        }
    }

    pub fn append_array(&mut self, arr: &'a dyn Array) {
        self.builder.push(arr)
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.builder.push_empty()
    }

    pub fn append_series(&mut self, s: &'a Series) -> PolarsResult<()> {
        match s.dtype() {
            // Empty arrays tend to be null type and thus differ
            // if we would push it the concat would fail.
            DataType::Null if s.is_empty() => self.append_empty(),
            dt => self.inner_dtype.update(dt)?,
        }
        self.builder.push_multiple(s.chunks());
        Ok(())
    }

    pub fn finish(&mut self) -> ListChunked {
        // Don't use self from here on out.
        let slf = std::mem::take(self);
        if slf.builder.is_empty() {
            ListChunked::full_null_with_dtype(
                &slf.name,
                0,
                &slf.inner_dtype.materialize().unwrap_or(DataType::Null),
            )
        } else {
            let inner_dtype = slf.inner_dtype.materialize();

            let inner_dtype_physical = inner_dtype
                .as_ref()
                .map(|dt| dt.to_physical().to_arrow(CompatLevel::newest()));
            let arr = slf.builder.finish(inner_dtype_physical.as_ref()).unwrap();

            let list_dtype_logical = match inner_dtype {
                None => DataType::from(arr.data_type()),
                Some(dt) => DataType::List(Box::new(dt)),
            };

            let mut ca = ListChunked::with_chunk("", arr);
            if slf.fast_explode {
                ca.set_fast_explode();
            }
            ca.field = Arc::new(Field::new(&slf.name, list_dtype_logical));
            ca
        }
    }
}

pub struct AnonymousOwnedListBuilder {
    name: String,
    builder: AnonymousBuilder<'static>,
    owned: Vec<Series>,
    inner_dtype: DtypeMerger,
    fast_explode: bool,
}

impl Default for AnonymousOwnedListBuilder {
    fn default() -> Self {
        Self::new("", 0, None)
    }
}

impl ListBuilderTrait for AnonymousOwnedListBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        if s.is_empty() {
            self.append_empty();
        } else {
            unsafe {
                self.inner_dtype.update(s.dtype())?;
                self.builder
                    .push_multiple(&*(s.chunks().as_ref() as *const [ArrayRef]));
            }
            // This make sure that the underlying ArrayRef's are not dropped.
            self.owned.push(s.clone());
        }
        Ok(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null()
    }

    fn finish(&mut self) -> ListChunked {
        let inner_dtype = std::mem::take(&mut self.inner_dtype).materialize();
        // Don't use self from here on out.
        let slf = std::mem::take(self);
        let inner_dtype_physical = inner_dtype
            .as_ref()
            .map(|dt| dt.to_physical().to_arrow(CompatLevel::newest()));
        let arr = slf.builder.finish(inner_dtype_physical.as_ref()).unwrap();

        let list_dtype_logical = match inner_dtype {
            None => DataType::from_arrow(arr.data_type(), false),
            Some(dt) => DataType::List(Box::new(dt)),
        };

        let mut ca = ListChunked::with_chunk("", arr);
        if slf.fast_explode {
            ca.set_fast_explode();
        }
        ca.field = Arc::new(Field::new(&slf.name, list_dtype_logical));
        ca
    }
}

impl AnonymousOwnedListBuilder {
    pub fn new(name: &str, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name: name.into(),
            builder: AnonymousBuilder::new(capacity),
            owned: Vec::with_capacity(capacity),
            inner_dtype: DtypeMerger::new(inner_dtype),
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.builder.push_empty()
    }
}
