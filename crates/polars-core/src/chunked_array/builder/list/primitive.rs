use super::*;

pub struct ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub builder: LargePrimitiveBuilder<T::Native>,
    field: Field,
    fast_explode: bool,
}

impl<T> ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub fn new(
        name: &str,
        capacity: usize,
        values_capacity: usize,
        logical_type: DataType,
    ) -> Self {
        let values = MutablePrimitiveArray::<T::Native>::with_capacity(values_capacity);
        let builder = LargePrimitiveBuilder::<T::Native>::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(logical_type)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    pub fn new_with_values_type(
        name: &str,
        capacity: usize,
        values_capacity: usize,
        values_type: DataType,
        logical_type: DataType,
    ) -> Self {
        let values = MutablePrimitiveArray::<T::Native>::with_capacity_from(
            values_capacity,
            values_type.to_arrow(CompatLevel::newest()),
        );
        let builder = LargePrimitiveBuilder::<T::Native>::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(logical_type)));
        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_slice(&mut self, items: &[T::Native]) {
        let values = self.builder.mut_values();
        values.extend_from_slice(items);
        self.builder.try_push_valid().unwrap();

        if items.is_empty() {
            self.fast_explode = false;
        }
    }

    #[inline]
    pub fn append_opt_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(items) => self.append_slice(items),
            None => {
                self.builder.push_null();
            },
        }
    }
    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter_values<I: Iterator<Item = T::Native> + TrustedLen>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // SAFETY:
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_values_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter<I: Iterator<Item = Option<T::Native>> + TrustedLen>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // SAFETY:
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }
}

impl<T> ListBuilderTrait for ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let physical = s.to_physical_repr();
        let ca = physical.unpack::<T>()?;
        let values = self.builder.mut_values();

        ca.downcast_iter().for_each(|arr| {
            if arr.null_count() == 0 {
                values.extend_from_slice(arr.values().as_slice())
            } else {
                // SAFETY:
                // Arrow arrays are trusted length iterators.
                unsafe { values.extend_trusted_len_unchecked(arr.into_iter()) }
            }
        });
        // overflow of i64 is far beyond polars capable lengths.
        unsafe { self.builder.try_push_valid().unwrap_unchecked() };
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
