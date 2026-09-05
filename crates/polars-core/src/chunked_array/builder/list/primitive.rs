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
        name: PlSmallStr,
        capacity: usize,
        values_capacity: usize,
        inner_type: DataType,
    ) -> Self {
        debug_assert!(
            inner_type.to_physical().is_primitive_numeric(),
            "inner type must be primitive, got {inner_type}"
        );
        let values = PlPrimitiveArrayBuilder::<T::Native>::with_capacity(values_capacity);
        let builder = LargePrimitiveBuilder::<T::Native>::with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(inner_type)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_slice(&mut self, items: &[T::Native]) {
        self.builder.values_mut().push_values(items.iter().copied());
        self.builder.finish_row();

        if items.is_empty() {
            self.fast_explode = false;
        }
    }

    #[inline]
    pub fn append_opt_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(items) => self.append_slice(items),
            None => {
                self.builder.extend_nulls(1);
            },
        }
    }
    /// Appends from an iterator over values
    #[inline]
    pub fn append_values_iter_trusted_len<I: Iterator<Item = T::Native> + TrustedLen>(
        &mut self,
        iter: I,
    ) {
        self.append_values_iter(iter)
    }

    #[inline]
    pub fn append_values_iter<I: Iterator<Item = T::Native>>(&mut self, iter: I) {
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        self.builder.values_mut().push_values(iter);
        self.builder.finish_row();
    }

    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter<I: Iterator<Item = Option<T::Native>> + TrustedLen>(&mut self, iter: I) {
        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        let values = self.builder.values_mut();
        for value in iter {
            values.push(value);
        }
        self.builder.finish_row();
    }
}

impl<T> ListBuilderTrait for ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.extend_nulls(1);
    }

    #[inline]
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let physical = s.to_physical_repr();
        let ca = physical.unpack::<T>().map_err(|_| {
            polars_err!(SchemaMismatch: "cannot build list with different dtypes 

Expected {}, got {}.", self.field.dtype(), s.dtype())
        })?;
        // The chunks are appended whole, which leaves each of them in whatever representation it
        // is in rather than reading it an element at a time.
        let values = self.builder.values_mut();
        ca.downcast_iter()
            .for_each(|arr| values.extend(arr, ShareStrategy::Always));
        self.builder.finish_row();
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
