use super::*;
use crate::chunked_array::cast::cast_chunks;
use crate::prelude::*;

pub type DecimalChunked = Logical<DecimalType, Int128Type>;

impl Int128Chunked {
    fn update_chunks_dtype(&mut self, precision: Option<usize>, scale: usize) {
        // physical i128 type doesn't exist
        // so we update the decimal dtype
        for arr in self.chunks.iter_mut() {
            let mut default = PrimitiveArray::new_empty(arr.data_type().clone());
            let arr = arr
                .as_any_mut()
                .downcast_mut::<PrimitiveArray<i128>>()
                .unwrap();
            std::mem::swap(arr, &mut default);
            let (_, values, validity) = default.into_inner();

            *arr = PrimitiveArray::new(
                DataType::Decimal(precision, Some(scale)).to_arrow(),
                values,
                validity,
            );
        }
    }

    #[inline]
    pub fn into_decimal_unchecked(
        mut self,
        precision: Option<usize>,
        scale: usize,
    ) -> DecimalChunked {
        self.update_chunks_dtype(precision, scale);
        let mut dt = DecimalChunked::new_logical(self);
        dt.2 = Some(DataType::Decimal(precision, Some(scale)));
        dt
    }

    pub fn into_decimal(
        mut self,
        precision: Option<usize>,
        scale: usize,
    ) -> PolarsResult<DecimalChunked> {
        self.update_chunks_dtype(precision, scale);
        // TODO: if precision is None, do we check that the value fits within precision of 38?...
        if let Some(precision) = precision {
            let precision_max = 10_i128.pow(precision as u32);
            // note: this is not too efficient as it scans through the data twice...
            if let (Some(min), Some(max)) = (self.min(), self.max()) {
                let max_abs = max.abs().max(min.abs());
                polars_ensure!(
                    max_abs < precision_max,
                    ComputeError: "decimal precision {} can't fit values with {} digits",
                    precision,
                    max_abs.to_string().len()
                );
            }
        }
        Ok(self.into_decimal_unchecked(precision, scale))
    }
}

impl LogicalType for DecimalChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    #[inline]
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        Ok(unsafe { self.get_any_value_unchecked(i) })
    }

    #[inline]
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        match self.0.get_unchecked(i) {
            Some(v) => AnyValue::Decimal(v, self.scale()),
            None => AnyValue::Null,
        }
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        let (precision_src, scale_src) = (self.precision(), self.scale());
        if let &DataType::Decimal(precision_dst, scale_dst) = dtype {
            let scale_dst = scale_dst.ok_or_else(
                || polars_err!(ComputeError: "cannot cast to Decimal with unknown scale"),
            )?;
            // for now, let's just allow same-scale conversions
            // where precision is either the same or bigger or gets converted to `None`
            // (these are the easy cases requiring no checks and arithmetics which we can add later)
            let is_widen = match (precision_src, precision_dst) {
                (Some(precision_src), Some(precision_dst)) => precision_dst >= precision_src,
                (_, None) => true,
                _ => false,
            };
            if scale_src == scale_dst && is_widen {
                return self.0.cast(dtype); // no conversion or checks needed
            }
        }
        let chunks = cast_chunks(&self.chunks, dtype, true)?;
        unsafe {
            Ok(Series::from_chunks_and_dtype_unchecked(
                self.name(),
                chunks,
                dtype,
            ))
        }
    }
}

impl DecimalChunked {
    pub fn precision(&self) -> Option<usize> {
        match self.2.as_ref().unwrap() {
            DataType::Decimal(precision, _) => *precision,
            _ => unreachable!(),
        }
    }

    pub fn scale(&self) -> usize {
        match self.2.as_ref().unwrap() {
            DataType::Decimal(_, scale) => scale.unwrap_or_else(|| unreachable!()),
            _ => unreachable!(),
        }
    }
}
