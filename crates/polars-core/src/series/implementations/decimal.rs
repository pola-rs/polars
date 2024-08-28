use super::*;
use crate::prelude::*;

unsafe impl IntoSeries for DecimalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DecimalChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl SeriesWrap<DecimalChunked> {
    fn apply_physical_to_s<F: Fn(&Int128Chunked) -> Int128Chunked>(&self, f: F) -> Series {
        f(&self.0)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn apply_physical<T, F: Fn(&Int128Chunked) -> T>(&self, f: F) -> T {
        f(&self.0)
    }

    fn scale_factor(&self) -> u128 {
        10u128.pow(self.0.scale() as u32)
    }

    fn apply_scale(&self, mut scalar: Scalar) -> Scalar {
        debug_assert_eq!(scalar.dtype(), &DataType::Float64);
        let v = scalar
            .value()
            .try_extract::<f64>()
            .expect("should be f64 scalar");
        scalar.update((v / self.scale_factor() as f64).into());
        scalar
    }

    fn agg_helper<F: Fn(&Int128Chunked) -> Series>(&self, f: F) -> Series {
        let agg_s = f(&self.0);
        match agg_s.dtype() {
            DataType::Decimal(_, _) => {
                let ca = agg_s.decimal().unwrap();
                let ca = ca.as_ref().clone();
                let precision = self.0.precision();
                let scale = self.0.scale();
                ca.into_decimal_unchecked(precision, scale).into_series()
            },
            DataType::List(dtype) if dtype.is_decimal() => {
                let dtype = self.0.dtype();
                let ca = agg_s.list().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                // SAFETY: dtype is passed correctly
                let s = unsafe {
                    Series::from_chunks_and_dtype_unchecked("", vec![arr.values().clone()], dtype)
                };
                let new_values = s.array_ref(0).clone();
                let data_type =
                    ListArray::<i64>::default_datatype(dtype.to_arrow(CompatLevel::newest()));
                let new_arr = ListArray::<i64>::new(
                    data_type,
                    arr.offsets().clone(),
                    new_values,
                    arr.validity().cloned(),
                );
                unsafe {
                    ListChunked::from_chunks_and_dtype_unchecked(
                        agg_s.name(),
                        vec![Box::new(new_arr)],
                        DataType::List(Box::new(self.dtype().clone())),
                    )
                    .into_series()
                }
            },
            _ => unreachable!(),
        }
    }
}

unsafe impl IntoSeries for Int128Chunked {
    fn into_series(self) -> Series
    where
        Self: Sized,
    {
        // this is incorrect as it ignores the datatype
        // the caller must correct this.
        let mut ca = DecimalChunked::new_logical(self);
        ca.2 = Some(DataType::Decimal(None, None));
        ca.into_series()
    }
}

impl private::PrivateSeries for SeriesWrap<DecimalChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }
    fn _get_flags(&self) -> MetadataFlags {
        self.0.get_flags()
    }
    fn _set_flags(&mut self, flags: MetadataFlags) {
        self.0.set_flags(flags)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        Ok(self
            .0
            .zip_with(mask, other.as_ref().as_ref())?
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }
    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        (&self.0).into_total_eq_inner()
    }
    fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
        (&self.0).into_total_ord_inner()
    }

    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.vec_hash(random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        build_hasher: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.0.vec_hash_combine(build_hasher, hashes)?;
        Ok(())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        self.agg_helper(|ca| ca.agg_sum(groups))
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.agg_helper(|ca| ca.agg_min(groups))
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.agg_helper(|ca| ca.agg_max(groups))
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.agg_helper(|ca| ca.agg_list(groups))
    }

    fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
        let rhs = rhs.decimal()?;
        ((&self.0) - rhs).map(|ca| ca.into_series())
    }
    fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
        let rhs = rhs.decimal()?;
        ((&self.0) + rhs).map(|ca| ca.into_series())
    }
    fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
        let rhs = rhs.decimal()?;
        ((&self.0) * rhs).map(|ca| ca.into_series())
    }
    fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
        let rhs = rhs.decimal()?;
        ((&self.0) / rhs).map(|ca| ca.into_series())
    }
    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        self.0.group_tuples(multithreaded, sorted)
    }
}

impl SeriesTrait for SeriesWrap<DecimalChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name)
    }

    fn chunk_lengths(&self) -> ChunkLenIter {
        self.0.chunk_lengths()
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.chunks()
    }
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.apply_physical_to_s(|ca| ca.slice(offset, length))
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (a, b) = self.0.split_at(offset);
        let a = a
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series();
        let b = b
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series();
        (a, b)
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        let other = other.decimal()?;
        self.0.append(&other.0)?;
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        let other = other.decimal()?;
        self.0.extend(&other.0)?;
        Ok(())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        Ok(self
            .0
            .filter(filter)?
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(self
            .0
            .take(indices)?
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        self.0
            .take_unchecked(indices)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        Ok(self
            .0
            .take(indices)?
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        self.0
            .take_unchecked(indices)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        let ca = self.0.rechunk();
        ca.into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.0
            .new_from_index(index, length)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn cast(&self, data_type: &DataType, cast_options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(data_type, cast_options)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        Ok(self
            .0
            .sort_with(options)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.has_nulls()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn reverse(&self) -> Series {
        self.apply_physical_to_s(|ca| ca.reverse())
    }

    fn shift(&self, periods: i64) -> Series {
        self.apply_physical_to_s(|ca| ca.shift(periods))
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn sum_reduce(&self) -> PolarsResult<Scalar> {
        Ok(self.apply_physical(|ca| {
            let sum = ca.sum();
            let DataType::Decimal(_, Some(scale)) = self.dtype() else {
                unreachable!()
            };
            let av = AnyValue::Decimal(sum.unwrap(), *scale);
            Scalar::new(self.dtype().clone(), av)
        }))
    }
    fn min_reduce(&self) -> PolarsResult<Scalar> {
        Ok(self.apply_physical(|ca| {
            let min = ca.min();
            let DataType::Decimal(_, Some(scale)) = self.dtype() else {
                unreachable!()
            };
            let av = if let Some(min) = min {
                AnyValue::Decimal(min, *scale)
            } else {
                AnyValue::Null
            };
            Scalar::new(self.dtype().clone(), av)
        }))
    }
    fn max_reduce(&self) -> PolarsResult<Scalar> {
        Ok(self.apply_physical(|ca| {
            let max = ca.max();
            let DataType::Decimal(_, Some(scale)) = self.dtype() else {
                unreachable!()
            };
            let av = if let Some(m) = max {
                AnyValue::Decimal(m, *scale)
            } else {
                AnyValue::Null
            };
            Scalar::new(self.dtype().clone(), av)
        }))
    }

    fn mean(&self) -> Option<f64> {
        self.0.mean().map(|v| v / self.scale_factor() as f64)
    }

    fn median(&self) -> Option<f64> {
        self.0.median().map(|v| v / self.scale_factor() as f64)
    }
    fn median_reduce(&self) -> PolarsResult<Scalar> {
        Ok(self.apply_scale(self.0.median_reduce()))
    }

    fn std(&self, ddof: u8) -> Option<f64> {
        self.0.std(ddof).map(|v| v / self.scale_factor() as f64)
    }
    fn std_reduce(&self, ddof: u8) -> PolarsResult<Scalar> {
        Ok(self.apply_scale(self.0.std_reduce(ddof)))
    }

    fn quantile_reduce(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Scalar> {
        self.0
            .quantile_reduce(quantile, interpol)
            .map(|v| self.apply_scale(v))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}
