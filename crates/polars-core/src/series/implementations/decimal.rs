use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::prelude::*;

unsafe impl IntoSeries for DecimalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DecimalChunked> {}

impl SeriesWrap<DecimalChunked> {
    fn apply_physical<F: Fn(&Int128Chunked) -> Int128Chunked>(&self, f: F) -> Series {
        f(&self.0)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn agg_helper<F: Fn(&Int128Chunked) -> Series>(&self, f: F) -> Series {
        let agg_s = f(&self.0);
        let ca = agg_s.decimal().unwrap();
        let ca = ca.as_ref().clone();
        let precision = self.0.precision();
        let scale = self.0.scale();
        ca.into_decimal_unchecked(precision, scale).into_series()
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
    fn _get_flags(&self) -> Settings {
        self.0.get_flags()
    }
    fn _set_flags(&mut self, flags: Settings) {
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
        self.0.agg_list(groups)
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
}

impl SeriesTrait for SeriesWrap<DecimalChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name)
    }

    fn chunk_lengths(&self) -> ChunkIdIter {
        self.0.chunk_id()
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
        self.apply_physical(|ca| ca.slice(offset, length))
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        let other = other.decimal()?;
        self.0.append(&other.0);
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        let other = other.decimal()?;
        self.0.extend(&other.0);
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

    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.0.cast(data_type)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn reverse(&self) -> Series {
        self.apply_physical(|ca| ca.reverse())
    }

    fn shift(&self, periods: i64) -> Series {
        self.apply_physical(|ca| ca.shift(periods))
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn _sum_as_series(&self) -> PolarsResult<Series> {
        Ok(self.apply_physical(|ca| {
            let sum = ca.sum();
            Int128Chunked::from_slice_options(self.name(), &[sum])
        }))
    }
    fn min_as_series(&self) -> PolarsResult<Series> {
        Ok(self.apply_physical(|ca| {
            let min = ca.min();
            Int128Chunked::from_slice_options(self.name(), &[min])
        }))
    }
    fn max_as_series(&self) -> PolarsResult<Series> {
        Ok(self.apply_physical(|ca| {
            let max = ca.max();
            Int128Chunked::from_slice_options(self.name(), &[max])
        }))
    }
    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}
