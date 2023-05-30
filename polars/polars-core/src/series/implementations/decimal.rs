use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::prelude::*;

unsafe impl IntoSeries for DecimalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DecimalChunked> {}

impl SeriesWrap<DecimalChunked> {
    fn apply_logical<F: Fn(&Int128Chunked) -> Int128Chunked>(&self, f: F) -> Series {
        f(&self.0)
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
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

    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        Ok(self
            .0
            .zip_with(mask, other.as_ref().as_ref())?
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
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

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.apply_logical(|ca| ca.slice(offset, length))
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        let other = other.decimal()?;
        self.0.append(&other.0);
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref());
        Ok(())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        Ok(self
            .0
            .filter(filter)?
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        let ca = self.0.deref().take_chunked_unchecked(by, sorted);
        ca.into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        self.apply_logical(|ca| ca.take_opt_chunked_unchecked(by))
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), iter.into()).map(|ca| {
            ca.into_decimal_unchecked(self.0.precision(), self.0.scale())
                .into_series()
        })
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        let mut out = ChunkTake::take_unchecked(self.0.deref(), idx.into());

        if self.0.is_sorted_ascending_flag()
            && (idx.is_sorted_ascending_flag() || idx.is_sorted_descending_flag())
        {
            out.set_sorted_flag(idx.is_sorted_flag())
        }

        Ok(out
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_decimal_unchecked(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), indices.into()).map(|ca| {
            ca.into_decimal_unchecked(self.0.precision(), self.0.scale())
                .into_series()
        })
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
    #[cfg(feature = "private")]
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
        self.apply_logical(|ca| ca.reverse())
    }

    fn shift(&self, periods: i64) -> Series {
        self.apply_logical(|ca| ca.shift(periods))
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }
}
