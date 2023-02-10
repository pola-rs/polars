use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::prelude::*;

unsafe impl IntoSeries for DecimalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DecimalChunked> {}

impl private::PrivateSeries for SeriesWrap<DecimalChunked> {}

impl SeriesTrait for SeriesWrap<DecimalChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name)
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.chunks()
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_decimal(self.0.precision(), self.0.scale())
            .into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        let mut out = ChunkTake::take_unchecked(self.0.deref(), idx.into());

        if self.0.is_sorted_flag() && (idx.is_sorted_flag() || idx.is_sorted_reverse_flag()) {
            out.set_sorted_flag(idx.is_sorted_flag2())
        }

        Ok(out
            .into_decimal(self.0.precision(), self.0.scale())
            .into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_decimal(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), indices.into()).map(|ca| {
            ca.into_decimal(self.0.precision(), self.0.scale())
                .into_series()
        })
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), iter.into()).map(|ca| {
            ca.into_decimal(self.0.precision(), self.0.scale())
                .into_series()
        })
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn take_every(&self, n: usize) -> Series {
        self.0
            .take_every(n)
            .into_decimal(self.0.precision(), self.0.scale())
            .into_series()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        let ca = self.0.deref().take_chunked_unchecked(by, sorted);
        ca.into_decimal(self.0.precision(), self.0.scale())
            .into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        let ca = self.0.deref().take_opt_chunked_unchecked(by);
        ca.into_decimal(self.0.precision(), self.0.scale())
            .into_series()
    }
}
