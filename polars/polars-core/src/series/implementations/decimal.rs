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
        todo!()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        todo!()
    }

    fn take_iter(&self, _iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        todo!()
    }

    unsafe fn take_iter_unchecked(&self, _iter: &mut dyn TakeIterator) -> Series {
        todo!()
    }

    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> PolarsResult<Series> {
        todo!()
    }

    unsafe fn take_opt_iter_unchecked(&self, _iter: &mut dyn TakeIteratorNulls) -> Series {
        todo!()
    }

    fn take(&self, _indices: &IdxCa) -> PolarsResult<Series> {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn take_every(&self, n: usize) -> Series {
        todo!()
    }

    fn has_validity(&self) -> bool {
        todo!()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        todo!()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        todo!()
    }
}
