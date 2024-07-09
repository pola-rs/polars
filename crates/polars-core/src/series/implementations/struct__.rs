use crate::chunked_array::StructChunked2;
use super::*;
use crate::hashing::series_to_hashes;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};


impl PrivateSeriesNumeric for SeriesWrap<StructChunked2> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl PrivateSeries for SeriesWrap<StructChunked2> {
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _get_flags(&self) -> MetadataFlags {
        MetadataFlags::empty()
    }

    fn _set_flags(&mut self, _flags: MetadataFlags) {}


}

impl SeriesTrait for SeriesWrap<StructChunked2> {
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
        &self.0.chunks
    }

    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (l, r) = self.0.split_at(offset);
        (l.into_series(), r.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append(other.as_ref().as_ref());
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref());
        Ok(())
    }

    fn filter(&self, _filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, _filter).map(|ca| ca.into_series())
    }

    fn take(&self, _indices: &IdxCa) -> PolarsResult<Series> {
        self.0.take(_indices).map(|ca| ca.into_series())
    }

    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> Series {
        self.0.take_unchecked(_idx).into_series()
    }

    fn take_slice(&self, _indices: &[IdxSize]) -> PolarsResult<Series> {
        todo!()
    }

    unsafe fn take_slice_unchecked(&self, _idx: &[IdxSize]) -> Series {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn rechunk(&self) -> Series {
        todo!()
    }

    fn new_from_index(&self, _index: usize, _length: usize) -> Series {
        todo!()
    }

    fn cast(&self, _data_type: &DataType, options: CastOptions) -> PolarsResult<Series> {
        todo!()
    }

    fn get(&self, _index: usize) -> PolarsResult<AnyValue> {
        todo!()
    }

    fn null_count(&self) -> usize {
        todo!()
    }

    fn has_validity(&self) -> bool {
        todo!()
    }

    fn is_null(&self) -> BooleanChunked {
        todo!()
    }

    fn is_not_null(&self) -> BooleanChunked {
        todo!()
    }

    fn reverse(&self) -> Series {
        todo!()
    }

    fn shift(&self, _periods: i64) -> Series {
        todo!()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        todo!()
    }
}
