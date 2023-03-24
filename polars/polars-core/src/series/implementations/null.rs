use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::prelude::ArrayRef;
use polars_utils::IdxSize;

use crate::datatypes::IdxCa;
use crate::error::PolarsResult;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use crate::series::*;
use crate::utils::slice_offsets;

impl Series {
    pub fn new_null(name: &str, len: usize) -> Series {
        NullChunked::new(Arc::from(name), len).into_series()
    }
}

#[derive(Clone)]
pub struct NullChunked {
    name: Arc<str>,
    length: IdxSize,
    // we still need chunks as many series consumers expect
    // chunks to be there
    chunks: Vec<ArrayRef>,
}

impl NullChunked {
    fn new(name: Arc<str>, len: usize) -> Self {
        Self {
            name,
            length: len as IdxSize,
            chunks: vec![Box::new(arrow::array::NullArray::new(
                ArrowDataType::Null,
                len,
            ))],
        }
    }
}
impl PrivateSeriesNumeric for NullChunked {}

impl PrivateSeries for NullChunked {
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(Field::new(self.name(), DataType::Null))
    }

    fn _dtype(&self) -> &DataType {
        &DataType::Null
    }

    fn zip_with_same_type(&self, _mask: &BooleanChunked, _other: &Series) -> PolarsResult<Series> {
        Ok(self.clone().into_series())
    }
}

impl SeriesTrait for NullChunked {
    fn name(&self) -> &str {
        self.name.as_ref()
    }

    fn rename(&mut self, name: &str) {
        self.name = Arc::from(name)
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    fn chunk_lengths(&self) -> ChunkIdIter {
        self.chunks.iter().map(|chunk| chunk.len())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], _sorted: IsSorted) -> Series {
        NullChunked::new(self.name.clone(), by.len()).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        NullChunked::new(self.name.clone(), by.len()).into_series()
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), iter.size_hint().0).into_series())
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        NullChunked::new(self.name.clone(), iter.size_hint().0).into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), idx.len()).into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        NullChunked::new(self.name.clone(), iter.size_hint().0).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), indices.len()).into_series())
    }

    fn len(&self) -> usize {
        self.length as usize
    }

    fn take_every(&self, n: usize) -> Series {
        NullChunked::new(self.name.clone(), self.len() / n).into_series()
    }

    fn has_validity(&self) -> bool {
        true
    }

    fn rechunk(&self) -> Series {
        self.clone().into_series()
    }

    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        Ok(Series::full_null(self.name.as_ref(), self.len(), data_type))
    }

    fn null_count(&self) -> usize {
        self.len()
    }

    fn new_from_index(&self, _index: usize, length: usize) -> Series {
        NullChunked::new(self.name.clone(), length).into_series()
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        polars_ensure!(index < self.len(), oob = index, self.len());
        Ok(AnyValue::Null)
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        let (_, length) = slice_offsets(offset, length, self.len());
        NullChunked::new(self.name.clone(), length).into_series()
    }

    fn is_null(&self) -> BooleanChunked {
        BooleanChunked::full(self.name(), true, self.len())
    }

    fn is_not_null(&self) -> BooleanChunked {
        BooleanChunked::full(self.name(), false, self.len())
    }

    fn reverse(&self) -> Series {
        self.clone().into_series()
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        let len = filter.sum().unwrap_or(0);
        Ok(NullChunked::new(self.name.clone(), len as usize).into_series())
    }

    fn shift(&self, _periods: i64) -> Series {
        self.clone().into_series()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        *self = NullChunked::new(self.name.clone(), self.len() + other.len());
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        self.append(other)
    }
}

unsafe impl IntoSeries for NullChunked {
    fn into_series(self) -> Series
    where
        Self: Sized,
    {
        Series(Arc::new(self))
    }
}
