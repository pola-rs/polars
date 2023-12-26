use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::ArrayRef;
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_utils::IdxSize;

use crate::datatypes::IdxCa;
use crate::error::PolarsResult;
use crate::prelude::explode::ExplodeByOffsets;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use crate::series::*;

impl Series {
    pub fn new_null(name: &str, len: usize) -> Series {
        NullChunked::new(Arc::from(name), len).into_series()
    }
}

#[derive(Clone)]
pub struct NullChunked {
    pub(crate) name: Arc<str>,
    length: IdxSize,
    // we still need chunks as many series consumers expect
    // chunks to be there
    chunks: Vec<ArrayRef>,
}

impl NullChunked {
    pub(crate) fn new(name: Arc<str>, len: usize) -> Self {
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
    fn compute_len(&mut self) {
        fn inner(chunks: &[ArrayRef]) -> usize {
            match chunks.len() {
                // fast path
                1 => chunks[0].len(),
                _ => chunks.iter().fold(0, |acc, arr| acc + arr.len()),
            }
        }
        self.length = IdxSize::try_from(inner(&self.chunks)).expect(LENGTH_LIMIT_MSG);
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(Field::new(self.name(), DataType::Null))
    }

    #[allow(unused)]
    fn _set_flags(&mut self, flags: Settings) {}

    fn _dtype(&self) -> &DataType {
        &DataType::Null
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, _mask: &BooleanChunked, _other: &Series) -> PolarsResult<Series> {
        Ok(self.clone().into_series())
    }
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        ExplodeByOffsets::explode_by_offsets(self, offsets)
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> PolarsResult<GroupsProxy> {
        Ok(if self.is_empty() {
            GroupsProxy::default()
        } else {
            GroupsProxy::Slice {
                groups: vec![[0, self.length]],
                rolling: false,
            }
        })
    }

    fn _get_flags(&self) -> Settings {
        Settings::empty()
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
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        &mut self.chunks
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

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), indices.len()).into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        NullChunked::new(self.name.clone(), indices.len()).into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), indices.len()).into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        NullChunked::new(self.name.clone(), indices.len()).into_series()
    }

    fn len(&self) -> usize {
        self.length as usize
    }

    fn has_validity(&self) -> bool {
        true
    }

    fn rechunk(&self) -> Series {
        NullChunked::new(self.name.clone(), self.len()).into_series()
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
        let (chunks, len) = chunkops::slice(&self.chunks, offset, length, self.len());
        NullChunked {
            name: self.name.clone(),
            length: len as IdxSize,
            chunks,
        }
        .into_series()
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
        polars_ensure!(other.dtype() == &DataType::Null, ComputeError: "expected null dtype");
        // we don't create a new null array to keep probability of aligned chunks higher
        self.chunks.extend(other.chunks().iter().cloned());
        self.length += other.len() as IdxSize;
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        *self = NullChunked::new(self.name.clone(), self.len() + other.len());
        Ok(())
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(self.clone())
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
