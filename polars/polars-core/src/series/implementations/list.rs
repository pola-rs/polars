use std::any::Any;
use std::borrow::Cow;

use super::{private, IntoSeries, SeriesTrait};
use crate::chunked_array::comparison::*;
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::AsSinglePtr;
use crate::fmt::FmtList;
use crate::frame::groupby::*;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::series::IsSorted;

impl private::PrivateSeries for SeriesWrap<ListChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.ref_field().data_type()
    }
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0.explode_by_offsets(offsets)
    }

    fn _set_sorted_flag(&mut self, is_sorted: IsSorted) {
        self.0.set_sorted_flag(is_sorted)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref()).map(|ca| ca.into_series())
    }

    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
    }
}

impl SeriesTrait for SeriesWrap<ListChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name);
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
    fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        if self.0.dtype() == other.dtype() {
            self.0.append(other.as_ref().as_ref())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }
    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        if self.0.dtype() == other.dtype() {
            self.0.extend(other.as_ref().as_ref())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot extend Series; data types don't match".into(),
            ))
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        self.0.take_chunked_unchecked(by, sorted).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        self.0.take_opt_chunked_unchecked(by).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        let indices = if indices.chunks.len() > 1 {
            Cow::Owned(indices.rechunk())
        } else {
            Cow::Borrowed(indices)
        };
        Ok(ChunkTake::take(&self.0, (&*indices).into())?.into_series())
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
    }

    fn take_every(&self, n: usize) -> Series {
        self.0.take_every(n).into_series()
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        let idx = if idx.chunks.len() > 1 {
            Cow::Owned(idx.rechunk())
        } else {
            Cow::Borrowed(idx)
        };
        Ok(ChunkTake::take_unchecked(&self.0, (&*idx).into()).into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
    }

    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> PolarsResult<Series> {
        Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0.rechunk().into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::new_from_index(&self.0, index, length).into_series()
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
        ChunkReverse::reverse(&self.0).into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        ChunkShift::shift(&self.0, periods).into_series()
    }

    fn _sum_as_series(&self) -> Series {
        ChunkAggSeries::sum_as_series(&self.0)
    }
    fn max_as_series(&self) -> Series {
        ChunkAggSeries::max_as_series(&self.0)
    }
    fn min_as_series(&self) -> Series {
        ChunkAggSeries::min_as_series(&self.0)
    }
    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }
    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }
    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    /// Get a hold to self as `Any` trait reference.
    /// Only implemented for ObjectType
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}
