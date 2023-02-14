use std::borrow::Cow;

use ahash::RandomState;

use super::{private, IntoSeries, SeriesTrait, *};
use crate::chunked_array::comparison::*;
use crate::chunked_array::ops::compare_inner::{
    IntoPartialEqInner, IntoPartialOrdInner, PartialEqInner, PartialOrdInner,
};
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::AsSinglePtr;
use crate::fmt::FmtList;
use crate::frame::groupby::*;
use crate::frame::hash_join::ZipOuterJoinColumn;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;

impl private::PrivateSeries for SeriesWrap<Utf8Chunked> {
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
    fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
        (&self.0).into_partial_eq_inner()
    }
    fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
        (&self.0).into_partial_ord_inner()
    }

    fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.vec_hash(random_state, buf);
        Ok(())
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) -> PolarsResult<()> {
        self.0.vec_hash_combine(build_hasher, hashes);
        Ok(())
    }

    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_min(groups)
    }

    unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_max(groups)
    }

    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        ZipOuterJoinColumn::zip_outer_join_column(&self.0, right_column, opt_join_tuples)
    }
    fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
        NumOpsDispatch::subtract(&self.0, rhs)
    }
    fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
        NumOpsDispatch::add_to(&self.0, rhs)
    }
    fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
        NumOpsDispatch::multiply(&self.0, rhs)
    }
    fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
        NumOpsDispatch::divide(&self.0, rhs)
    }
    fn remainder(&self, rhs: &Series) -> PolarsResult<Series> {
        NumOpsDispatch::remainder(&self.0, rhs)
    }
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
    }

    #[cfg(feature = "sort_multiple")]
    fn arg_sort_multiple(&self, by: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        self.0.arg_sort_multiple(by, reverse)
    }
}

impl SeriesTrait for SeriesWrap<Utf8Chunked> {
    fn is_sorted_flag(&self) -> IsSorted {
        if self.0.is_sorted_flag() {
            IsSorted::Ascending
        } else if self.0.is_sorted_reverse_flag() {
            IsSorted::Descending
        } else {
            IsSorted::Not
        }
    }

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
            // todo! add object
            self.0.append(other.as_ref().as_ref());
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        if self.0.dtype() == other.dtype() {
            self.0.extend(other.as_ref().as_ref());
            Ok(())
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

        let mut out = ChunkTake::take_unchecked(&self.0, (&*idx).into());

        if self.0.is_sorted_flag() && (idx.is_sorted_flag() || idx.is_sorted_reverse_flag()) {
            out.set_sorted_flag(idx.is_sorted_flag2())
        }

        Ok(out.into_series())
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

    fn sort_with(&self, options: SortOptions) -> Series {
        ChunkSort::sort_with(&self.0, options).into_series()
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        ChunkSort::arg_sort(&self.0, options)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    fn unique(&self) -> PolarsResult<Series> {
        ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        ChunkUnique::n_unique(&self.0)
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        ChunkUnique::arg_unique(&self.0)
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        ChunkUnique::is_unique(&self.0)
    }

    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        ChunkUnique::is_duplicated(&self.0)
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

    #[cfg(feature = "is_in")]
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        IsIn::is_in(&self.0, other)
    }
    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        RepeatBy::repeat_by(&self.0, by)
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> PolarsResult<Series> {
        Ok(self.0.mode()?.into_series())
    }

    #[cfg(feature = "concat_str")]
    fn str_concat(&self, delimiter: &str) -> Utf8Chunked {
        self.0.str_concat(delimiter)
    }
}
