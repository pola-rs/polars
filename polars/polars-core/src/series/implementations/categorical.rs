use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use super::*;
use crate::chunked_array::comparison::*;
use crate::chunked_array::{
    ops::{
        compare_inner::{IntoPartialEqInner, IntoPartialOrdInner, PartialEqInner, PartialOrdInner},
        explode::ExplodeByOffsets,
        ChunkFullNull,
    },
    AsSinglePtr, ChunkIdIter,
};
use crate::fmt::FmtList;
use crate::frame::groupby::*;
use crate::frame::hash_join::{HashJoin, ZipOuterJoinColumn};
use crate::prelude::*;
#[cfg(feature = "checked_arithmetic")]
use crate::series::arithmetic::checked::NumOpsDispatchChecked;
use crate::series::implementations::SeriesWrap;
use ahash::RandomState;
use arrow::array::ArrayRef;
use polars_arrow::prelude::QuantileInterpolOptions;
use std::borrow::Cow;

impl IntoSeries for CategoricalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeries for SeriesWrap<CategoricalChunked> {
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.ref_field().data_type()
    }

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0.explode_by_offsets(offsets)
    }

    fn set_sorted(&mut self, reverse: bool) {
        self.0.set_sorted(reverse)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref()).map(|ca| ca.into_series())
    }
    fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
        (&self.0).into_partial_eq_inner()
    }
    fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
        (&self.0).into_partial_ord_inner()
    }

    fn vec_hash(&self, random_state: RandomState) -> Vec<u64> {
        self.0.vec_hash(random_state)
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
        self.0.vec_hash_combine(build_hasher, hashes)
    }

    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        self.0.agg_list(groups)
    }

    fn hash_join_inner(&self, other: &Series) -> Vec<(IdxSize, IdxSize)> {
        HashJoin::hash_join_inner(&self.0, other.as_ref().as_ref())
    }
    fn hash_join_left(&self, other: &Series) -> Vec<(IdxSize, Option<IdxSize>)> {
        HashJoin::hash_join_left(&self.0, other.as_ref().as_ref())
    }
    fn hash_join_outer(&self, other: &Series) -> Vec<(Option<IdxSize>, Option<IdxSize>)> {
        HashJoin::hash_join_outer(&self.0, other.as_ref().as_ref())
    }
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        let categorical_map_out = Some(
            self.0
                .merge_categorical_map(right_column.categorical().unwrap()),
        );
        let s_left = self.0.cast(&DataType::UInt32).unwrap();
        let ca = s_left.u32().unwrap();

        let right = right_column.cast(&DataType::UInt32).unwrap();
        let out = ZipOuterJoinColumn::zip_outer_join_column(ca, &right, opt_join_tuples)
            .cast(&DataType::Categorical)
            .unwrap();
        let mut out = out.categorical().unwrap().clone();
        out.categorical_map = categorical_map_out;
        out.into_series()
    }
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
    }

    #[cfg(feature = "sort_multiple")]
    fn argsort_multiple(&self, by: &[Series], reverse: &[bool]) -> Result<IdxCa> {
        self.0.argsort_multiple(by, reverse)
    }
}

impl SeriesTrait for SeriesWrap<CategoricalChunked> {
    fn is_sorted(&self) -> IsSorted {
        if self.0.is_sorted() {
            IsSorted::Ascending
        } else if self.0.is_sorted_reverse() {
            IsSorted::Descending
        } else {
            IsSorted::Not
        }
    }

    #[cfg(feature = "interpolate")]
    fn interpolate(&self) -> Series {
        self.0.interpolate().into_series()
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

    fn categorical(&self) -> Result<&CategoricalChunked> {
        if matches!(self.0.dtype(), DataType::Categorical) {
            unsafe { Ok(&*(self as *const dyn SeriesTrait as *const CategoricalChunked)) }
        } else {
            Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot unpack Series: {:?} of type {:?} into categorical",
                    self.name(),
                    self.dtype(),
                )
                .into(),
            ))
        }
    }

    fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        self.0.append_array(other)
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn append(&mut self, other: &Series) -> Result<()> {
        if self.0.dtype() == other.dtype() {
            let other = other.categorical()?;
            self.0.append(other);
            self.0.categorical_map = Some(self.0.merge_categorical_map(other));
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }
    fn extend(&mut self, other: &Series) -> Result<()> {
        if self.0.dtype() == other.dtype() {
            let other = other.categorical()?;
            self.0.extend(other);
            self.0.categorical_map = Some(self.0.merge_categorical_map(other));
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot extend Series; data types don't match".into(),
            ))
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    fn take(&self, indices: &IdxCa) -> Result<Series> {
        let indices = if indices.chunks.len() > 1 {
            Cow::Owned(indices.rechunk())
        } else {
            Cow::Borrowed(indices)
        };
        Ok(ChunkTake::take(&self.0, (&*indices).into())?.into_series())
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
        Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
    }

    fn take_every(&self, n: usize) -> Series {
        self.0.take_every(n).into_series()
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> Result<Series> {
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
    fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
        Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        ChunkOps::rechunk(&self.0).into_series()
    }

    fn expand_at_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::expand_at_index(&self.0, index, length).into_series()
    }

    fn cast(&self, data_type: &DataType) -> Result<Series> {
        self.0.cast(data_type)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        ChunkUnique::value_counts(&self.0)
    }

    fn get(&self, index: usize) -> AnyValue {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn sort_with(&self, options: SortOptions) -> Series {
        ChunkSort::sort_with(&self.0, options).into_series()
    }

    fn argsort(&self, reverse: bool) -> IdxCa {
        ChunkSort::argsort(&self.0, reverse)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    fn unique(&self) -> Result<Series> {
        ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
    }

    fn n_unique(&self) -> Result<usize> {
        ChunkUnique::n_unique(&self.0)
    }

    fn arg_unique(&self) -> Result<IdxCa> {
        ChunkUnique::arg_unique(&self.0)
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        ChunkUnique::is_unique(&self.0)
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        ChunkUnique::is_duplicated(&self.0)
    }

    fn reverse(&self) -> Series {
        ChunkReverse::reverse(&self.0).into_series()
    }

    fn as_single_ptr(&mut self) -> Result<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        ChunkShift::shift(&self.0, periods).into_series()
    }

    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Series> {
        ChunkFillNull::fill_null(&self.0, strategy).map(|ca| ca.into_series())
    }

    fn _sum_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.name(), 1).into_series()
    }
    fn max_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.name(), 1).into_series()
    }
    fn min_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.name(), 1).into_series()
    }
    fn median_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.name(), 1).into_series()
    }
    fn var_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.name(), 1).into_series()
    }
    fn std_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.name(), 1).into_series()
    }
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
        Ok(CategoricalChunked::full_null(self.name(), 1).into_series())
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }
    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    #[cfg(feature = "is_in")]
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        IsIn::is_in(&self.0, other)
    }
    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        RepeatBy::repeat_by(&self.0, by)
    }

    #[cfg(feature = "checked_arithmetic")]
    fn checked_div(&self, rhs: &Series) -> Result<Series> {
        self.0.checked_div(rhs)
    }

    #[cfg(feature = "is_first")]
    fn is_first(&self) -> Result<BooleanChunked> {
        self.0.is_first()
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Series> {
        Ok(self.0.mode()?.into_series())
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<CategoricalChunked> {
    fn bit_repr_is_large(&self) -> bool {
        CategoricalChunked::bit_repr_is_large()
    }
    fn bit_repr_large(&self) -> UInt64Chunked {
        self.0.bit_repr_large()
    }
    fn bit_repr_small(&self) -> UInt32Chunked {
        self.0.bit_repr_small()
    }
}
