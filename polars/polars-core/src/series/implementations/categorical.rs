use std::borrow::Cow;

use ahash::RandomState;
use polars_arrow::prelude::QuantileInterpolOptions;

use super::{private, IntoSeries, SeriesTrait, *};
use crate::chunked_array::comparison::*;
use crate::chunked_array::ops::compare_inner::{IntoPartialOrdInner, PartialOrdInner};
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::AsSinglePtr;
use crate::fmt::FmtList;
use crate::frame::groupby::*;
use crate::frame::hash_join::ZipOuterJoinColumn;
#[cfg(feature = "is_in")]
use crate::frame::hash_join::_check_categorical_src;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;

unsafe impl IntoSeries for CategoricalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl SeriesWrap<CategoricalChunked> {
    fn finish_with_state(&self, keep_fast_unique: bool, cats: UInt32Chunked) -> CategoricalChunked {
        let mut out = unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(cats, self.0.get_rev_map().clone())
        };
        if keep_fast_unique && self.0.can_fast_unique() {
            out.set_fast_unique(true)
        }
        out.set_lexical_sorted(self.0.use_lexical_sort());
        out
    }

    fn with_state<F>(&self, keep_fast_unique: bool, apply: F) -> CategoricalChunked
    where
        F: Fn(&UInt32Chunked) -> UInt32Chunked,
    {
        let cats = apply(self.0.logical());
        self.finish_with_state(keep_fast_unique, cats)
    }

    fn try_with_state<'a, F>(
        &'a self,
        keep_fast_unique: bool,
        apply: F,
    ) -> PolarsResult<CategoricalChunked>
    where
        F: for<'b> Fn(&'a UInt32Chunked) -> PolarsResult<UInt32Chunked>,
    {
        let cats = apply(self.0.logical())?;
        Ok(self.finish_with_state(keep_fast_unique, cats))
    }
}

impl private::PrivateSeries for SeriesWrap<CategoricalChunked> {
    fn compute_len(&mut self) {
        self.0.logical_mut().compute_len()
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        // TODO! explode by offset should return concrete type
        self.with_state(true, |cats| {
            cats.explode_by_offsets(offsets).u32().unwrap().clone()
        })
        .into_series()
    }

    fn _set_sorted_flag(&mut self, is_sorted: IsSorted) {
        self.0.logical_mut().set_sorted_flag(is_sorted)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.logical().equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        self.0
            .zip_with(mask, other.categorical()?)
            .map(|ca| ca.into_series())
    }
    fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
        (&self.0).into_partial_ord_inner()
    }

    fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.logical().vec_hash(random_state, buf);
        Ok(())
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) -> PolarsResult<()> {
        self.0.logical().vec_hash_combine(build_hasher, hashes);
        Ok(())
    }

    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        self.0
            .logical()
            .agg_list(groups)
            .cast(&DataType::List(Box::new(self.dtype().clone())))
            .unwrap()
    }

    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        let new_rev_map = self
            .0
            .merge_categorical_map(right_column.categorical().unwrap())
            .unwrap();
        let left = self.0.logical();
        let right = right_column
            .categorical()
            .unwrap()
            .logical()
            .clone()
            .into_series();

        let cats = left.zip_outer_join_column(&right, opt_join_tuples);
        let cats = cats.u32().unwrap().clone();

        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(cats, new_rev_map).into_series()
        }
    }
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        self.0.logical().group_tuples(multithreaded, sorted)
    }

    #[cfg(feature = "sort_multiple")]
    fn arg_sort_multiple(&self, by: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        self.0.arg_sort_multiple(by, reverse)
    }
}

impl SeriesTrait for SeriesWrap<CategoricalChunked> {
    fn is_sorted_flag(&self) -> IsSorted {
        if self.0.logical().is_sorted_flag() {
            IsSorted::Ascending
        } else if self.0.logical().is_sorted_reverse_flag() {
            IsSorted::Descending
        } else {
            IsSorted::Not
        }
    }

    fn rename(&mut self, name: &str) {
        self.0.logical_mut().rename(name);
    }

    fn chunk_lengths(&self) -> ChunkIdIter {
        self.0.logical().chunk_id()
    }
    fn name(&self) -> &str {
        self.0.logical().name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.logical().chunks()
    }
    fn shrink_to_fit(&mut self) {
        self.0.logical_mut().shrink_to_fit()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.with_state(false, |cats| cats.slice(offset, length))
            .into_series()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        if self.0.dtype() == other.dtype() {
            self.0.append(other.categorical().unwrap())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }
    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        if self.0.dtype() == other.dtype() {
            let other = other.categorical()?;
            self.0.logical_mut().extend(other.logical());
            let new_rev_map = self.0.merge_categorical_map(other)?;
            // safety:
            // rev_maps are merged
            unsafe { self.0.set_rev_map(new_rev_map, false) };
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot extend Series; data types don't match".into(),
            ))
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        self.try_with_state(false, |cats| cats.filter(filter))
            .map(|ca| ca.into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        let cats = self.0.logical().take_chunked_unchecked(by, sorted);
        self.finish_with_state(false, cats).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        let cats = self.0.logical().take_opt_chunked_unchecked(by);
        self.finish_with_state(false, cats).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        let indices = if indices.chunks.len() > 1 {
            Cow::Owned(indices.rechunk())
        } else {
            Cow::Borrowed(indices)
        };
        self.try_with_state(false, |cats| cats.take((&*indices).into()))
            .map(|ca| ca.into_series())
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        let cats = self.0.logical().take(iter.into())?;
        Ok(self.finish_with_state(false, cats).into_series())
    }

    fn take_every(&self, n: usize) -> Series {
        self.with_state(true, |cats| cats.take_every(n))
            .into_series()
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        let cats = self.0.logical().take_unchecked(iter.into());
        self.finish_with_state(false, cats).into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        let idx = if idx.chunks.len() > 1 {
            Cow::Owned(idx.rechunk())
        } else {
            Cow::Borrowed(idx)
        };
        Ok(self
            .with_state(false, |cats| cats.take_unchecked((&*idx).into()))
            .into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        let cats = self.0.logical().take_unchecked(iter.into());
        self.finish_with_state(false, cats).into_series()
    }

    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> PolarsResult<Series> {
        let cats = self.0.logical().take(iter.into())?;
        Ok(self.finish_with_state(false, cats).into_series())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.with_state(true, |ca| ca.rechunk()).into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.with_state(true, |cats| cats.new_from_index(index, length))
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

    fn sort_with(&self, options: SortOptions) -> Series {
        self.0.sort_with(options).into_series()
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }

    fn null_count(&self) -> usize {
        self.0.logical().null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.logical().has_validity()
    }

    fn unique(&self) -> PolarsResult<Series> {
        self.0.unique().map(|ca| ca.into_series())
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.n_unique()
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.logical().arg_unique()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.logical().is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.logical().is_not_null()
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        self.0.logical().is_unique()
    }

    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        self.0.logical().is_duplicated()
    }

    fn reverse(&self) -> Series {
        self.with_state(true, |cats| cats.reverse()).into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.logical_mut().as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        self.with_state(false, |ca| ca.shift(periods)).into_series()
    }

    fn _sum_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.0.logical().name(), 1).into_series()
    }
    fn max_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.0.logical().name(), 1).into_series()
    }
    fn min_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.0.logical().name(), 1).into_series()
    }
    fn median_as_series(&self) -> Series {
        CategoricalChunked::full_null(self.0.logical().name(), 1).into_series()
    }
    fn var_as_series(&self, _ddof: u8) -> Series {
        CategoricalChunked::full_null(self.0.logical().name(), 1).into_series()
    }
    fn std_as_series(&self, _ddof: u8) -> Series {
        CategoricalChunked::full_null(self.0.logical().name(), 1).into_series()
    }
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        Ok(CategoricalChunked::full_null(self.0.logical().name(), 1).into_series())
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }
    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    #[cfg(feature = "is_in")]
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        _check_categorical_src(self.dtype(), other.dtype())?;
        self.0.logical().is_in(&other.to_physical_repr())
    }
    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        let out = self.0.logical().repeat_by(by);
        let casted = out
            .cast(&DataType::List(Box::new(self.dtype().clone())))
            .unwrap();
        casted.list().unwrap().clone()
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> PolarsResult<Series> {
        Ok(CategoricalChunked::full_null(self.0.logical().name(), 1).into_series())
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<CategoricalChunked> {
    fn bit_repr_is_large(&self) -> bool {
        false
    }
    fn bit_repr_small(&self) -> UInt32Chunked {
        self.0.logical().clone()
    }
}
