use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

use ahash::RandomState;

use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::chunked_array::comparison::*;
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::AsSinglePtr;
use crate::fmt::FmtList;
use crate::frame::groupby::*;
use crate::frame::hash_join::*;
use crate::prelude::*;

unsafe impl IntoSeries for DurationChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DurationChunked> {
    fn bit_repr_is_large(&self) -> bool {
        true
    }
    fn bit_repr_large(&self) -> UInt64Chunked {
        self.0.bit_repr_large()
    }
}

impl private::PrivateSeries for SeriesWrap<DurationChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0
            .explode_by_offsets(offsets)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummax(&self, reverse: bool) -> Series {
        self.0
            .cummax(reverse)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummin(&self, reverse: bool) -> Series {
        self.0
            .cummin(reverse)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn _set_sorted_flag(&mut self, is_sorted: IsSorted) {
        self.0.deref_mut().set_sorted_flag(is_sorted)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let other = other.to_physical_repr().into_owned();
        self.0
            .zip_with(mask, other.as_ref().as_ref())
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.vec_hash(random_state, buf);
        Ok(())
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) -> PolarsResult<()> {
        self.0.vec_hash_combine(build_hasher, hashes);
        Ok(())
    }

    unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.0
            .agg_min(groups)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.0
            .agg_max(groups)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        self.0
            .agg_sum(groups)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        self.0
            .agg_std(groups, ddof)
            // cast f64 back to physical type
            .cast(&DataType::Int64)
            .unwrap()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        self.0
            .agg_var(groups, ddof)
            // cast f64 back to physical type
            .cast(&DataType::Int64)
            .unwrap()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        self.0
            .agg_list(groups)
            .cast(&DataType::List(Box::new(self.dtype().clone())))
            .unwrap()
    }

    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        let right_column = right_column.to_physical_repr().into_owned();
        self.0
            .zip_outer_join_column(&right_column, opt_join_tuples)
            .into_duration(self.0.time_unit())
            .into_series()
    }
    fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Duration(tu), DataType::Duration(tur)) => {
                assert_eq!(tu, tur);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs.subtract(&rhs)?.into_duration(*tu).into_series())
            }
            (dtl, dtr) => Err(PolarsError::ComputeError(
                format!("cannot do subtraction on these date types: {dtl:?}, {dtr:?}",).into(),
            )),
        }
    }
    fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Duration(tu), DataType::Duration(tur)) => {
                assert_eq!(tu, tur);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs.add_to(&rhs)?.into_duration(*tu).into_series())
            }
            (DataType::Duration(tu), DataType::Datetime(tur, tz)) => {
                assert_eq!(tu, tur);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs
                    .add_to(&rhs)?
                    .into_datetime(*tu, tz.clone())
                    .into_series())
            }
            (dtl, dtr) => Err(PolarsError::ComputeError(
                format!("cannot do addition on these date types: {dtl:?}, {dtr:?}",).into(),
            )),
        }
    }
    fn multiply(&self, _rhs: &Series) -> PolarsResult<Series> {
        Err(PolarsError::ComputeError(
            "cannot do multiplication on logical".into(),
        ))
    }
    fn divide(&self, _rhs: &Series) -> PolarsResult<Series> {
        self.0.deref().divide(_rhs.to_physical_repr().as_ref())
    }
    fn remainder(&self, _rhs: &Series) -> PolarsResult<Series> {
        Err(PolarsError::ComputeError(
            "cannot do remainder operation on logical".into(),
        ))
    }
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        self.0.group_tuples(multithreaded, sorted)
    }
    #[cfg(feature = "sort_multiple")]
    fn arg_sort_multiple(&self, by: &[Series], reverse: &[bool]) -> PolarsResult<IdxCa> {
        self.0.deref().arg_sort_multiple(by, reverse)
    }
}

impl SeriesTrait for SeriesWrap<DurationChunked> {
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
        self.0
            .slice(offset, length)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn mean(&self) -> Option<f64> {
        self.0.mean()
    }

    fn median(&self) -> Option<f64> {
        self.0.median()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        if self.0.dtype() == other.dtype() {
            let other = other.to_physical_repr().into_owned();
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
            let other = other.to_physical_repr();
            self.0.extend(other.as_ref().as_ref().as_ref());
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot extend Series; data types don't match".into(),
            ))
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        self.0
            .filter(filter)
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        let ca = self.0.deref().take_chunked_unchecked(by, sorted);
        ca.into_duration(self.0.time_unit()).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        let ca = self.0.deref().take_opt_chunked_unchecked(by);
        ca.into_duration(self.0.time_unit()).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), indices.into())
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), iter.into())
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    fn take_every(&self, n: usize) -> Series {
        self.0
            .take_every(n)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_duration(self.0.time_unit())
            .into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        let mut out = ChunkTake::take_unchecked(self.0.deref(), idx.into());

        if self.0.is_sorted_flag() && (idx.is_sorted_flag() || idx.is_sorted_reverse_flag()) {
            out.set_sorted_flag(idx.is_sorted_flag2())
        }

        Ok(out.into_duration(self.0.time_unit()).into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> PolarsResult<Series> {
        ChunkTake::take(self.0.deref(), iter.into())
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0
            .rechunk()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.0
            .new_from_index(index, length)
            .into_duration(self.0.time_unit())
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
        self.0
            .sort_with(options)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    fn unique(&self) -> PolarsResult<Series> {
        self.0
            .unique()
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.n_unique()
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.arg_unique()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        self.0.is_unique()
    }

    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        self.0.is_duplicated()
    }

    fn reverse(&self) -> Series {
        self.0
            .reverse()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0
            .shift(periods)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn _sum_as_series(&self) -> Series {
        self.0.sum_as_series().into_duration(self.0.time_unit())
    }

    fn max_as_series(&self) -> Series {
        self.0.max_as_series().into_duration(self.0.time_unit())
    }
    fn min_as_series(&self) -> Series {
        self.0.min_as_series().into_duration(self.0.time_unit())
    }
    fn median_as_series(&self) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn var_as_series(&self, _ddof: u8) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn std_as_series(&self, _ddof: u8) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        Ok(Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap())
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn peak_max(&self) -> BooleanChunked {
        self.0.peak_max()
    }

    fn peak_min(&self) -> BooleanChunked {
        self.0.peak_min()
    }
    #[cfg(feature = "is_in")]
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        self.0.is_in(other)
    }
    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        self.0
            .repeat_by(by)
            .cast(&DataType::List(Box::new(DataType::Duration(
                self.0.time_unit(),
            ))))
            .unwrap()
            .list()
            .unwrap()
            .clone()
    }
    #[cfg(feature = "mode")]
    fn mode(&self) -> PolarsResult<Series> {
        self.0
            .mode()
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }
}
