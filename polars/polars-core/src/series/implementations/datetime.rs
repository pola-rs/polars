use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use super::SeriesWrap;
use super::*;
use crate::chunked_array::{ops::explode::ExplodeByOffsets, AsSinglePtr, ChunkIdIter};
use crate::fmt::FmtList;
use crate::frame::{groupby::*, hash_join::*};
use crate::prelude::*;
use ahash::RandomState;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

impl IntoSeries for DatetimeChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DatetimeChunked> {
    fn bit_repr_is_large(&self) -> bool {
        true
    }
    fn bit_repr_large(&self) -> UInt64Chunked {
        self.0.bit_repr_large()
    }
}

impl private::PrivateSeries for SeriesWrap<DatetimeChunked> {
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0
            .explode_by_offsets(offsets)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummax(&self, reverse: bool) -> Series {
        self.0
            .cummax(reverse)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummin(&self, reverse: bool) -> Series {
        self.0
            .cummin(reverse)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn set_sorted(&mut self, reverse: bool) {
        self.0.deref_mut().set_sorted(reverse)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        let other = other.to_physical_repr().into_owned();
        self.0.zip_with(mask, other.as_ref().as_ref()).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn vec_hash(&self, random_state: RandomState) -> Vec<u64> {
        self.0.vec_hash(random_state)
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
        self.0.vec_hash_combine(build_hasher, hashes)
    }

    fn agg_mean(&self, _groups: &GroupsProxy) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_min(&self, groups: &GroupsProxy) -> Option<Series> {
        self.0.agg_min(groups).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn agg_max(&self, groups: &GroupsProxy) -> Option<Series> {
        self.0.agg_max(groups).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn agg_sum(&self, _groups: &GroupsProxy) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_std(&self, _groups: &GroupsProxy) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_var(&self, _groups: &GroupsProxy) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        self.0.agg_list(groups).map(|s| {
            s.cast(&DataType::List(Box::new(self.dtype().clone())))
                .unwrap()
        })
    }

    fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Option<Series> {
        self.0.agg_quantile(groups, quantile, interpol).map(|s| {
            s.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn agg_median(&self, groups: &GroupsProxy) -> Option<Series> {
        self.0.agg_median(groups).map(|s| {
            s.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        let right_column = right_column.to_physical_repr().into_owned();
        self.0
            .zip_outer_join_column(&right_column, opt_join_tuples)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }
    fn subtract(&self, rhs: &Series) -> Result<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Datetime(tu, tz), DataType::Datetime(tur, tzr)) => {
                assert_eq!(tu, tur);
                assert_eq!(tz, tzr);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs.subtract(&rhs)?.into_duration(*tu).into_series())
            }
            (DataType::Datetime(tu, tz), DataType::Duration(tur)) => {
                assert_eq!(tu, tur);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs
                    .subtract(&rhs)?
                    .into_datetime(*tu, tz.clone())
                    .into_series())
            }
            (dtl, dtr) => Err(PolarsError::ComputeError(
                format!(
                    "cannot do subtraction on these date types: {:?}, {:?}",
                    dtl, dtr
                )
                .into(),
            )),
        }
    }
    fn add_to(&self, rhs: &Series) -> Result<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Datetime(tu, tz), DataType::Duration(tur)) => {
                assert_eq!(tu, tur);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs
                    .add_to(&rhs)?
                    .into_datetime(*tu, tz.clone())
                    .into_series())
            }
            (dtl, dtr) => Err(PolarsError::ComputeError(
                format!(
                    "cannot do addition on these date types: {:?}, {:?}",
                    dtl, dtr
                )
                .into(),
            )),
        }
    }
    fn multiply(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do multiplication on logical".into(),
        ))
    }
    fn divide(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do division on logical".into(),
        ))
    }
    fn remainder(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do remainder operation on logical".into(),
        ))
    }
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        self.0.group_tuples(multithreaded, sorted)
    }
    #[cfg(feature = "sort_multiple")]
    fn argsort_multiple(&self, by: &[Series], reverse: &[bool]) -> Result<IdxCa> {
        self.0.deref().argsort_multiple(by, reverse)
    }
}

impl SeriesTrait for SeriesWrap<DatetimeChunked> {
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
        self.0
            .interpolate()
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
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

    fn datetime(&self) -> Result<&DatetimeChunked> {
        unsafe { Ok(&*(self as *const dyn SeriesTrait as *const DatetimeChunked)) }
    }

    fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        self.0.append_array(other)
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0
            .slice(offset, length)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn mean(&self) -> Option<f64> {
        self.0.mean()
    }

    fn median(&self) -> Option<f64> {
        self.0.median()
    }

    fn append(&mut self, other: &Series) -> Result<()> {
        if self.0.dtype() == other.dtype() {
            let other = other.to_physical_repr();
            self.0.append(other.as_ref().as_ref().as_ref());
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }

    fn extend(&mut self, other: &Series) -> Result<()> {
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

    fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
        self.0.filter(filter).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn take(&self, indices: &IdxCa) -> Result<Series> {
        ChunkTake::take(self.0.deref(), indices.into()).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
        ChunkTake::take(self.0.deref(), iter.into()).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn take_every(&self, n: usize) -> Series {
        self.0
            .take_every(n)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> Result<Series> {
        Ok(ChunkTake::take_unchecked(self.0.deref(), idx.into())
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(self.0.deref(), iter.into())
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
        ChunkTake::take(self.0.deref(), iter.into()).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0
            .rechunk()
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn expand_at_index(&self, index: usize, length: usize) -> Series {
        self.0
            .expand_at_index(index, length)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn cast(&self, data_type: &DataType) -> Result<Series> {
        self.0.cast(data_type)
    }

    fn to_dummies(&self) -> Result<DataFrame> {
        self.0.to_dummies()
    }

    fn get(&self, index: usize) -> AnyValue {
        self.0.get_any_value(index)
    }

    #[inline]
    #[cfg(feature = "private")]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0
            .get_any_value_unchecked(index)
            .into_datetime(self.0.time_unit(), self.0.time_zone())
    }

    fn sort_with(&self, options: SortOptions) -> Series {
        self.0
            .sort_with(options)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn argsort(&self, options: SortOptions) -> IdxCa {
        self.0.argsort(options)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    fn unique(&self) -> Result<Series> {
        self.0.unique().map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn n_unique(&self) -> Result<usize> {
        self.0.n_unique()
    }

    fn arg_unique(&self) -> Result<IdxCa> {
        self.0.arg_unique()
    }

    fn arg_min(&self) -> Option<usize> {
        self.0.arg_min()
    }

    fn arg_max(&self) -> Option<usize> {
        self.0.arg_max()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        self.0.is_unique()
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        self.0.is_duplicated()
    }

    fn reverse(&self) -> Series {
        self.0
            .reverse()
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn as_single_ptr(&mut self) -> Result<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0
            .shift(periods)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Series> {
        self.0.fill_null(strategy).map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }

    fn _sum_as_series(&self) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn max_as_series(&self) -> Series {
        self.0
            .max_as_series()
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
    }
    fn min_as_series(&self) -> Series {
        self.0
            .min_as_series()
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
    }
    fn median_as_series(&self) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn var_as_series(&self) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn std_as_series(&self) -> Series {
        Int32Chunked::full_null(self.name(), 1)
            .cast(self.dtype())
            .unwrap()
    }
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
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
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        self.0.is_in(other)
    }
    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: &IdxCa) -> ListChunked {
        self.0
            .repeat_by(by)
            .cast(&DataType::List(Box::new(DataType::Datetime(
                self.0.time_unit(),
                self.0.time_zone().clone(),
            ))))
            .unwrap()
            .list()
            .unwrap()
            .clone()
    }
    #[cfg(feature = "is_first")]
    fn is_first(&self) -> Result<BooleanChunked> {
        self.0.is_first()
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Series> {
        self.0.mode().map(|ca| {
            ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone())
                .into_series()
        })
    }
}
