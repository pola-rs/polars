//! This module exists to reduce compilation times.
//!
//! All the data types are backed by a physical type in memory e.g. Date -> i32, Datetime-> i64.
//!
//! Series lead to code implementations of all traits. Whereas there are a lot of duplicates due to
//! data types being backed by the same physical type. In this module we reduce compile times by
//! opting for a little more run time cost. We cast to the physical type -> apply the operation and
//! (depending on the result) cast back to the original type
//!
use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use super::SeriesWrap;
use super::*;
use crate::chunked_array::{
    ops::{explode::ExplodeByOffsets, ToBitRepr},
    AsSinglePtr, ChunkIdIter,
};
use crate::fmt::FmtList;
use crate::frame::{groupby::*, hash_join::*};
use crate::prelude::*;
use ahash::RandomState;
use polars_arrow::prelude::QuantileInterpolOptions;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

macro_rules! impl_dyn_series {
    ($ca: ident, $into_logical: ident) => {
        impl IntoSeries for $ca {
            fn into_series(self) -> Series {
                Series(Arc::new(SeriesWrap(self)))
            }
        }

        impl private::PrivateSeries for SeriesWrap<$ca> {
            fn _field(&self) -> Cow<Field> {
                Cow::Owned(self.0.field())
            }
            fn _dtype(&self) -> &DataType {
                self.0.dtype()
            }

            fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
                self.0
                    .explode_by_offsets(offsets)
                    .$into_logical()
                    .into_series()
            }

            #[cfg(feature = "cum_agg")]
            fn _cummax(&self, reverse: bool) -> Series {
                self.0.cummax(reverse).$into_logical().into_series()
            }

            #[cfg(feature = "cum_agg")]
            fn _cummin(&self, reverse: bool) -> Series {
                self.0.cummin(reverse).$into_logical().into_series()
            }

            fn set_sorted(&mut self, reverse: bool) {
                self.0.deref_mut().set_sorted(reverse)
            }

            #[cfg(feature = "zip_with")]
            fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
                let other = other.to_physical_repr().into_owned();
                self.0
                    .zip_with(mask, &other.as_ref().as_ref())
                    .map(|ca| ca.$into_logical().into_series())
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
                self.0
                    .agg_min(groups)
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn agg_max(&self, groups: &GroupsProxy) -> Option<Series> {
                self.0
                    .agg_max(groups)
                    .map(|ca| ca.$into_logical().into_series())
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
                self.0
                    .agg_quantile(groups, quantile, interpol)
                    .map(|s| s.$into_logical().into_series())
            }

            fn agg_median(&self, groups: &GroupsProxy) -> Option<Series> {
                self.0
                    .agg_median(groups)
                    .map(|s| s.$into_logical().into_series())
            }

            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
            ) -> Series {
                let right_column = right_column.to_physical_repr().into_owned();
                self.0
                    .zip_outer_join_column(&right_column, opt_join_tuples)
                    .$into_logical()
                    .into_series()
            }

            fn subtract(&self, rhs: &Series) -> Result<Series> {
                match (self.dtype(), rhs.dtype()) {
                    (DataType::Date, DataType::Date) => {
                        let dt = DataType::Datetime(TimeUnit::Milliseconds, None);
                        let lhs = self.cast(&dt)?;
                        let rhs = rhs.cast(&dt)?;
                        lhs.subtract(&rhs)
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

        impl SeriesTrait for SeriesWrap<$ca> {
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
                self.0.interpolate().$into_logical().into_series()
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

            #[cfg(feature = "dtype-time")]
            fn time(&self) -> Result<&TimeChunked> {
                if matches!(self.0.dtype(), DataType::Time) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const TimeChunked)) }
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into Time",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            #[cfg(feature = "dtype-date")]
            fn date(&self) -> Result<&DateChunked> {
                if matches!(self.0.dtype(), DataType::Date) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const DateChunked)) }
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into Date",
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
                self.0.slice(offset, length).$into_logical().into_series()
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
                    // 3 refs
                    // ref Cow
                    // ref SeriesTrait
                    // ref ChunkedArray
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
                    // 3 refs
                    // ref Cow
                    // ref SeriesTrait
                    // ref ChunkedArray
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
                self.0
                    .filter(filter)
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn take(&self, indices: &IdxCa) -> Result<Series> {
                ChunkTake::take(self.0.deref(), indices.into())
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
                ChunkTake::take(self.0.deref(), iter.into())
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn take_every(&self, n: usize) -> Series {
                self.0.take_every(n).$into_logical().into_series()
            }

            unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
                ChunkTake::take_unchecked(self.0.deref(), iter.into())
                    .$into_logical()
                    .into_series()
            }

            unsafe fn take_unchecked(&self, idx: &IdxCa) -> Result<Series> {
                Ok(ChunkTake::take_unchecked(self.0.deref(), idx.into())
                    .$into_logical()
                    .into_series())
            }

            unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
                ChunkTake::take_unchecked(self.0.deref(), iter.into())
                    .$into_logical()
                    .into_series()
            }

            #[cfg(feature = "take_opt_iter")]
            fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
                ChunkTake::take(self.0.deref(), iter.into())
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                self.0.rechunk().$into_logical().into_series()
            }

            fn expand_at_index(&self, index: usize, length: usize) -> Series {
                self.0
                    .expand_at_index(index, length)
                    .$into_logical()
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
                self.0.get_any_value_unchecked(index).$into_logical()
            }

            fn sort_with(&self, options: SortOptions) -> Series {
                self.0.sort_with(options).$into_logical().into_series()
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
                self.0.unique().map(|ca| ca.$into_logical().into_series())
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
                self.0.reverse().$into_logical().into_series()
            }

            fn as_single_ptr(&mut self) -> Result<usize> {
                self.0.as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                self.0.shift(periods).$into_logical().into_series()
            }

            fn fill_null(&self, strategy: FillNullStrategy) -> Result<Series> {
                self.0
                    .fill_null(strategy)
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn _sum_as_series(&self) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn max_as_series(&self) -> Series {
                self.0.max_as_series().$into_logical()
            }
            fn min_as_series(&self) -> Series {
                self.0.min_as_series().$into_logical()
            }
            fn median_as_series(&self) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn var_as_series(&self) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn std_as_series(&self) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn quantile_as_series(
                &self,
                _quantile: f64,
                _interpol: QuantileInterpolOptions,
            ) -> Result<Series> {
                Ok(Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into())
            }

            fn fmt_list(&self) -> String {
                FmtList::fmt_list(&self.0)
            }

            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }

            fn pow(&self, _exponent: f64) -> Result<Series> {
                Err(PolarsError::ComputeError(
                    "cannot compute power of logical".into(),
                ))
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
                match self.0.dtype() {
                    DataType::Date => self
                        .0
                        .repeat_by(by)
                        .cast(&DataType::List(Box::new(DataType::Date)))
                        .unwrap()
                        .list()
                        .unwrap()
                        .clone(),
                    DataType::Time => self
                        .0
                        .repeat_by(by)
                        .cast(&DataType::List(Box::new(DataType::Time)))
                        .unwrap()
                        .list()
                        .unwrap()
                        .clone(),
                    _ => unreachable!(),
                }
            }
            #[cfg(feature = "is_first")]
            fn is_first(&self) -> Result<BooleanChunked> {
                self.0.is_first()
            }

            #[cfg(feature = "mode")]
            fn mode(&self) -> Result<Series> {
                self.0.mode().map(|ca| ca.$into_logical().into_series())
            }
        }
    };
}

#[cfg(feature = "dtype-date")]
impl_dyn_series!(DateChunked, into_date);
#[cfg(feature = "dtype-time")]
impl_dyn_series!(TimeChunked, into_time);

macro_rules! impl_dyn_series_numeric {
    ($ca: ident) => {
        impl private::PrivateSeriesNumeric for SeriesWrap<$ca> {
            fn bit_repr_is_large(&self) -> bool {
                if let DataType::Time = self.dtype() {
                    true
                } else {
                    false
                }
            }
            fn bit_repr_large(&self) -> UInt64Chunked {
                self.0.bit_repr_large()
            }
            fn bit_repr_small(&self) -> UInt32Chunked {
                self.0.bit_repr_small()
            }
        }
    };
}

#[cfg(feature = "dtype-date")]
impl_dyn_series_numeric!(DateChunked);
#[cfg(feature = "dtype-time")]
impl_dyn_series_numeric!(TimeChunked);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "dtype-datetime")]
    fn test_agg_list_type() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;

        let l = s
            .agg_list(&GroupsProxy::Idx(vec![(0, vec![0, 1, 2])].into()))
            .unwrap();

        match l.dtype() {
            DataType::List(inner) => {
                assert!(matches!(
                    &**inner,
                    DataType::Datetime(TimeUnit::Nanoseconds, None)
                ))
            }
            _ => assert!(false),
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-datetime")]
    #[cfg_attr(miri, ignore)]
    fn test_datelike_join() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let mut s1 = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;
        s1.rename("bar");

        let df = DataFrame::new(vec![s, s1])?;

        let out = df.left_join(&df.clone(), ["bar"], ["bar"])?;
        assert!(matches!(
            out.column("bar")?.dtype(),
            DataType::Datetime(TimeUnit::Nanoseconds, None)
        ));

        let out = df.inner_join(&df.clone(), ["bar"], ["bar"])?;
        assert!(matches!(
            out.column("bar")?.dtype(),
            DataType::Datetime(TimeUnit::Nanoseconds, None)
        ));

        let out = df.outer_join(&df.clone(), ["bar"], ["bar"])?;
        assert!(matches!(
            out.column("bar")?.dtype(),
            DataType::Datetime(TimeUnit::Nanoseconds, None)
        ));
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "dtype-datetime", feature = "dtype-duration"))]
    fn test_datelike_methods() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;

        let out = s.subtract(&s)?;
        assert!(matches!(
            out.dtype(),
            DataType::Duration(TimeUnit::Nanoseconds)
        ));

        let mut a = s.clone();
        a.append(&s).unwrap();
        assert_eq!(a.len(), 6);

        Ok(())
    }

    #[test]
    #[cfg(all(feature = "dtype-datetime", feature = "dtype-duration"))]
    fn test_arithmetic_dispatch() {
        let s = Int64Chunked::new("", &[1, 2, 3])
            .into_datetime(TimeUnit::Nanoseconds, None)
            .into_series();

        // check if we don't panic.
        let out = &s * 100;
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = &s / 100;
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = &s + 100;
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = &s - 100;
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = &s % 100;
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );

        let out = 100.mul(&s);
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = 100.div(&s);
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = 100.sub(&s);
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = 100.add(&s);
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        let out = 100.rem(&s);
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
    }

    #[test]
    #[cfg(feature = "dtype-duration")]
    fn test_duration() -> Result<()> {
        let a = Int64Chunked::new("", &[1, 2, 3])
            .into_datetime(TimeUnit::Nanoseconds, None)
            .into_series();
        let b = Int64Chunked::new("", &[2, 3, 4])
            .into_datetime(TimeUnit::Nanoseconds, None)
            .into_series();
        let c = Int64Chunked::new("", &[1, 1, 1])
            .into_duration(TimeUnit::Nanoseconds)
            .into_series();
        assert_eq!(
            *b.subtract(&a)?.dtype(),
            DataType::Duration(TimeUnit::Nanoseconds)
        );
        assert_eq!(
            *a.add_to(&c)?.dtype(),
            DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
        assert_eq!(
            b.subtract(&a)?,
            Int64Chunked::full("", 1, a.len())
                .into_duration(TimeUnit::Nanoseconds)
                .into_series()
        );
        Ok(())
    }
}
