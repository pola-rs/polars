//! This module exists to reduce compilation times.
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
use crate::chunked_array::{
    comparison::*,
    ops::{explode::ExplodeByOffsets, ToBitRepr},
    AsSinglePtr, ChunkIdIter,
};
use crate::fmt::FmtList;
#[cfg(feature = "pivot")]
use crate::frame::groupby::pivot::*;
use crate::frame::{groupby::*, hash_join::*};
use crate::prelude::*;
use ahash::RandomState;
#[cfg(feature = "object")]
use std::any::Any;
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

            #[cfg(feature = "cum_agg")]
            fn _cumsum(&self, _reverse: bool) -> Series {
                panic!("cannot sum logical")
            }

            #[cfg(feature = "asof_join")]
            fn join_asof(&self, other: &Series) -> Result<Vec<Option<u32>>> {
                let other = other.to_physical_repr();
                self.0.deref().join_asof(&other)
            }

            fn set_sorted(&mut self, reverse: bool) {
                self.0.deref_mut().set_sorted(reverse)
            }

            unsafe fn equal_element(
                &self,
                idx_self: usize,
                idx_other: usize,
                other: &Series,
            ) -> bool {
                self.0.equal_element(idx_self, idx_other, other)
            }

            #[cfg(feature = "zip_with")]
            fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
                let other = other.to_physical_repr().into_owned();
                self.0
                    .zip_with(mask, &other.as_ref().as_ref())
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
                self.0.vec_hash(random_state)
            }

            fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
                self.0.vec_hash_combine(build_hasher, hashes)
            }

            fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on logical
                None
            }

            fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                self.0
                    .agg_min(groups)
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                self.0
                    .agg_max(groups)
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on logical
                None
            }

            fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
                self.0.agg_first(groups).$into_logical().into_series()
            }

            fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
                self.0.agg_last(groups).$into_logical().into_series()
            }

            fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on logical
                None
            }

            fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on logical
                None
            }

            fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
                self.0.agg_n_unique(groups)
            }

            fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // we cannot cast and dispatch as the inner type of the list would be incorrect
                self.0.agg_list(groups).map(|s| {
                    s.cast(&DataType::List(Box::new(self.dtype().clone())))
                        .unwrap()
                })
            }

            fn agg_quantile(
                &self,
                groups: &[(u32, Vec<u32>)],
                quantile: f64,
                interpol: QuantileInterpolOptions,
            ) -> Option<Series> {
                self.0
                    .agg_quantile(groups, quantile, interpol)
                    .map(|s| s.$into_logical().into_series())
            }

            fn agg_median(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                self.0
                    .agg_median(groups)
                    .map(|s| s.$into_logical().into_series())
            }
            #[cfg(feature = "lazy")]
            fn agg_valid_count(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                self.0.agg_valid_count(groups)
            }

            #[cfg(feature = "pivot")]
            fn pivot<'a>(
                &self,
                pivot_series: &'a Series,
                keys: Vec<Series>,
                groups: &[(u32, Vec<u32>)],
                agg_type: PivotAgg,
            ) -> Result<DataFrame> {
                self.0.pivot(pivot_series, keys, groups, agg_type)
            }

            #[cfg(feature = "pivot")]
            fn pivot_count<'a>(
                &self,
                pivot_series: &'a Series,
                keys: Vec<Series>,
                groups: &[(u32, Vec<u32>)],
            ) -> Result<DataFrame> {
                self.0.pivot_count(pivot_series, keys, groups)
            }
            fn hash_join_inner(&self, other: &Series) -> Vec<(u32, u32)> {
                let other = other.to_physical_repr().into_owned();
                self.0.hash_join_inner(&other.as_ref().as_ref())
            }
            fn hash_join_left(&self, other: &Series) -> Vec<(u32, Option<u32>)> {
                let other = other.to_physical_repr().into_owned();
                self.0.hash_join_left(&other.as_ref().as_ref())
            }
            fn hash_join_outer(&self, other: &Series) -> Vec<(Option<u32>, Option<u32>)> {
                let other = other.to_physical_repr().into_owned();
                self.0.hash_join_outer(&other.as_ref().as_ref())
            }
            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<u32>, Option<u32>)],
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
                        let lhs = self.cast(&DataType::Int32).unwrap();
                        let rhs = rhs.cast(&DataType::Int32).unwrap();
                        Ok(lhs.subtract(&rhs)?.$into_logical().into_series())
                    }
                    (DataType::Datetime, DataType::Datetime) => {
                        let lhs = self.cast(&DataType::Int64).unwrap();
                        let rhs = rhs.cast(&DataType::Int64).unwrap();
                        Ok(lhs.subtract(&rhs)?.$into_logical().into_series())
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
            fn add_to(&self, _rhs: &Series) -> Result<Series> {
                Err(PolarsError::ComputeError(
                    "cannot do addition on logical".into(),
                ))
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
            fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
                self.0.group_tuples(multithreaded)
            }
            #[cfg(feature = "sort_multiple")]
            fn argsort_multiple(&self, by: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
                self.0.deref().argsort_multiple(by, reverse)
            }

            fn str_value(&self, index: usize) -> Cow<str> {
                // get AnyValue
                Cow::Owned(format!("{}", self.get(index)))
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {
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

            fn datetime(&self) -> Result<&DatetimeChunked> {
                if matches!(self.0.dtype(), DataType::Datetime) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const DatetimeChunked)) }
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into datetime",
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
                    let other = other.to_physical_repr().into_owned();
                    self.0.append(other.as_ref().as_ref());
                    Ok(())
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        "cannot append Series; data types don't match".into(),
                    ))
                }
            }

            fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
                self.0
                    .filter(filter)
                    .map(|ca| ca.$into_logical().into_series())
            }

            fn take(&self, indices: &UInt32Chunked) -> Result<Series> {
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

            unsafe fn take_unchecked(&self, idx: &UInt32Chunked) -> Result<Series> {
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

            fn head(&self, length: Option<usize>) -> Series {
                self.0.head(length).$into_logical().into_series()
            }

            fn tail(&self, length: Option<usize>) -> Series {
                self.0.tail(length).$into_logical().into_series()
            }

            fn expand_at_index(&self, index: usize, length: usize) -> Series {
                self.0
                    .expand_at_index(index, length)
                    .$into_logical()
                    .into_series()
            }

            fn cast(&self, data_type: &DataType) -> Result<Series> {
                const NS_IN_DAY: i64 = 86400000_000_000;
                use DataType::*;
                let ca = match (self.dtype(), data_type) {
                    #[cfg(feature = "dtype-datetime")]
                    (Date, Datetime) => {
                        let casted = self.0.cast(data_type)?;
                        let casted = casted.datetime().unwrap();
                        return Ok((casted.deref() * NS_IN_DAY).into_date().into_series());
                    }
                    #[cfg(feature = "dtype-date")]
                    (Datetime, Date) => {
                        let ca = self.0.deref() / NS_IN_DAY;
                        Cow::Owned(ca)
                    }
                    _ => Cow::Borrowed(self.0.deref()),
                };
                ca.cast(data_type)
            }

            fn to_dummies(&self) -> Result<DataFrame> {
                self.0.to_dummies()
            }

            fn value_counts(&self) -> Result<DataFrame> {
                self.0.value_counts()
            }

            fn get(&self, index: usize) -> AnyValue {
                self.0.get_any_value(index)
            }

            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
                self.0.get_any_value_unchecked(index).$into_logical()
            }

            fn sort_with(&self, options: SortOptions) -> Series {
                self.0.sort_with(options).$into_logical().into_series()
            }

            fn argsort(&self, reverse: bool) -> UInt32Chunked {
                self.0.argsort(reverse)
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

            fn arg_unique(&self) -> Result<UInt32Chunked> {
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
            fn mean_as_series(&self) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
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
            fn repeat_by(&self, by: &UInt32Chunked) -> ListChunked {
                match self.0.dtype() {
                    DataType::Date => self
                        .0
                        .repeat_by(by)
                        .cast(&DataType::List(Box::new(DataType::Date)))
                        .unwrap()
                        .list()
                        .unwrap()
                        .clone(),
                    DataType::Datetime => self
                        .0
                        .repeat_by(by)
                        .cast(&DataType::List(Box::new(DataType::Datetime)))
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

            #[cfg(feature = "object")]
            fn as_any(&self) -> &dyn Any {
                &self.0
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
#[cfg(feature = "dtype-datetime")]
impl_dyn_series!(DatetimeChunked, into_date);
#[cfg(feature = "dtype-time")]
impl_dyn_series!(TimeChunked, into_time);

macro_rules! impl_dyn_series_numeric {
    ($ca: ident) => {
        impl private::PrivateSeriesNumeric for SeriesWrap<$ca> {
            fn bit_repr_is_large(&self) -> bool {
                if let DataType::Datetime = self.dtype() {
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
#[cfg(feature = "dtype-datetime")]
impl_dyn_series_numeric!(DatetimeChunked);
#[cfg(feature = "dtype-time")]
impl_dyn_series_numeric!(TimeChunked);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "dtype-datetime")]
    fn test_agg_list_type() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast(&DataType::Datetime)?;

        let l = s.agg_list(&[(0, vec![0, 1, 2])]).unwrap();

        match l.dtype() {
            DataType::List(inner) => {
                assert!(matches!(&**inner, DataType::Datetime))
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
        let mut s1 = s.cast(&DataType::Datetime)?;
        s1.rename("bar");

        let df = DataFrame::new(vec![s, s1])?;

        let out = df.left_join(&df.clone(), "bar", "bar")?;
        assert!(matches!(out.column("bar")?.dtype(), DataType::Datetime));

        let out = df.inner_join(&df.clone(), "bar", "bar")?;
        assert!(matches!(out.column("bar")?.dtype(), DataType::Datetime));

        let out = df.outer_join(&df.clone(), "bar", "bar")?;
        assert!(matches!(out.column("bar")?.dtype(), DataType::Datetime));
        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-datetime")]
    fn test_datelike_methods() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast(&DataType::Datetime)?;

        let out = s.subtract(&s)?;
        assert!(matches!(out.dtype(), DataType::Datetime));

        let mut a = s.clone();
        a.append(&s).unwrap();
        assert_eq!(a.len(), 6);

        Ok(())
    }

    #[test]
    fn test_arithmetic_dispatch() {
        let s = Int64Chunked::new("", &[1, 2, 3]).into_date().into_series();

        // check if we don't panic.
        let out = &s * 100;
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = &s / 100;
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = &s + 100;
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = &s - 100;
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = &s % 100;
        assert_eq!(out.dtype(), &DataType::Datetime);

        let out = 100.mul(&s);
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = 100.div(&s);
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = 100.sub(&s);
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = 100.add(&s);
        assert_eq!(out.dtype(), &DataType::Datetime);
        let out = 100.rem(&s);
        assert_eq!(out.dtype(), &DataType::Datetime);
    }
}
