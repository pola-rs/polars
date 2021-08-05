//! This module exists to reduce compilation times.
//! All the data types are backed by a physical type in memory e.g. Date32 -> i32, Date64 -> i64.
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
use crate::chunked_array::{comparison::*, AsSinglePtr, ChunkIdIter};
use crate::fmt::FmtList;
#[cfg(feature = "pivot")]
use crate::frame::groupby::pivot::*;
use crate::frame::groupby::*;
use crate::prelude::*;
use ahash::RandomState;
use arrow::array::{ArrayData, ArrayRef};
use arrow::buffer::Buffer;
#[cfg(feature = "object")]
use std::any::Any;
use std::borrow::Cow;

impl<T> ChunkedArray<T> {
    /// get the physical memory type of a date type
    fn physical_type(&self) -> DataType {
        match self.dtype() {
            DataType::Duration(_) | DataType::Date64 | DataType::Time64(_) => DataType::Int64,
            DataType::Date32 => DataType::Int32,
            dt => panic!("already a physical type: {:?}", dt),
        }
    }
}

/// Dispatch the method call to the physical type and coerce back to logical type
macro_rules! physical_dispatch {
    ($s: expr, $method: ident, $($args:expr),*) => {{
        let dtype = $s.dtype();
        let phys_type = $s.physical_type();
        let s = $s.cast_with_dtype(&phys_type).unwrap();
        let s = s.$method($($args),*);

        // if the type is unchanged we return the original type
        if s.dtype() == &phys_type {
            s.cast_with_dtype(dtype).unwrap()
        }
        // else the change of type is part of the operation.
        else {
            s
        }
    }}
}

macro_rules! try_physical_dispatch {
    ($s: expr, $method: ident, $($args:expr),*) => {{
        let dtype = $s.dtype();
        let phys_type = $s.physical_type();
        let s = $s.cast_with_dtype(&phys_type).unwrap();
        let s = s.$method($($args),*)?;

        // if the type is unchanged we return the original type
        if s.dtype() == &phys_type {
            s.cast_with_dtype(dtype)
        }
        // else the change of type is part of the operation.
        else {
            Ok(s)
        }
    }}
}

macro_rules! opt_physical_dispatch {
    ($s: expr, $method: ident, $($args:expr),*) => {{
        let dtype = $s.dtype();
        let phys_type = $s.physical_type();
        let s = $s.cast_with_dtype(&phys_type).unwrap();
        let s = s.$method($($args),*)?;

        // if the type is unchanged we return the original type
        if s.dtype() == &phys_type {
            Some(s.cast_with_dtype(dtype).unwrap())
        }
        // else the change of type is part of the operation.
        else {
            Some(s)
        }
    }}
}

/// Same as physical dispatch, but doesnt care about return type
macro_rules! cast_and_apply {
    ($s: expr, $method: ident, $($args:expr),*) => {{
        let phys_type = $s.physical_type();
        let s = $s.cast_with_dtype(&phys_type).unwrap();
        s.$method($($args),*)
    }}
}

macro_rules! impl_dyn_series {
    ($ca: ident) => {
        impl IntoSeries for $ca {
            fn into_series(self) -> Series {
                Series(Arc::new(SeriesWrap(self)))
            }
        }

        impl private::PrivateSeries for SeriesWrap<$ca> {
            #[cfg(feature = "asof_join")]
            fn join_asof(&self, other: &Series) -> Result<Vec<Option<u32>>> {
                cast_and_apply!(self, join_asof, other)
            }

            fn set_sorted(&mut self, reverse: bool) {
                self.0.set_sorted(reverse)
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
                try_physical_dispatch!(self, zip_with_same_type, mask, other)
            }

            fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
                cast_and_apply!(self, vec_hash, random_state)
            }

            fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
                cast_and_apply!(self, vec_hash_combine, build_hasher, hashes)
            }

            fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on dates
                None
            }

            fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_min, groups)
            }

            fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_max, groups)
            }

            fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on dates
                None
            }

            fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
                physical_dispatch!(self, agg_first, groups)
            }

            fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
                physical_dispatch!(self, agg_last, groups)
            }

            fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on dates
                None
            }

            fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // does not make sense on dates
                None
            }

            fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
                cast_and_apply!(self, agg_n_unique, groups)
            }

            fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                // we cannot cast and dispatch as the inner type of the list would be incorrect
                self.0.agg_list(groups)
            }

            fn agg_quantile(&self, groups: &[(u32, Vec<u32>)], quantile: f64) -> Option<Series> {
                opt_physical_dispatch!(self, agg_quantile, groups, quantile)
            }

            fn agg_median(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_median, groups)
            }
            #[cfg(feature = "lazy")]
            fn agg_valid_count(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_valid_count, groups)
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
                let other = other.to_physical_repr();
                cast_and_apply!(self, hash_join_inner, &other)
            }
            fn hash_join_left(&self, other: &Series) -> Vec<(u32, Option<u32>)> {
                let other = other.to_physical_repr();
                cast_and_apply!(self, hash_join_left, &other)
            }
            fn hash_join_outer(&self, other: &Series) -> Vec<(Option<u32>, Option<u32>)> {
                let other = other.to_physical_repr();
                cast_and_apply!(self, hash_join_outer, &other)
            }
            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<u32>, Option<u32>)],
            ) -> Series {
                let right_column = right_column.to_physical_repr();
                physical_dispatch!(self, zip_outer_join_column, &right_column, opt_join_tuples)
            }
            fn subtract(&self, rhs: &Series) -> Result<Series> {
                match (self.dtype(), rhs.dtype()) {
                    (DataType::Date32, DataType::Date32) => {
                        let lhs = self.cast_with_dtype(&DataType::Int32).unwrap();
                        let rhs = rhs.cast_with_dtype(&DataType::Int32).unwrap();
                        Ok(lhs.subtract(&rhs)?.into_series())
                    }
                    (DataType::Date64, DataType::Date64) => {
                        let lhs = self.cast_with_dtype(&DataType::Int64).unwrap();
                        let rhs = rhs.cast_with_dtype(&DataType::Int64).unwrap();
                        Ok(lhs.subtract(&rhs)?.into_series())
                    }
                    (dtl, dtr) => Err(PolarsError::Other(
                        format!(
                            "cannot do subtraction on these date types: {:?}, {:?}",
                            dtl, dtr
                        )
                        .into(),
                    )),
                }
            }
            fn add_to(&self, _rhs: &Series) -> Result<Series> {
                Err(PolarsError::Other("cannot do addition on dates".into()))
            }
            fn multiply(&self, _rhs: &Series) -> Result<Series> {
                Err(PolarsError::Other(
                    "cannot do multiplication on dates".into(),
                ))
            }
            fn divide(&self, _rhs: &Series) -> Result<Series> {
                Err(PolarsError::Other("cannot do division on dates".into()))
            }
            fn remainder(&self, _rhs: &Series) -> Result<Series> {
                Err(PolarsError::Other(
                    "cannot do remainder operation on dates".into(),
                ))
            }
            fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
                cast_and_apply!(self, group_tuples, multithreaded)
            }
            #[cfg(feature = "sort_multiple")]
            fn argsort_multiple(&self, by: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
                let phys_type = self.0.physical_type();
                let s = self.cast_with_dtype(&phys_type).unwrap();

                self.0
                    .unpack_series_matching_type(&s)?
                    .argsort_multiple(by, reverse)
            }

            fn str_value(&self, index: usize) -> Cow<str> {
                // get AnyValue
                Cow::Owned(format!("{}", self.get(index)))
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {
            fn cum_max(&self, reverse: bool) -> Series {
                physical_dispatch!(self, cum_max, reverse)
            }

            fn cum_min(&self, reverse: bool) -> Series {
                physical_dispatch!(self, cum_min, reverse)
            }

            fn cum_sum(&self, _reverse: bool) -> Series {
                panic!("cannot sum dates")
            }

            fn rename(&mut self, name: &str) {
                self.0.rename(name);
            }

            fn array_data(&self) -> Vec<&ArrayData> {
                self.0.array_data()
            }

            fn chunk_lengths(&self) -> ChunkIdIter {
                self.0.chunk_id()
            }
            fn name(&self) -> &str {
                self.0.name()
            }

            fn field(&self) -> &Field {
                self.0.ref_field()
            }

            fn chunks(&self) -> &Vec<ArrayRef> {
                self.0.chunks()
            }

            fn shrink_to_fit(&mut self) {
                self.0.shrink_to_fit()
            }

            fn date32(&self) -> Result<&Date32Chunked> {
                if matches!(self.0.dtype(), DataType::Date32) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Date32Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into date32",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn date64(&self) -> Result<&Date64Chunked> {
                if matches!(self.0.dtype(), DataType::Date64) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Date64Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into date64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn time64_nanosecond(&self) -> Result<&Time64NanosecondChunked> {
                if matches!(self.0.dtype(), DataType::Time64(TimeUnit::Nanosecond)) {
                    unsafe {
                        Ok(&*(self as *const dyn SeriesTrait as *const Time64NanosecondChunked))
                    }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into time64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn duration_nanosecond(&self) -> Result<&DurationNanosecondChunked> {
                if matches!(self.0.dtype(), DataType::Duration(TimeUnit::Nanosecond)) {
                    unsafe {
                        Ok(&*(self as *const dyn SeriesTrait as *const DurationNanosecondChunked))
                    }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into duration_nanosecond",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn duration_millisecond(&self) -> Result<&DurationMillisecondChunked> {
                if matches!(self.0.dtype(), DataType::Duration(TimeUnit::Millisecond)) {
                    unsafe {
                        Ok(&*(self as *const dyn SeriesTrait as *const DurationMillisecondChunked))
                    }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into duration_millisecond",
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

            fn mean(&self) -> Option<f64> {
                cast_and_apply!(self, mean,)
            }

            fn median(&self) -> Option<f64> {
                cast_and_apply!(self, median,)
            }

            fn append(&mut self, other: &Series) -> Result<()> {
                if self.0.dtype() == other.dtype() {
                    // todo! add object
                    self.0.append(other.as_ref().as_ref());
                    Ok(())
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        "cannot append Series; data types don't match".into(),
                    ))
                }
            }

            fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
                try_physical_dispatch!(self, filter, filter)
            }

            fn take(&self, indices: &UInt32Chunked) -> Result<Series> {
                try_physical_dispatch!(self, take, indices)
            }

            fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
                try_physical_dispatch!(self, take_iter, iter)
            }

            fn take_every(&self, n: usize) -> Series {
                physical_dispatch!(self, take_every, n)
            }

            unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
                physical_dispatch!(self, take_iter_unchecked, iter)
            }

            unsafe fn take_unchecked(&self, idx: &UInt32Chunked) -> Result<Series> {
                try_physical_dispatch!(self, take_unchecked, idx)
            }

            unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
                physical_dispatch!(self, take_opt_iter_unchecked, iter)
            }

            #[cfg(feature = "take_opt_iter")]
            fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
                try_physical_dispatch!(self, take_opt_iter, iter)
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                physical_dispatch!(self, rechunk,)
            }

            fn head(&self, length: Option<usize>) -> Series {
                self.0.head(length).into_series()
            }

            fn tail(&self, length: Option<usize>) -> Series {
                self.0.tail(length).into_series()
            }

            fn expand_at_index(&self, index: usize, length: usize) -> Series {
                physical_dispatch!(self, expand_at_index, index, length)
            }

            fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series> {
                self.0.cast_with_dtype(data_type)
            }

            fn to_dummies(&self) -> Result<DataFrame> {
                cast_and_apply!(self, to_dummies,)
            }

            fn value_counts(&self) -> Result<DataFrame> {
                cast_and_apply!(self, value_counts,)
            }

            fn get(&self, index: usize) -> AnyValue {
                self.0.get_any_value(index)
            }

            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
                self.0.get_any_value_unchecked(index)
            }

            fn sort_in_place(&mut self, reverse: bool) {
                ChunkSort::sort_in_place(&mut self.0, reverse);
            }

            fn sort(&self, reverse: bool) -> Series {
                physical_dispatch!(self, sort, reverse)
            }

            fn argsort(&self, reverse: bool) -> UInt32Chunked {
                cast_and_apply!(self, argsort, reverse)
            }

            fn null_count(&self) -> usize {
                self.0.null_count()
            }

            fn unique(&self) -> Result<Series> {
                try_physical_dispatch!(self, unique,)
            }

            fn n_unique(&self) -> Result<usize> {
                cast_and_apply!(self, n_unique,)
            }

            fn arg_unique(&self) -> Result<UInt32Chunked> {
                cast_and_apply!(self, arg_unique,)
            }

            fn arg_min(&self) -> Option<usize> {
                cast_and_apply!(self, arg_min,)
            }

            fn arg_max(&self) -> Option<usize> {
                cast_and_apply!(self, arg_max,)
            }

            fn arg_true(&self) -> Result<UInt32Chunked> {
                let ca: &BooleanChunked = self.bool()?;
                Ok(ca.arg_true())
            }

            fn is_null(&self) -> BooleanChunked {
                cast_and_apply!(self, is_null,)
            }

            fn is_not_null(&self) -> BooleanChunked {
                cast_and_apply!(self, is_not_null,)
            }

            fn is_unique(&self) -> Result<BooleanChunked> {
                cast_and_apply!(self, is_unique,)
            }

            fn is_duplicated(&self) -> Result<BooleanChunked> {
                cast_and_apply!(self, is_duplicated,)
            }

            fn null_bits(&self) -> Vec<(usize, Option<&Buffer>)> {
                self.0.null_bits().collect()
            }

            fn reverse(&self) -> Series {
                physical_dispatch!(self, reverse,)
            }

            fn as_single_ptr(&mut self) -> Result<usize> {
                self.0.as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                physical_dispatch!(self, shift, periods)
            }

            fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Series> {
                try_physical_dispatch!(self, fill_none, strategy)
            }

            fn sum_as_series(&self) -> Series {
                panic!("cannot compute sum of dates")
            }
            fn max_as_series(&self) -> Series {
                physical_dispatch!(self, max_as_series,)
            }
            fn min_as_series(&self) -> Series {
                physical_dispatch!(self, min_as_series,)
            }
            fn mean_as_series(&self) -> Series {
                panic!("cannot compute mean of dates")
            }
            fn median_as_series(&self) -> Series {
                panic!("cannot compute median of dates")
            }
            fn var_as_series(&self) -> Series {
                panic!("cannot compute variance of dates")
            }
            fn std_as_series(&self) -> Series {
                physical_dispatch!(self, std_as_series,)
            }
            fn quantile_as_series(&self, quantile: f64) -> Result<Series> {
                try_physical_dispatch!(self, quantile_as_series, quantile)
            }
            fn rolling_mean(
                &self,
                window_size: u32,
                weight: Option<&[f64]>,
                ignore_null: bool,
                min_periods: u32,
            ) -> Result<Series> {
                try_physical_dispatch!(
                    self,
                    rolling_mean,
                    window_size,
                    weight,
                    ignore_null,
                    min_periods
                )
            }
            fn rolling_sum(
                &self,
                _window_size: u32,
                _weight: Option<&[f64]>,
                _ignore_null: bool,
                _min_periods: u32,
            ) -> Result<Series> {
                Err(PolarsError::Other(
                    "cannot compute rolling sum of dates".into(),
                ))
            }
            fn rolling_min(
                &self,
                window_size: u32,
                weight: Option<&[f64]>,
                ignore_null: bool,
                min_periods: u32,
            ) -> Result<Series> {
                try_physical_dispatch!(
                    self,
                    rolling_min,
                    window_size,
                    weight,
                    ignore_null,
                    min_periods
                )
            }
            fn rolling_max(
                &self,
                window_size: u32,
                weight: Option<&[f64]>,
                ignore_null: bool,
                min_periods: u32,
            ) -> Result<Series> {
                try_physical_dispatch!(
                    self,
                    rolling_max,
                    window_size,
                    weight,
                    ignore_null,
                    min_periods
                )
            }

            fn fmt_list(&self) -> String {
                FmtList::fmt_list(&self.0)
            }

            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }

            #[cfg(feature = "random")]
            #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
            fn sample_n(&self, n: usize, with_replacement: bool) -> Result<Series> {
                try_physical_dispatch!(self, sample_n, n, with_replacement)
            }

            #[cfg(feature = "random")]
            #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
            fn sample_frac(&self, frac: f64, with_replacement: bool) -> Result<Series> {
                try_physical_dispatch!(self, sample_frac, frac, with_replacement)
            }

            fn pow(&self, _exponent: f64) -> Result<Series> {
                Err(PolarsError::Other("cannot compute power of dates".into()))
            }

            fn peak_max(&self) -> BooleanChunked {
                cast_and_apply!(self, peak_max,)
            }

            fn peak_min(&self) -> BooleanChunked {
                cast_and_apply!(self, peak_min,)
            }
            #[cfg(feature = "is_in")]
            fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
                IsIn::is_in(&self.0, other)
            }
            #[cfg(feature = "repeat_by")]
            fn repeat_by(&self, by: &UInt32Chunked) -> ListChunked {
                RepeatBy::repeat_by(&self.0, by)
            }
            #[cfg(feature = "is_first")]
            fn is_first(&self) -> Result<BooleanChunked> {
                cast_and_apply!(self, is_first,)
            }

            #[cfg(feature = "object")]
            fn as_any(&self) -> &dyn Any {
                &self.0
            }
            #[cfg(feature = "mode")]
            fn mode(&self) -> Result<Series> {
                try_physical_dispatch!(self, mode,)
            }
        }
    };
}

#[cfg(feature = "dtype-duration-ns")]
impl_dyn_series!(DurationNanosecondChunked);
#[cfg(feature = "dtype-duration-ms")]
impl_dyn_series!(DurationMillisecondChunked);
#[cfg(feature = "dtype-date32")]
impl_dyn_series!(Date32Chunked);
#[cfg(feature = "dtype-date64")]
impl_dyn_series!(Date64Chunked);
#[cfg(feature = "dtype-time64-ns")]
impl_dyn_series!(Time64NanosecondChunked);

macro_rules! impl_dyn_series_numeric {
    ($ca: ident) => {
        impl private::PrivateSeriesNumeric for SeriesWrap<$ca> {
            fn bit_repr_is_large(&self) -> bool {
                cast_and_apply!(self, bit_repr_is_large,)
            }
            fn bit_repr_large(&self) -> UInt64Chunked {
                cast_and_apply!(self, bit_repr_large,)
            }
            fn bit_repr_small(&self) -> UInt32Chunked {
                cast_and_apply!(self, bit_repr_small,)
            }
        }
    };
}

#[cfg(feature = "dtype-duration-ns")]
impl_dyn_series_numeric!(DurationNanosecondChunked);
#[cfg(feature = "dtype-duration-ms")]
impl_dyn_series_numeric!(DurationMillisecondChunked);
#[cfg(feature = "dtype-date32")]
impl_dyn_series_numeric!(Date32Chunked);
#[cfg(feature = "dtype-date64")]
impl_dyn_series_numeric!(Date64Chunked);
#[cfg(feature = "dtype-time64-ns")]
impl_dyn_series_numeric!(Time64NanosecondChunked);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "dtype-date64")]
    fn test_agg_list_type() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast_with_dtype(&DataType::Date64)?;

        let l = s.agg_list(&[(0, vec![0, 1, 2])]).unwrap();
        assert!(matches!(l.dtype(), DataType::List(ArrowDataType::Date64)));

        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-date64")]
    #[cfg_attr(miri, ignore)]
    fn test_datelike_join() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let mut s1 = s.cast_with_dtype(&DataType::Date64)?;
        s1.rename("bar");

        let df = DataFrame::new(vec![s, s1])?;

        let out = df.left_join(&df.clone(), "bar", "bar")?;
        assert!(matches!(out.column("bar")?.dtype(), DataType::Date64));

        let out = df.inner_join(&df.clone(), "bar", "bar")?;
        assert!(matches!(out.column("bar")?.dtype(), DataType::Date64));

        let out = df.outer_join(&df.clone(), "bar", "bar")?;
        assert!(matches!(out.column("bar")?.dtype(), DataType::Date64));
        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-date64")]
    fn test_datelike_methods() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast_with_dtype(&DataType::Date64)?;

        let out = s.subtract(&s)?;
        assert!(matches!(out.dtype(), DataType::Int64));
        Ok(())
    }
}
