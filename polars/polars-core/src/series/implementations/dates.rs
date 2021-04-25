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
use crate::chunked_array::comparison::*;
use crate::chunked_array::AsSinglePtr;
use crate::fmt::FmtList;
#[cfg(feature = "pivot")]
use crate::frame::groupby::pivot::*;
use crate::frame::groupby::*;
use crate::prelude::*;
use ahash::RandomState;
use arrow::array::{ArrayData, ArrayRef};
use arrow::buffer::Buffer;

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
        let s = $s.cast_with_datatype(&phys_type).unwrap();
        let s = s.$method($($args),*);

        // if the type is unchanged we return the original type
        if s.dtype() == &phys_type {
            s.cast_with_datatype(dtype).unwrap()
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
        let s = $s.cast_with_datatype(&phys_type).unwrap();
        let s = s.$method($($args),*)?;

        // if the type is unchanged we return the original type
        if s.dtype() == &phys_type {
            s.cast_with_datatype(dtype)
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
        let s = $s.cast_with_datatype(&phys_type).unwrap();
        let s = s.$method($($args),*)?;

        // if the type is unchanged we return the original type
        if s.dtype() == &phys_type {
            Some(s.cast_with_datatype(dtype).unwrap())
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
        let s = $s.cast_with_datatype(&phys_type).unwrap();
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
            unsafe fn equal_element(
                &self,
                idx_self: usize,
                idx_other: usize,
                other: &Series,
            ) -> bool {
                self.0.equal_element(idx_self, idx_other, other)
            }

            fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
                try_physical_dispatch!(self, zip_with_same_type, mask, other)
            }

            fn is_in_same_type(&self, list_array: &ListChunked) -> Result<BooleanChunked> {
                cast_and_apply!(self, is_in_same_type, list_array)
            }

            fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
                cast_and_apply!(self, vec_hash, random_state)
            }

            fn agg_mean(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_mean, groups)
            }

            fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_min, groups)
            }

            fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_max, groups)
            }

            fn agg_sum(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_sum, groups)
            }

            fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
                physical_dispatch!(self, agg_first, groups)
            }

            fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
                physical_dispatch!(self, agg_last, groups)
            }

            fn agg_std(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_std, groups)
            }

            fn agg_var(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
                opt_physical_dispatch!(self, agg_var, groups)
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

            #[cfg(feature = "pivot")]
            fn pivot<'a>(
                &self,
                pivot_series: &'a (dyn SeriesTrait + 'a),
                keys: Vec<Series>,
                groups: &[(u32, Vec<u32>)],
                agg_type: PivotAgg,
            ) -> Result<DataFrame> {
                self.0.pivot(pivot_series, keys, groups, agg_type)
            }

            #[cfg(feature = "pivot")]
            fn pivot_count<'a>(
                &self,
                pivot_series: &'a (dyn SeriesTrait + 'a),
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
                try_physical_dispatch!(self, subtract, rhs)
            }
            fn add_to(&self, rhs: &Series) -> Result<Series> {
                try_physical_dispatch!(self, add_to, rhs)
            }
            fn multiply(&self, rhs: &Series) -> Result<Series> {
                try_physical_dispatch!(self, multiply, rhs)
            }
            fn divide(&self, rhs: &Series) -> Result<Series> {
                try_physical_dispatch!(self, divide, rhs)
            }
            fn remainder(&self, rhs: &Series) -> Result<Series> {
                try_physical_dispatch!(self, remainder, rhs)
            }
            fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
                cast_and_apply!(self, group_tuples, multithreaded)
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {
            fn cum_max(&self, reverse: bool) -> Series {
                physical_dispatch!(self, cum_max, reverse)
            }

            fn cum_min(&self, reverse: bool) -> Series {
                physical_dispatch!(self, cum_min, reverse)
            }

            fn cum_sum(&self, reverse: bool) -> Series {
                physical_dispatch!(self, cum_sum, reverse)
            }

            fn rename(&mut self, name: &str) {
                self.0.rename(name);
            }

            fn array_data(&self) -> Vec<&ArrayData> {
                self.0.array_data()
            }

            fn chunk_lengths(&self) -> &Vec<usize> {
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

            fn take(&self, indices: &UInt32Chunked) -> Series {
                physical_dispatch!(self, take, indices)
            }

            fn take_iter(&self, iter: &mut dyn Iterator<Item = usize>) -> Series {
                physical_dispatch!(self, take_iter, iter)
            }

            fn take_every(&self, n: usize) -> Series {
                physical_dispatch!(self, take_every, n)
            }

            unsafe fn take_iter_unchecked(&self, iter: &mut dyn Iterator<Item = usize>) -> Series {
                physical_dispatch!(self, take_iter_unchecked, iter)
            }

            unsafe fn take_unchecked(&self, idx: &UInt32Chunked) -> Result<Series> {
                try_physical_dispatch!(self, take_unchecked, idx)
            }

            unsafe fn take_opt_iter_unchecked(
                &self,
                iter: &mut dyn Iterator<Item = Option<usize>>,
            ) -> Series {
                physical_dispatch!(self, take_opt_iter_unchecked, iter)
            }

            fn take_opt_iter(&self, iter: &mut dyn Iterator<Item = Option<usize>>) -> Series {
                physical_dispatch!(self, take_opt_iter, iter)
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

            fn cast_with_datatype(&self, data_type: &DataType) -> Result<Series> {
                use DataType::*;
                match data_type {
                    Boolean => ChunkCast::cast::<BooleanType>(&self.0).map(|ca| ca.into_series()),
                    Utf8 => ChunkCast::cast::<Utf8Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-u8")]
                    UInt8 => ChunkCast::cast::<UInt8Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-u16")]
                    UInt16 => ChunkCast::cast::<UInt16Type>(&self.0).map(|ca| ca.into_series()),
                    UInt32 => ChunkCast::cast::<UInt32Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-u64")]
                    UInt64 => ChunkCast::cast::<UInt64Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-i8")]
                    Int8 => ChunkCast::cast::<Int8Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-i16")]
                    Int16 => ChunkCast::cast::<Int16Type>(&self.0).map(|ca| ca.into_series()),
                    Int32 => ChunkCast::cast::<Int32Type>(&self.0).map(|ca| ca.into_series()),
                    Int64 => ChunkCast::cast::<Int64Type>(&self.0).map(|ca| ca.into_series()),
                    Float32 => ChunkCast::cast::<Float32Type>(&self.0).map(|ca| ca.into_series()),
                    Float64 => ChunkCast::cast::<Float64Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-date32")]
                    Date32 => ChunkCast::cast::<Date32Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-date64")]
                    Date64 => ChunkCast::cast::<Date64Type>(&self.0).map(|ca| ca.into_series()),
                    #[cfg(feature = "dtype-time64-ns")]
                    Time64(TimeUnit::Nanosecond) => {
                        ChunkCast::cast::<Time64NanosecondType>(&self.0).map(|ca| ca.into_series())
                    }
                    #[cfg(feature = "dtype-duration-ns")]
                    Duration(TimeUnit::Nanosecond) => {
                        ChunkCast::cast::<DurationNanosecondType>(&self.0)
                            .map(|ca| ca.into_series())
                    }
                    #[cfg(feature = "dtype-duration-ms")]
                    Duration(TimeUnit::Millisecond) => {
                        ChunkCast::cast::<DurationMillisecondType>(&self.0)
                            .map(|ca| ca.into_series())
                    }
                    List(_) => ChunkCast::cast::<ListType>(&self.0).map(|ca| ca.into_series()),
                    Categorical => {
                        ChunkCast::cast::<CategoricalType>(&self.0).map(|ca| ca.into_series())
                    }
                    dt => Err(PolarsError::Other(
                        format!("Casting to {:?} is not supported", dt).into(),
                    )),
                }
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

            fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
                self.0.null_bits()
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
                physical_dispatch!(self, sum_as_series,)
            }
            fn max_as_series(&self) -> Series {
                physical_dispatch!(self, max_as_series,)
            }
            fn min_as_series(&self) -> Series {
                physical_dispatch!(self, min_as_series,)
            }
            fn mean_as_series(&self) -> Series {
                physical_dispatch!(self, mean_as_series,)
            }
            fn median_as_series(&self) -> Series {
                physical_dispatch!(self, median_as_series,)
            }
            fn var_as_series(&self) -> Series {
                physical_dispatch!(self, var_as_series,)
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
                window_size: u32,
                weight: Option<&[f64]>,
                ignore_null: bool,
                min_periods: u32,
            ) -> Result<Series> {
                try_physical_dispatch!(
                    self,
                    rolling_sum,
                    window_size,
                    weight,
                    ignore_null,
                    min_periods
                )
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

            fn pow(&self, exponent: f64) -> Result<Series> {
                try_physical_dispatch!(self, pow, exponent)
            }

            fn peak_max(&self) -> BooleanChunked {
                cast_and_apply!(self, peak_max,)
            }

            fn peak_min(&self) -> BooleanChunked {
                cast_and_apply!(self, peak_min,)
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "dtype-date64")]
    fn test_agg_list_type() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let s = s.cast_with_datatype(&DataType::Date64)?;

        let l = s.agg_list(&[(0, vec![0, 1, 2])]).unwrap();
        assert!(matches!(l.dtype(), DataType::List(ArrowDataType::Date64)));

        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-date64")]
    fn test_datelike_join() -> Result<()> {
        let s = Series::new("foo", &[1, 2, 3]);
        let mut s1 = s.cast_with_datatype(&DataType::Date64)?;
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
        let s = s.cast_with_datatype(&DataType::Date64)?;

        let out = s.subtract(&s)?;
        assert!(matches!(out.dtype(), DataType::Date64));
        let out = s.add_to(&s)?;
        assert!(matches!(out.dtype(), DataType::Date64));
        let out = s.multiply(&s)?;
        assert!(matches!(out.dtype(), DataType::Date64));
        let out = s.divide(&s)?;
        assert!(matches!(out.dtype(), DataType::Date64));
        Ok(())
    }
}
