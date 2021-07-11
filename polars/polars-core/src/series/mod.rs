//! Type agnostic columnar data structure.
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::{array::ArrayRef, buffer::Buffer};
pub(crate) mod arithmetic;
mod comparison;
pub mod implementations;
pub(crate) mod iterator;

use crate::chunked_array::{builder::get_list_builder, ChunkIdIter};
use crate::utils::{split_ca, split_series};
use crate::{series::arithmetic::coerce_lhs_rhs, POOL};
use arrow::array::ArrayData;
use arrow::compute::cast;
use itertools::Itertools;
use num::NumCast;
use rayon::prelude::*;
use std::any::Any;
use std::convert::TryFrom;
use std::ops::Deref;
use std::sync::Arc;

pub trait IntoSeries {
    fn into_series(self) -> Series
    where
        Self: Sized;
}

pub(crate) mod private {
    use super::*;
    #[cfg(feature = "pivot")]
    use crate::frame::groupby::pivot::PivotAgg;
    use crate::frame::groupby::GroupTuples;

    use crate::chunked_array::ops::compare_inner::{PartialEqInner, PartialOrdInner};
    use ahash::RandomState;
    use std::borrow::Cow;

    pub trait PrivateSeriesNumeric {
        fn bit_repr_is_large(&self) -> bool {
            unimplemented!()
        }
        fn bit_repr_large(&self) -> UInt64Chunked {
            unimplemented!()
        }
        fn bit_repr_small(&self) -> UInt32Chunked {
            unimplemented!()
        }
    }

    pub trait PrivateSeries {
        #[cfg(feature = "asof_join")]
        fn join_asof(&self, _other: &Series) -> Result<Vec<Option<u32>>> {
            unimplemented!()
        }

        fn set_sorted(&mut self, _reverse: bool) {
            unimplemented!()
        }

        unsafe fn equal_element(
            &self,
            _idx_self: usize,
            _idx_other: usize,
            _other: &Series,
        ) -> bool {
            unimplemented!()
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
            unimplemented!()
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
            unimplemented!()
        }
        fn vec_hash(&self, _build_hasher: RandomState) -> AlignedVec<u64> {
            unimplemented!()
        }
        fn vec_hash_combine(&self, _build_hasher: RandomState, _hashes: &mut [u64]) {
            unimplemented!()
        }
        fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_min(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_max(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_first(&self, _groups: &[(u32, Vec<u32>)]) -> Series {
            unimplemented!()
        }
        fn agg_last(&self, _groups: &[(u32, Vec<u32>)]) -> Series {
            unimplemented!()
        }
        fn agg_n_unique(&self, _groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
            unimplemented!()
        }
        fn agg_list(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_quantile(&self, _groups: &[(u32, Vec<u32>)], _quantile: f64) -> Option<Series> {
            unimplemented!()
        }
        fn agg_median(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        #[cfg(feature = "lazy")]
        fn agg_valid_count(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            unimplemented!()
        }
        #[cfg(feature = "pivot")]
        fn pivot<'a>(
            &self,
            _pivot_series: &'a Series,
            _keys: Vec<Series>,
            _groups: &[(u32, Vec<u32>)],
            _agg_type: PivotAgg,
        ) -> Result<DataFrame> {
            unimplemented!()
        }

        #[cfg(feature = "pivot")]
        fn pivot_count<'a>(
            &self,
            _pivot_series: &'a Series,
            _keys: Vec<Series>,
            _groups: &[(u32, Vec<u32>)],
        ) -> Result<DataFrame> {
            unimplemented!()
        }

        fn hash_join_inner(&self, _other: &Series) -> Vec<(u32, u32)> {
            unimplemented!()
        }
        fn hash_join_left(&self, _other: &Series) -> Vec<(u32, Option<u32>)> {
            unimplemented!()
        }
        fn hash_join_outer(&self, _other: &Series) -> Vec<(Option<u32>, Option<u32>)> {
            unimplemented!()
        }
        fn zip_outer_join_column(
            &self,
            _right_column: &Series,
            _opt_join_tuples: &[(Option<u32>, Option<u32>)],
        ) -> Series {
            unimplemented!()
        }

        fn subtract(&self, _rhs: &Series) -> Result<Series> {
            unimplemented!()
        }
        fn add_to(&self, _rhs: &Series) -> Result<Series> {
            unimplemented!()
        }
        fn multiply(&self, _rhs: &Series) -> Result<Series> {
            unimplemented!()
        }
        fn divide(&self, _rhs: &Series) -> Result<Series> {
            unimplemented!()
        }
        fn remainder(&self, _rhs: &Series) -> Result<Series> {
            unimplemented!()
        }
        fn group_tuples(&self, _multithreaded: bool) -> GroupTuples {
            unimplemented!()
        }
        fn zip_with_same_type(&self, _mask: &BooleanChunked, _other: &Series) -> Result<Series> {
            unimplemented!()
        }
        #[cfg(feature = "sort_multiple")]
        fn argsort_multiple(&self, _by: &[Series], _reverse: &[bool]) -> Result<UInt32Chunked> {
            Err(PolarsError::InvalidOperation(
                "argsort_multiple is not implemented for this Series".into(),
            ))
        }
        /// Formatted string representation. Can used in formatting.
        fn str_value(&self, _index: usize) -> Cow<str> {
            unimplemented!()
        }
    }
}

pub trait SeriesTrait:
    Send + Sync + private::PrivateSeries + private::PrivateSeriesNumeric
{
    /// Get an array with the cumulative max computed at every element
    fn cum_max(&self, _reverse: bool) -> Series {
        panic!("operation cum_max not supported for this dtype")
    }

    /// Get an array with the cumulative min computed at every element
    fn cum_min(&self, _reverse: bool) -> Series {
        panic!("operation cum_min not supported for this dtype")
    }

    /// Get an array with the cumulative sum computed at every element
    fn cum_sum(&self, _reverse: bool) -> Series {
        panic!("operation cum_sum not supported for this dtype")
    }

    /// Rename the Series.
    fn rename(&mut self, name: &str);

    /// Get Arrow ArrayData
    fn array_data(&self) -> Vec<&ArrayData> {
        unimplemented!()
    }

    /// Get the lengths of the underlying chunks
    fn chunk_lengths(&self) -> ChunkIdIter {
        unimplemented!()
    }
    /// Name of series.
    fn name(&self) -> &str {
        unimplemented!()
    }

    /// Get field (used in schema)
    fn field(&self) -> &Field {
        unimplemented!()
    }

    /// Get datatype of series.
    fn dtype(&self) -> &DataType {
        self.field().data_type()
    }

    /// Underlying chunks.
    fn chunks(&self) -> &Vec<ArrayRef> {
        unimplemented!()
    }

    /// Number of chunks in this Series
    fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    /// Shrink the capacity of this array to fit it's length.
    fn shrink_to_fit(&mut self) {
        eprintln!(
            "shrink to fit, not yet supported for this {:?}",
            self.dtype()
        )
    }

    /// Unpack to ChunkedArray of dtype i8
    fn i8(&self) -> Result<&Int8Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != i8", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray i16
    fn i16(&self) -> Result<&Int16Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != i16", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray
    /// ```
    /// # use polars_core::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let s_squared: Series = s.i32()
    ///     .unwrap()
    ///     .into_iter()
    ///     .map(|opt_v| {
    ///         match opt_v {
    ///             Some(v) => Some(v * v),
    ///             None => None, // null value
    ///         }
    /// }).collect();
    /// ```
    fn i32(&self) -> Result<&Int32Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != i32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype i64
    fn i64(&self) -> Result<&Int64Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != i64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype f32
    fn f32(&self) -> Result<&Float32Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != f32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype f64
    fn f64(&self) -> Result<&Float64Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != f64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u8
    fn u8(&self) -> Result<&UInt8Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != u8", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u16
    fn u16(&self) -> Result<&UInt16Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != u16", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u32
    fn u32(&self) -> Result<&UInt32Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != u32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u64
    fn u64(&self) -> Result<&UInt64Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != u32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype bool
    fn bool(&self) -> Result<&BooleanChunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != bool", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype utf8
    fn utf8(&self) -> Result<&Utf8Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != utf8", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype date32
    fn date32(&self) -> Result<&Date32Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != date32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype date64
    fn date64(&self) -> Result<&Date64Chunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != date64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype time64_nanosecond
    fn time64_nanosecond(&self) -> Result<&Time64NanosecondChunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != time64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype duration_nanosecond
    fn duration_nanosecond(&self) -> Result<&DurationNanosecondChunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != duration_nanosecond", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype duration_millisecond
    fn duration_millisecond(&self) -> Result<&DurationMillisecondChunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} !== duration_millisecond", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype list
    fn list(&self) -> Result<&ListChunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != list", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype categorical
    fn categorical(&self) -> Result<&CategoricalChunked> {
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} != categorical", self.dtype()).into(),
        ))
    }

    /// Check if underlying physical data is numeric.
    ///
    /// Date types and Categoricals are also considered numeric.
    fn is_numeric_physical(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        #[allow(clippy::match_like_matches_macro)]
        match self.dtype() {
            DataType::Utf8 | DataType::List(_) | DataType::Boolean | DataType::Null => false,
            #[cfg(feature = "object")]
            DataType::Object(_) => false,
            _ => true,
        }
    }

    /// Check if underlying data is numeric
    fn is_numeric(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        #[allow(clippy::match_like_matches_macro)]
        match self.dtype() {
            DataType::Utf8
            | DataType::List(_)
            | DataType::Categorical
            | DataType::Date32
            | DataType::Date64
            | DataType::Duration(_)
            | DataType::Time64(_)
            | DataType::Boolean
            | DataType::Null => false,
            #[cfg(feature = "object")]
            DataType::Object(_) => false,
            _ => true,
        }
    }

    /// Append Arrow array of same dtype to this Series.
    fn append_array(&mut self, _other: ArrayRef) -> Result<()> {
        unimplemented!()
    }

    /// Take `num_elements` from the top as a zero copy view.
    fn limit(&self, num_elements: usize) -> Series {
        self.slice(0, num_elements)
    }

    /// Get a zero copy view of the data.
    ///
    /// When offset is negative the offset is counted from the
    /// end of the array
    fn slice(&self, _offset: i64, _length: usize) -> Series {
        unimplemented!()
    }

    /// Append a Series of the same type in place.
    fn append(&mut self, _other: &Series) -> Result<()> {
        unimplemented!()
    }

    /// Filter by boolean mask. This operation clones data.
    fn filter(&self, _filter: &BooleanChunked) -> Result<Series> {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value for that element.
    fn take_iter(&self, _iter: &mut dyn Iterator<Item = usize>) -> Series {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// This doesn't check any bounds.
    unsafe fn take_iter_unchecked(&self, _iter: &mut dyn Iterator<Item = usize>) -> Series {
        unimplemented!()
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_unchecked(&self, _idx: &UInt32Chunked) -> Result<Series> {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// This doesn't check any bounds.
    unsafe fn take_opt_iter_unchecked(
        &self,
        _iter: &mut dyn Iterator<Item = Option<usize>>,
    ) -> Series {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value for that element
    fn take_opt_iter(&self, _iter: &mut dyn Iterator<Item = Option<usize>>) -> Series {
        unimplemented!()
    }

    /// Take by index. This operation is clone.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value for that element.
    fn take(&self, _indices: &UInt32Chunked) -> Series {
        unimplemented!()
    }

    /// Get length of series.
    fn len(&self) -> usize {
        unimplemented!()
    }

    /// Check if Series is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Aggregate all chunks to a contiguous array of memory.
    fn rechunk(&self) -> Series {
        unimplemented!()
    }

    /// Get the head of the Series.
    fn head(&self, _length: Option<usize>) -> Series {
        unimplemented!()
    }

    /// Get the tail of the Series.
    fn tail(&self, _length: Option<usize>) -> Series {
        unimplemented!()
    }

    /// Take every nth value as a new Series
    fn take_every(&self, n: usize) -> Series;

    /// Drop all null values and return a new Series.
    fn drop_nulls(&self) -> Series {
        if self.null_count() == 0 {
            Series(self.clone_inner())
        } else {
            self.filter(&self.is_not_null()).unwrap()
        }
    }

    /// Returns the mean value in the array
    /// Returns an option because the array is nullable.
    fn mean(&self) -> Option<f64> {
        unimplemented!()
    }

    /// Returns the median value in the array
    /// Returns an option because the array is nullable.
    fn median(&self) -> Option<f64> {
        unimplemented!()
    }

    /// Create a new Series filled with values at that index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// let s = Series::new("a", [0i32, 1, 8]);
    /// let expanded = s.expand_at_index(2, 4);
    /// assert_eq!(Vec::from(expanded.i32().unwrap()), &[Some(8), Some(8), Some(8), Some(8)])
    /// ```
    fn expand_at_index(&self, _index: usize, _length: usize) -> Series {
        unimplemented!()
    }

    fn cast_with_dtype(&self, _data_type: &DataType) -> Result<Series> {
        unimplemented!()
    }

    /// Create dummy variables. See [DataFrame](DataFrame::to_dummies)
    fn to_dummies(&self) -> Result<DataFrame> {
        unimplemented!()
    }

    fn value_counts(&self) -> Result<DataFrame> {
        unimplemented!()
    }

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    fn get(&self, _index: usize) -> AnyValue {
        unimplemented!()
    }

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    ///
    /// # Safety
    /// Does not do any bounds checking
    unsafe fn get_unchecked(&self, _index: usize) -> AnyValue {
        unimplemented!()
    }

    /// Sort in place.
    fn sort_in_place(&mut self, _reverse: bool) {
        unimplemented!()
    }

    fn sort(&self, _reverse: bool) -> Series {
        unimplemented!()
    }

    /// Retrieve the indexes needed for a sort.
    fn argsort(&self, _reverse: bool) -> UInt32Chunked {
        unimplemented!()
    }

    /// Count the null values.
    fn null_count(&self) -> usize {
        unimplemented!()
    }

    /// Get unique values in the Series.
    fn unique(&self) -> Result<Series> {
        unimplemented!()
    }

    /// Get unique values in the Series.
    fn n_unique(&self) -> Result<usize> {
        unimplemented!()
    }

    /// Get first indexes of unique values.
    fn arg_unique(&self) -> Result<UInt32Chunked> {
        unimplemented!()
    }

    /// Get min index
    fn arg_min(&self) -> Option<usize> {
        unimplemented!()
    }

    /// Get max index
    fn arg_max(&self) -> Option<usize> {
        unimplemented!()
    }

    /// Get indexes that evaluate true
    fn arg_true(&self) -> Result<UInt32Chunked> {
        Err(PolarsError::InvalidOperation(
            "arg_true can only be called for boolean dtype".into(),
        ))
    }

    /// Get a mask of the null values.
    fn is_null(&self) -> BooleanChunked {
        unimplemented!()
    }

    /// Get a mask of the non-null values.
    fn is_not_null(&self) -> BooleanChunked {
        unimplemented!()
    }

    /// Get a mask of all the unique values.
    fn is_unique(&self) -> Result<BooleanChunked> {
        unimplemented!()
    }

    /// Get a mask of all the duplicated values.
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        unimplemented!()
    }

    /// Get the bits that represent the null values of the underlying ChunkedArray
    fn null_bits(&self) -> Vec<(usize, Option<&Buffer>)> {
        unimplemented!()
    }

    /// return a Series in reversed order
    fn reverse(&self) -> Series {
        unimplemented!()
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&mut self) -> Result<usize> {
        Err(PolarsError::InvalidOperation(
            "operation 'as_single_ptr' not supported".into(),
        ))
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`shift` operation on `ChunkedArray<T>`](../chunked_array/ops/trait.ChunkShift.html).*
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example() -> Result<()> {
    ///     let s = Series::new("series", &[1, 2, 3]);
    ///
    ///     let shifted = s.shift(1);
    ///     assert_eq!(Vec::from(shifted.i32()?), &[None, Some(1), Some(2)]);
    ///
    ///     let shifted = s.shift(-1);
    ///     assert_eq!(Vec::from(shifted.i32()?), &[Some(2), Some(3), None]);
    ///
    ///     let shifted = s.shift(2);
    ///     assert_eq!(Vec::from(shifted.i32()?), &[None, None, Some(1)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    fn shift(&self, _periods: i64) -> Series {
        unimplemented!()
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`fill_none` operation on `ChunkedArray<T>`](../chunked_array/ops/trait.ChunkFillNone.html)*.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example() -> Result<()> {
    ///     let s = Series::new("some_missing", &[Some(1), None, Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Forward)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Backward)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Min)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Max)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Mean)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    fn fill_none(&self, _strategy: FillNoneStrategy) -> Result<Series> {
        unimplemented!()
    }

    /// Get the sum of the Series as a new Series of length 1.
    fn sum_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the max of the Series as a new Series of length 1.
    fn max_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the min of the Series as a new Series of length 1.
    fn min_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the mean of the Series as a new Series of length 1.
    fn mean_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the median of the Series as a new Series of length 1.
    fn median_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the variance of the Series as a new Series of length 1.
    fn var_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the standard deviation of the Series as a new Series of length 1.
    fn std_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_as_series(&self, _quantile: f64) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling mean to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_mean).
    fn rolling_mean(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling sum to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_sum).
    fn rolling_sum(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling min to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_min).
    fn rolling_min(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling max to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_max).
    fn rolling_max(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Series> {
        unimplemented!()
    }

    fn fmt_list(&self) -> String {
        "fmt implemented".into()
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> Result<UInt32Chunked> {
        self.date64().map(|ca| ca.hour())
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> Result<UInt32Chunked> {
        self.date64().map(|ca| ca.minute())
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> Result<UInt32Chunked> {
        self.date64().map(|ca| ca.second())
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> Result<UInt32Chunked> {
        self.date64().map(|ca| ca.nanosecond())
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.day()),
            DataType::Date64 => self.date64().map(|ca| ca.day()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }
    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.weekday()),
            DataType::Date64 => self.date64().map(|ca| ca.weekday()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.week()),
            DataType::Date64 => self.date64().map(|ca| ca.week()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal_day(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.ordinal()),
            DataType::Date64 => self.date64().map(|ca| ca.ordinal()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.month()),
            DataType::Date64 => self.date64().map(|ca| ca.month()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Result<Int32Chunked> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.year()),
            DataType::Date64 => self.date64().map(|ca| ca.year()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Format Date32/Date64 with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn strftime(&self, fmt: &str) -> Result<Series> {
        match self.dtype() {
            DataType::Date32 => self.date32().map(|ca| ca.strftime(fmt).into_series()),
            DataType::Date64 => self.date64().map(|ca| ca.strftime(fmt).into_series()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Convert date(time) object to timestamp in ms.
    fn timestamp(&self) -> Result<Int64Chunked> {
        match self.dtype() {
            DataType::Date32 => self
                .date32()
                .map(|ca| (ca.cast::<Int64Type>().unwrap() * 1000)),
            DataType::Date64 => self.date64().map(|ca| ca.cast::<Int64Type>().unwrap()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Clone inner ChunkedArray and wrap in a new Arc
    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        unimplemented!()
    }

    #[cfg(feature = "random")]
    #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
    /// Sample n datapoints from this Series.
    fn sample_n(&self, n: usize, with_replacement: bool) -> Result<Series>;

    #[cfg(feature = "random")]
    #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
    /// Sample a fraction between 0.0-1.0 of this ChunkedArray.
    fn sample_frac(&self, frac: f64, with_replacement: bool) -> Result<Series>;

    /// Get the value at this index as a downcastable Any trait ref.
    fn get_as_any(&self, _index: usize) -> &dyn Any {
        unimplemented!()
    }

    /// Raise a numeric series to the power of exponent.
    fn pow(&self, _exponent: f64) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!("power operation not supported on dtype {:?}", self.dtype()).into(),
        ))
    }

    /// Get a boolean mask of the local maximum peaks.
    fn peak_max(&self) -> BooleanChunked {
        unimplemented!()
    }

    /// Get a boolean mask of the local minimum peaks.
    fn peak_min(&self) -> BooleanChunked {
        unimplemented!()
    }

    /// Check if elements of this Series are in the right Series, or List values of the right Series.
    #[cfg(feature = "is_in")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_in")))]
    fn is_in(&self, _other: &Series) -> Result<BooleanChunked> {
        unimplemented!()
    }
    #[cfg(feature = "repeat_by")]
    #[cfg_attr(docsrs, doc(cfg(feature = "repeat_by")))]
    fn repeat_by(&self, _by: &UInt32Chunked) -> ListChunked {
        unimplemented!()
    }
    #[cfg(feature = "checked_arithmetic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "checked_arithmetic")))]
    fn checked_div(&self, _rhs: &Series) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "is_first")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_first")))]
    /// Get a mask of the first unique values.
    fn is_first(&self) -> Result<BooleanChunked> {
        unimplemented!()
    }
}

impl<'a> (dyn SeriesTrait + 'a) {
    pub fn unpack<N: 'static>(&self) -> Result<&ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        if &N::get_dtype() == self.dtype() {
            Ok(self.as_ref())
        } else {
            Err(PolarsError::DataTypeMisMatch(
                "cannot unpack Series; data types don't match".into(),
            ))
        }
    }
}

/// # Series
/// The columnar data type for a DataFrame.
///
/// Most of the available functions are definedin the [SeriesTrait trait](crate::series::SeriesTrait).
///
/// The `Series` struct consists
/// of typed [ChunkedArray](../chunked_array/struct.ChunkedArray.html)'s. To quickly cast
/// a `Series` to a `ChunkedArray` you can call the method with the name of the type:
///
/// ```
/// # use polars_core::prelude::*;
/// let s: Series = [1, 2, 3].iter().collect();
/// // Quickly obtain the ChunkedArray wrapped by the Series.
/// let chunked_array = s.i32().unwrap();
/// ```
///
/// ## Arithmetic
///
/// You can do standard arithmetic on series.
/// ```
/// # use polars_core::prelude::*;
/// let s: Series = [1, 2, 3].iter().collect();
/// let out_add = &s + &s;
/// let out_sub = &s - &s;
/// let out_div = &s / &s;
/// let out_mul = &s * &s;
/// ```
///
/// Or with series and numbers.
///
/// ```
/// # use polars_core::prelude::*;
/// let s: Series = (1..3).collect();
/// let out_add_one = &s + 1;
/// let out_multiply = &s * 10;
///
/// // Could not overload left hand side operator.
/// let out_divide = 1.div(&s);
/// let out_add = 1.add(&s);
/// let out_subtract = 1.sub(&s);
/// let out_multiply = 1.mul(&s);
/// ```
///
/// ## Comparison
/// You can obtain boolean mask by comparing series.
///
/// ```
/// # use polars_core::prelude::*;
/// use itertools::Itertools;
/// let s = Series::new("dollars", &[1, 2, 3]);
/// let mask = s.eq(1);
/// let valid = [true, false, false].iter();
/// assert!(mask
///     .into_iter()
///     .map(|opt_bool| opt_bool.unwrap()) // option, because series can be null
///     .zip(valid)
///     .all(|(a, b)| a == *b))
/// ```
///
/// See all the comparison operators in the [CmpOps trait](../chunked_array/comparison/trait.CmpOps.html)
///
/// ## Iterators
/// The Series variants contain differently typed [ChunkedArray's](../chunked_array/struct.ChunkedArray.html).
/// These structs can be turned into iterators, making it possible to use any function/ closure you want
/// on a Series.
///
/// These iterators return an `Option<T>` because the values of a series may be null.
///
/// ```
/// use polars_core::prelude::*;
/// let pi = 3.14;
/// let s = Series::new("angle", [2f32 * pi, pi, 1.5 * pi].as_ref());
/// let s_cos: Series = s.f32()
///                     .expect("series was not an f32 dtype")
///                     .into_iter()
///                     .map(|opt_angle| opt_angle.map(|angle| angle.cos()))
///                     .collect();
/// ```
///
/// ## Creation
/// Series can be create from different data structures. Below we'll show a few ways we can create
/// a Series object.
///
/// ```
/// # use polars_core::prelude::*;
/// // Series van be created from Vec's, slices and arrays
/// Series::new("boolean series", &vec![true, false, true]);
/// Series::new("int series", &[1, 2, 3]);
/// // And can be nullable
/// Series::new("got nulls", &[Some(1), None, Some(2)]);
///
/// // Series can also be collected from iterators
/// let from_iter: Series = (0..10)
///     .into_iter()
///     .collect();
///
/// ```
#[derive(Clone)]
pub struct Series(pub Arc<dyn SeriesTrait>);

impl Series {
    pub(crate) fn get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
        if Arc::weak_count(&self.0) + Arc::strong_count(&self.0) != 1 {
            self.0 = self.0.clone_inner();
        }
        Arc::get_mut(&mut self.0).expect("implementation error")
    }

    /// Rename series.
    pub fn rename(&mut self, name: &str) -> &mut Series {
        self.get_inner_mut().rename(name);
        self
    }

    /// Shrink the capacity of this array to fit it's length.
    pub fn shrink_to_fit(&mut self) {
        self.get_inner_mut().shrink_to_fit()
    }

    /// Append arrow array of same datatype.
    pub fn append_array(&mut self, other: ArrayRef) -> Result<&mut Self> {
        self.get_inner_mut().append_array(other)?;
        Ok(self)
    }

    /// Append a Series of the same type in place.
    pub fn append(&mut self, other: &Series) -> Result<&mut Self> {
        self.get_inner_mut().append(other)?;
        Ok(self)
    }

    /// Sort in place.
    pub fn sort_in_place(&mut self, reverse: bool) -> &mut Self {
        self.get_inner_mut().sort_in_place(reverse);
        self
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> Result<usize> {
        self.get_inner_mut().as_single_ptr()
    }

    /// Cast to some primitive type.
    pub fn cast<N>(&self) -> Result<Self>
    where
        N: PolarsDataType,
    {
        self.0.cast_with_dtype(&N::get_dtype())
    }
    /// Returns `None` if the array is empty or only contains null values.
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
    /// assert_eq!(s.sum(), Some(6));
    /// ```
    pub fn sum<T>(&self) -> Option<T>
    where
        T: NumCast,
    {
        self.sum_as_series()
            .cast::<Float64Type>()
            .ok()
            .and_then(|s| s.f64().unwrap().get(0).and_then(T::from))
    }

    /// Returns the minimum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
    /// assert_eq!(s.min(), Some(1));
    /// ```
    pub fn min<T>(&self) -> Option<T>
    where
        T: NumCast,
    {
        self.min_as_series()
            .cast::<Float64Type>()
            .ok()
            .and_then(|s| s.f64().unwrap().get(0).and_then(T::from))
    }

    /// Returns the maximum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
    /// assert_eq!(s.max(), Some(3));
    /// ```
    pub fn max<T>(&self) -> Option<T>
    where
        T: NumCast,
    {
        self.max_as_series()
            .cast::<Float64Type>()
            .ok()
            .and_then(|s| s.f64().unwrap().get(0).and_then(T::from))
    }

    /// Explode a list or utf8 Series. This expands every item to a new row..
    pub fn explode(&self) -> Result<Series> {
        match self.dtype() {
            DataType::List(_) => self.list().unwrap().explode(),
            DataType::Utf8 => self.utf8().unwrap().explode(),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "explode not supported for Series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is NaN (note this is different than missing/ null)
    pub fn is_nan(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_nan()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "is_nan not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is NaN (note this is different than missing/ null)
    pub fn is_not_nan(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_not_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_not_nan()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "is_nan not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is finite
    pub fn is_finite(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_finite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_finite()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "is_nan not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is finite
    pub fn is_infinite(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_infinite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_infinite()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "is_nan not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    #[cfg(feature = "zip_with")]
    #[cfg_attr(docsrs, doc(cfg(feature = "zip_with")))]
    pub fn zip_with(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        let (lhs, rhs) = coerce_lhs_rhs(self, other)?;
        lhs.zip_with_same_type(mask, rhs.as_ref())
    }

    /// Cast a datelike Series to their physical representation.
    /// Primitives remain unchanged
    ///
    /// * Date32 -> Int32
    /// * Date64 -> Int64
    /// * Time64 -> Int64
    /// * Duration -> Int64
    ///
    pub fn to_physical_repr(&self) -> Series {
        use DataType::*;
        let out = match self.dtype() {
            Date32 => self.cast_with_dtype(&DataType::Int32),
            Date64 => self.cast_with_dtype(&DataType::Int64),
            Time64(_) => self.cast_with_dtype(&DataType::Int64),
            Duration(_) => self.cast_with_dtype(&DataType::Int64),
            _ => return self.clone(),
        };
        out.unwrap()
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_unchecked_threaded(
        &self,
        idx: &UInt32Chunked,
        rechunk: bool,
    ) -> Result<Series> {
        let n_threads = POOL.current_num_threads();
        let idx = split_ca(idx, n_threads)?;

        let series: Result<Vec<_>> =
            POOL.install(|| idx.par_iter().map(|idx| self.take_unchecked(idx)).collect());

        let s = series?
            .into_iter()
            .reduce(|mut s, s1| {
                s.append(&s1).unwrap();
                s
            })
            .unwrap();
        if rechunk {
            Ok(s.rechunk())
        } else {
            Ok(s)
        }
    }

    /// Take by index. This operation is clone.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value
    pub fn take_threaded(&self, idx: &UInt32Chunked, rechunk: bool) -> Series {
        let n_threads = POOL.current_num_threads();
        let idx = split_ca(idx, n_threads).unwrap();

        let series: Vec<_> = POOL.install(|| idx.par_iter().map(|idx| self.take(idx)).collect());

        let s = series
            .into_iter()
            .reduce(|mut s, s1| {
                s.append(&s1).unwrap();
                s
            })
            .unwrap();
        if rechunk {
            s.rechunk()
        } else {
            s
        }
    }

    /// Filter by boolean mask. This operation clones data.
    pub fn filter_threaded(&self, filter: &BooleanChunked, rechunk: bool) -> Result<Series> {
        // this would fail if there is a broadcasting filter.
        // because we cannot split that filter over threads
        // besides they are a no-op, so we do the standard filter.
        if filter.len() == 1 {
            return self.filter(filter);
        }
        let n_threads = POOL.current_num_threads();
        let filters = split_ca(filter, n_threads).unwrap();
        let series = split_series(self, n_threads).unwrap();

        let series: Result<Vec<_>> = POOL.install(|| {
            filters
                .par_iter()
                .zip(series)
                .map(|(filter, s)| s.filter(filter))
                .collect()
        });

        let s = series?
            .into_iter()
            .reduce(|mut s, s1| {
                s.append(&s1).unwrap();
                s
            })
            .unwrap();
        if rechunk {
            Ok(s.rechunk())
        } else {
            Ok(s)
        }
    }

    /// Round underlying floating point array to given decimal.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn round(&self, decimals: u32) -> Result<Self> {
        use num::traits::Pow;
        if let Ok(ca) = self.f32() {
            let multiplier = 10.0.pow(decimals as f32) as f32;
            let s = ca
                .apply(|val| (val * multiplier).round() / multiplier)
                .into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let multiplier = 10.0.pow(decimals as f32) as f64;
            let s = ca
                .apply(|val| (val * multiplier).round() / multiplier)
                .into_series();
            return Ok(s);
        }
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} is not a floating point datatype", self.dtype()).into(),
        ))
    }

    #[cfg(feature = "dot_product")]
    pub fn dot(&self, other: &Series) -> Option<f64> {
        (self * other).sum::<f64>()
    }
}

impl Deref for Series {
    type Target = dyn SeriesTrait;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'a> AsRef<(dyn SeriesTrait + 'a)> for Series {
    fn as_ref(&self) -> &(dyn SeriesTrait + 'a) {
        &*self.0
    }
}

pub trait NamedFrom<T, Phantom: ?Sized> {
    /// Initialize by name and values.
    fn new(name: &str, _: T) -> Self;
}
//
macro_rules! impl_named_from {
    ($type:ty, $series_var:ident, $method:ident) => {
        impl<T: AsRef<$type>> NamedFrom<T, $type> for Series {
            fn new(name: &str, v: T) -> Self {
                ChunkedArray::<$series_var>::$method(name, v.as_ref()).into_series()
            }
        }
    };
}

impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_slice(name, v.as_ref()).into_series()
    }
}
impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_opt_slice(name, v.as_ref()).into_series()
    }
}

impl_named_from!([String], Utf8Type, new_from_slice);
impl_named_from!([bool], BooleanType, new_from_slice);
#[cfg(feature = "dtype-u8")]
impl_named_from!([u8], UInt8Type, new_from_slice);
#[cfg(feature = "dtype-u16")]
impl_named_from!([u16], UInt16Type, new_from_slice);
impl_named_from!([u32], UInt32Type, new_from_slice);
#[cfg(feature = "dtype-u64")]
impl_named_from!([u64], UInt64Type, new_from_slice);
#[cfg(feature = "dtype-i8")]
impl_named_from!([i8], Int8Type, new_from_slice);
#[cfg(feature = "dtype-i16")]
impl_named_from!([i16], Int16Type, new_from_slice);
impl_named_from!([i32], Int32Type, new_from_slice);
impl_named_from!([i64], Int64Type, new_from_slice);
impl_named_from!([f32], Float32Type, new_from_slice);
impl_named_from!([f64], Float64Type, new_from_slice);
impl_named_from!([Option<String>], Utf8Type, new_from_opt_slice);
impl_named_from!([Option<bool>], BooleanType, new_from_opt_slice);
#[cfg(feature = "dtype-u8")]
impl_named_from!([Option<u8>], UInt8Type, new_from_opt_slice);
#[cfg(feature = "dtype-u16")]
impl_named_from!([Option<u16>], UInt16Type, new_from_opt_slice);
impl_named_from!([Option<u32>], UInt32Type, new_from_opt_slice);
#[cfg(feature = "dtype-u64")]
impl_named_from!([Option<u64>], UInt64Type, new_from_opt_slice);
#[cfg(feature = "dtype-i8")]
impl_named_from!([Option<i8>], Int8Type, new_from_opt_slice);
#[cfg(feature = "dtype-i16")]
impl_named_from!([Option<i16>], Int16Type, new_from_opt_slice);
impl_named_from!([Option<i32>], Int32Type, new_from_opt_slice);
impl_named_from!([Option<i64>], Int64Type, new_from_opt_slice);
impl_named_from!([Option<f32>], Float32Type, new_from_opt_slice);
impl_named_from!([Option<f64>], Float64Type, new_from_opt_slice);

impl<T: AsRef<[Series]>> NamedFrom<T, ListType> for Series {
    fn new(name: &str, s: T) -> Self {
        let series_slice = s.as_ref();
        let values_cap = series_slice.iter().fold(0, |acc, s| acc + s.len());

        let dt = series_slice[0].dtype();
        let mut builder = get_list_builder(dt, values_cap, series_slice.len(), name);
        for series in series_slice {
            builder.append_series(series)
        }
        builder.finish().into_series()
    }
}

// TODO: add types
impl std::convert::TryFrom<(&str, Vec<ArrayRef>)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, Vec<ArrayRef>)) -> Result<Self> {
        let (name, chunks) = name_arr;

        let mut chunks_iter = chunks.iter();
        let data_type: &ArrowDataType = chunks_iter
            .next()
            .ok_or_else(|| PolarsError::NoData("Expected at least on ArrayRef".into()))?
            .data_type();

        for chunk in chunks_iter {
            if chunk.data_type() != data_type {
                return Err(PolarsError::InvalidOperation(
                    "Cannot create series from multiple arrays with different types".into(),
                ));
            }
        }

        match data_type {
            ArrowDataType::LargeUtf8 => {
                Ok(Utf8Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Utf8 => {
                let chunks = chunks
                    .iter()
                    .map(|arr| cast(arr, &ArrowDataType::LargeUtf8).unwrap())
                    .collect_vec();
                Ok(Utf8Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::List(fld) => {
                let chunks = chunks
                    .iter()
                    .map(|arr| cast(arr, &ArrowDataType::LargeList(fld.clone())).unwrap())
                    .collect();
                Ok(ListChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Boolean => {
                Ok(BooleanChunked::new_from_chunks(name, chunks).into_series())
            }
            #[cfg(feature = "dtype-u8")]
            ArrowDataType::UInt8 => Ok(UInt8Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-u16")]
            ArrowDataType::UInt16 => Ok(UInt16Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt32 => Ok(UInt32Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-u64")]
            ArrowDataType::UInt64 => Ok(UInt64Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i8")]
            ArrowDataType::Int8 => Ok(Int8Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i16")]
            ArrowDataType::Int16 => Ok(Int16Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int32 => Ok(Int32Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int64 => Ok(Int64Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Float32 => {
                Ok(Float32Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Float64 => {
                Ok(Float64Chunked::new_from_chunks(name, chunks).into_series())
            }
            #[cfg(feature = "dtype-date32")]
            ArrowDataType::Date32 => Ok(Date32Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-date64")]
            ArrowDataType::Date64 => Ok(Date64Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-time64-ns")]
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                Ok(Time64NanosecondChunked::new_from_chunks(name, chunks).into_series())
            }
            #[cfg(feature = "dtype-duration-ns")]
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                Ok(DurationNanosecondChunked::new_from_chunks(name, chunks).into_series())
            }
            #[cfg(feature = "dtype-duration-ms")]
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                Ok(DurationMillisecondChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::LargeList(_) => {
                Ok(ListChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Null => {
                // we don't support null types yet so we use a small digit type filled with nulls
                let len = chunks.iter().fold(0, |acc, array| acc + array.len());
                #[cfg(feature = "dtype-i8")]
                return Ok(Int8Chunked::full_null(name, len).into_series());
                #[cfg(not(feature = "dtype-i8"))]
                Ok(UInt32Chunked::full_null(name, len).into_series())
            }
            #[cfg(feature = "dtype-date64")]
            ArrowDataType::Timestamp(TimeUnit::Millisecond, None) => {
                let chunks = chunks
                    .iter()
                    .map(|arr| cast(arr, &ArrowDataType::Date64).unwrap())
                    .collect();
                Ok(Date64Chunked::new_from_chunks(name, chunks).into_series())
            }
            dt => Err(PolarsError::InvalidOperation(
                format!("Cannot create polars series from {:?} type", dt).into(),
            )),
        }
    }
}

impl TryFrom<(&str, ArrayRef)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, ArrayRef)) -> Result<Self> {
        let (name, arr) = name_arr;
        Series::try_from((name, vec![arr]))
    }
}

impl Default for Series {
    fn default() -> Self {
        Int64Chunked::default().into_series()
    }
}

impl<T> From<ChunkedArray<T>> for Series
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
{
    fn from(ca: ChunkedArray<T>) -> Self {
        ca.into_series()
    }
}

impl IntoSeries for Arc<dyn SeriesTrait> {
    fn into_series(self) -> Series {
        Series(self)
    }
}

impl IntoSeries for Series {
    fn into_series(self) -> Series {
        self
    }
}

impl<'a, T> AsRef<ChunkedArray<T>> for dyn SeriesTrait + 'a
where
    T: 'static + PolarsDataType,
{
    fn as_ref(&self) -> &ChunkedArray<T> {
        if &T::get_dtype() == self.dtype() ||
            // needed because we want to get ref of List no matter what the inner type is.
            (matches!(T::get_dtype(), DataType::List(_)) && matches!(self.dtype(), DataType::List(_)) )
        {
            unsafe { &*(self as *const dyn SeriesTrait as *const ChunkedArray<T>) }
        } else {
            panic!(
                "implementation error, cannot get ref {:?} from {:?}",
                T::get_dtype(),
                self.dtype()
            )
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::series::*;
    use arrow::array::*;

    #[test]
    fn cast() {
        let ar = UInt32Chunked::new_from_slice("a", &[1, 2]);
        let s = ar.into_series();
        let s2 = s.cast::<Int64Type>().unwrap();

        assert!(s2.i64().is_ok());
        let s2 = s.cast::<Float32Type>().unwrap();
        assert!(s2.f32().is_ok());
    }

    #[test]
    fn new_series() {
        Series::new("boolean series", &vec![true, false, true]);
        Series::new("int series", &[1, 2, 3]);
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        ca.into_series();
    }

    #[test]
    fn new_series_from_arrow_primitive_array() {
        let array = UInt32Array::from(vec![1, 2, 3, 4, 5]);
        let array_ref: ArrayRef = Arc::new(array);

        Series::try_from(("foo", array_ref)).unwrap();
    }

    #[test]
    fn series_append() {
        let mut s1 = Series::new("a", &[1, 2]);
        let s2 = Series::new("b", &[3]);
        s1.append(&s2).unwrap();
        assert_eq!(s1.len(), 3);

        // add wrong type
        let s2 = Series::new("b", &[3.0]);
        assert!(s1.append(&s2).is_err())
    }

    #[test]
    fn series_slice_works() {
        let series = Series::new("a", &[1i64, 2, 3, 4, 5]);

        let slice_1 = series.slice(-3, 3);
        let slice_2 = series.slice(-5, 5);
        let slice_3 = series.slice(0, 5);

        assert_eq!(slice_1.get(0), AnyValue::Int64(3));
        assert_eq!(slice_2.get(0), AnyValue::Int64(1));
        assert_eq!(slice_3.get(0), AnyValue::Int64(1));
    }

    #[test]
    fn out_of_range_slice_does_not_panic() {
        let series = Series::new("a", &[1i64, 2, 3, 4, 5]);

        series.slice(-3, 4);
        series.slice(-6, 2);
        series.slice(4, 2);
    }

    #[test]
    #[cfg(feature = "round_series")]
    fn test_round_series() {
        let series = Series::new("a", &[1.003, 2.23222, 3.4352]);
        let out = series.round(2).unwrap();
        let ca = out.f64().unwrap();
        assert_eq!(ca.get(0), Some(1.0));
    }
}
