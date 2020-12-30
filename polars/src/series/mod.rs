//! Type agnostic columnar data structure.
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::{array::ArrayRef, buffer::Buffer};
pub(crate) mod arithmetic;
mod comparison;
pub mod implementations;
pub(crate) mod iterator;

use crate::chunked_array::builder::get_list_builder;
use arrow::array::ArrayDataRef;
use num::NumCast;
use std::any::Any;
use std::convert::TryFrom;
use std::ops::Deref;
use std::sync::Arc;

pub(crate) mod private {
    use super::*;
    use crate::frame::group_by::PivotAgg;

    pub trait PrivateSeries {
        fn agg_mean(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_min(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_max(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_sum(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_std(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_var(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_first(&self, _groups: &[(usize, Vec<usize>)]) -> Series {
            unimplemented!()
        }
        fn agg_last(&self, _groups: &[(usize, Vec<usize>)]) -> Series {
            unimplemented!()
        }
        fn agg_n_unique(&self, _groups: &[(usize, Vec<usize>)]) -> Option<UInt32Chunked> {
            unimplemented!()
        }
        fn agg_list(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn agg_quantile(&self, _groups: &[(usize, Vec<usize>)], _quantile: f64) -> Option<Series> {
            unimplemented!()
        }
        fn agg_median(&self, _groups: &[(usize, Vec<usize>)]) -> Option<Series> {
            unimplemented!()
        }
        fn pivot<'a>(
            &self,
            _pivot_series: &'a (dyn SeriesTrait + 'a),
            _keys: Vec<Series>,
            _groups: &[(usize, Vec<usize>)],
            _agg_type: PivotAgg,
        ) -> Result<DataFrame> {
            unimplemented!()
        }

        fn pivot_count<'a>(
            &self,
            _pivot_series: &'a (dyn SeriesTrait + 'a),
            _keys: Vec<Series>,
            _groups: &[(usize, Vec<usize>)],
        ) -> Result<DataFrame> {
            unimplemented!()
        }

        fn hash_join_inner(&self, _other: &Series) -> Vec<(usize, usize)> {
            unimplemented!()
        }
        fn hash_join_left(&self, _other: &Series) -> Vec<(usize, Option<usize>)> {
            unimplemented!()
        }
        fn hash_join_outer(&self, _other: &Series) -> Vec<(Option<usize>, Option<usize>)> {
            unimplemented!()
        }
        fn zip_outer_join_column(
            &self,
            _right_column: &Series,
            _opt_join_tuples: &[(Option<usize>, Option<usize>)],
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
        fn group_tuples(&self) -> Vec<(usize, Vec<usize>)> {
            unimplemented!()
        }
    }
}

pub trait SeriesTrait: Send + Sync + private::PrivateSeries {
    /// Rename the Series.
    fn rename(&mut self, name: &str);

    /// Get Arrow ArrayData
    fn array_data(&self) -> Vec<ArrayDataRef> {
        unimplemented!()
    }

    /// Get the lengths of the underlying chunks
    fn chunk_lengths(&self) -> &Vec<usize> {
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
    fn dtype(&self) -> &ArrowDataType {
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
    /// # use polars::prelude::*;
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

    /// Append Arrow array of same dtype to this Series.
    fn append_array(&mut self, _other: ArrayRef) -> Result<()> {
        unimplemented!()
    }

    /// Take `num_elements` from the top as a zero copy view.
    fn limit(&self, num_elements: usize) -> Result<Series> {
        self.slice(0, num_elements)
    }

    /// Get a zero copy view of the data.
    fn slice(&self, _offset: usize, _length: usize) -> Result<Series> {
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
    /// Out of bounds access doesn't Error but will return a Null value
    fn take_iter(
        &self,
        _iter: &mut dyn Iterator<Item = usize>,
        _capacity: Option<usize>,
    ) -> Series {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// This doesn't check any bounds or null validity.
    unsafe fn take_iter_unchecked(
        &self,
        _iter: &mut dyn Iterator<Item = usize>,
        _capacity: Option<usize>,
    ) -> Series {
        unimplemented!()
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    unsafe fn take_from_single_chunked(&self, _idx: &UInt32Chunked) -> Result<Series> {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// This doesn't check any bounds or null validity.
    unsafe fn take_opt_iter_unchecked(
        &self,
        _iter: &mut dyn Iterator<Item = Option<usize>>,
        _capacity: Option<usize>,
    ) -> Series {
        unimplemented!()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value
    fn take_opt_iter(
        &self,
        _iter: &mut dyn Iterator<Item = Option<usize>>,
        _capacity: Option<usize>,
    ) -> Series {
        unimplemented!()
    }

    /// Take by index. This operation is clone.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value
    fn take(&self, indices: &dyn AsTakeIndex) -> Series {
        let mut iter = indices.as_take_iter();
        let capacity = indices.take_index_len();
        self.take_iter(&mut iter, Some(capacity))
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
    fn rechunk(&self, _chunk_lengths: Option<&[usize]>) -> Result<Series> {
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

    /// Drop all null values and return a new Series.
    fn drop_nulls(&self) -> Series {
        if self.null_count() == 0 {
            Series(self.clone_inner())
        } else {
            self.filter(&self.is_not_null()).unwrap()
        }
    }

    /// Create a new Series filled with values at that index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// let s = Series::new("a", [0i32, 1, 8]);
    /// let expanded = s.expand_at_index(2, 4);
    /// assert_eq!(Vec::from(expanded.i32().unwrap()), &[Some(8), Some(8), Some(8), Some(8)])
    /// ```
    fn expand_at_index(&self, _index: usize, _length: usize) -> Series {
        unimplemented!()
    }

    fn cast_with_arrow_datatype(&self, _data_type: &ArrowDataType) -> Result<Series> {
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
    fn get(&self, _index: usize) -> AnyType {
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
    fn argsort(&self, _reverse: bool) -> Vec<usize> {
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
    fn arg_unique(&self) -> Result<Vec<usize>> {
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
    fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
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
    /// # use polars::prelude::*;
    /// fn example() -> Result<()> {
    ///     let s = Series::new("series", &[1, 2, 3]);
    ///
    ///     let shifted = s.shift(1)?;
    ///     assert_eq!(Vec::from(shifted.i32()?), &[None, Some(1), Some(2)]);
    ///
    ///     let shifted = s.shift(-1)?;
    ///     assert_eq!(Vec::from(shifted.i32()?), &[Some(2), Some(3), None]);
    ///
    ///     let shifted = s.shift(2)?;
    ///     assert_eq!(Vec::from(shifted.i32()?), &[None, None, Some(1)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    fn shift(&self, _periods: i32) -> Result<Series> {
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
    /// # use polars::prelude::*;
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

    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    fn zip_with(&self, _mask: &BooleanChunked, _other: &Series) -> Result<Series> {
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
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
    ) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling sum to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_sum).
    fn rolling_sum(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
    ) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling min to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_min).
    fn rolling_min(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
    ) -> Result<Series> {
        unimplemented!()
    }
    /// Apply a rolling max to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_max).
    fn rolling_max(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
    ) -> Result<Series> {
        unimplemented!()
    }

    fn fmt_list(&self) -> String {
        "fmt implemented".into()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal_day(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Result<Series> {
        unimplemented!()
    }

    #[cfg(feature = "temporal")]
    #[doc(cfg(feature = "temporal"))]
    /// Format Date32/Date64 with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn datetime_str_fmt(&self, fmt: &str) -> Result<Series> {
        match self.dtype() {
            ArrowDataType::Date32(_) => self.date32().map(|ca| ca.str_fmt(fmt).into_series()),
            ArrowDataType::Date64(_) => self.date64().map(|ca| ca.str_fmt(fmt).into_series()),
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
    #[doc(cfg(feature = "random"))]
    /// Sample n datapoints from this Series.
    fn sample_n(&self, n: usize, with_replacement: bool) -> Result<Series>;

    #[cfg(feature = "random")]
    #[doc(cfg(feature = "random"))]
    /// Sample a fraction between 0.0-1.0 of this ChunkedArray.
    fn sample_frac(&self, frac: f64, with_replacement: bool) -> Result<Series>;

    /// Get the value at this index as a downcastable Any trait ref.
    fn get_as_any(&self, _index: usize) -> &dyn Any {
        unimplemented!()
    }
}

impl<'a> (dyn SeriesTrait + 'a) {
    pub fn unpack<N: 'static>(&self) -> Result<&ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        if &N::get_data_type() == self.dtype() {
            Ok(self.as_ref())
        } else {
            Err(PolarsError::DataTypeMisMatch(
                "cannot unpack Series; data types don't match".into(),
            ))
        }
    }
}

/// # Series
/// The columnar data type for a DataFrame. The [Series enum](enum.Series.html) consists
/// of typed [ChunkedArray](../chunked_array/struct.ChunkedArray.html)'s. To quickly cast
/// a `Series` to a `ChunkedArray` you can call the method with the name of the type:
///
/// ```
/// # use polars::prelude::*;
/// let s: Series = [1, 2, 3].iter().collect();
/// // Quickly obtain the ChunkedArray wrapped by the Series.
/// let chunked_array = s.i32().unwrap();
/// ```
///
/// ## Arithmetic
///
/// You can do standard arithmetic on series.
/// ```
/// # use polars::prelude::*;
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
/// # use polars::prelude::*;
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
/// # use polars::prelude::*;
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
/// use polars::prelude::*;
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
/// # use polars::prelude::*;
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
    fn get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
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

    /// Append arrow array of same datatype.
    fn append_array(&mut self, other: ArrayRef) -> Result<&mut Self> {
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
        self.0.cast_with_arrow_datatype(&N::get_data_type())
    }
    /// Returns `None` if the array is empty or only contains null values.
    /// ```
    /// # use polars::prelude::*;
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
    /// # use polars::prelude::*;
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
    /// # use polars::prelude::*;
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

    /// Returns the mean value in the array
    /// Returns an option because the array is nullable.
    pub fn mean<T>(&self) -> Option<T>
    where
        T: NumCast,
    {
        self.cast::<Float64Type>()
            .ok()
            .map(|s| s.mean_as_series())
            .and_then(|s| s.f64().unwrap().get(0).and_then(T::from))
    }

    /// Explode a list or utf8 Series. This expands every item to a new row..
    pub fn explode(&self) -> Result<Series> {
        match self.dtype() {
            ArrowDataType::List(_) => self.list().unwrap().explode(),
            ArrowDataType::Utf8 => self.utf8().unwrap().explode(),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "explode not supported for Series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
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
impl_named_from!([u8], UInt8Type, new_from_slice);
impl_named_from!([u16], UInt16Type, new_from_slice);
impl_named_from!([u32], UInt32Type, new_from_slice);
impl_named_from!([u64], UInt64Type, new_from_slice);
impl_named_from!([i8], Int8Type, new_from_slice);
impl_named_from!([i16], Int16Type, new_from_slice);
impl_named_from!([i32], Int32Type, new_from_slice);
impl_named_from!([i64], Int64Type, new_from_slice);
impl_named_from!([f32], Float32Type, new_from_slice);
impl_named_from!([f64], Float64Type, new_from_slice);
impl_named_from!([Option<String>], Utf8Type, new_from_opt_slice);
impl_named_from!([Option<bool>], BooleanType, new_from_opt_slice);
impl_named_from!([Option<u8>], UInt8Type, new_from_opt_slice);
impl_named_from!([Option<u16>], UInt16Type, new_from_opt_slice);
impl_named_from!([Option<u32>], UInt32Type, new_from_opt_slice);
impl_named_from!([Option<u64>], UInt64Type, new_from_opt_slice);
impl_named_from!([Option<i8>], Int8Type, new_from_opt_slice);
impl_named_from!([Option<i16>], Int16Type, new_from_opt_slice);
impl_named_from!([Option<i32>], Int32Type, new_from_opt_slice);
impl_named_from!([Option<i64>], Int64Type, new_from_opt_slice);
impl_named_from!([Option<f32>], Float32Type, new_from_opt_slice);
impl_named_from!([Option<f64>], Float64Type, new_from_opt_slice);

impl<T: AsRef<[Series]>> NamedFrom<T, ListType> for Series {
    fn new(name: &str, s: T) -> Self {
        let series_slice = s.as_ref();
        let dt = series_slice[0].dtype();
        let mut builder = get_list_builder(dt, series_slice.len(), name);
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
            ArrowDataType::Utf8 => Ok(Utf8Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Boolean => {
                Ok(BooleanChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::UInt8 => Ok(UInt8Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt16 => Ok(UInt16Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt32 => Ok(UInt32Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt64 => Ok(UInt64Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int8 => Ok(Int8Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int16 => Ok(Int16Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int32 => Ok(Int32Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int64 => Ok(Int64Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Float32 => {
                Ok(Float32Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Float64 => {
                Ok(Float64Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Date32(DateUnit::Day) => {
                Ok(Date32Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                Ok(Date64Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                Ok(Time64NanosecondChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                Ok(DurationNanosecondChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                Ok(DurationMillisecondChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::List(_) => Ok(ListChunked::new_from_chunks(name, chunks).into_series()),
            dt => Err(PolarsError::InvalidOperation(
                format!("Cannot create polars series from {:?}", dt).into(),
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
        UInt8Chunked::default().into_series()
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
        let array = UInt64Array::from(vec![1, 2, 3, 4, 5]);
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
}
