pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::array::ArrayRef;

#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::chunked_array::ChunkIdIter;
#[cfg(feature = "object")]
use std::any::Any;
use std::borrow::Cow;
#[cfg(feature = "temporal")]
use std::ops::Deref;
use std::sync::Arc;

macro_rules! invalid_operation {
    ($s:expr) => {
        Err(PolarsError::InvalidOperation(
            format!(
                "this operation is not implemented/valid for this dtype: {:?}",
                $s._dtype()
            )
            .into(),
        ))
    };
}
macro_rules! invalid_operation_panic {
    ($s:expr) => {
        panic!(
            "this operation is not implemented/valid for this dtype: {:?}",
            $s._dtype()
        )
    };
}

pub(crate) mod private {
    use super::*;
    #[cfg(feature = "pivot")]
    use crate::frame::groupby::pivot::PivotAgg;
    use crate::frame::groupby::GroupTuples;

    use crate::chunked_array::ops::compare_inner::{PartialEqInner, PartialOrdInner};
    use ahash::RandomState;

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
        /// Get field (used in schema)
        fn _field(&self) -> Cow<Field> {
            invalid_operation_panic!(self)
        }

        fn _dtype(&self) -> &DataType {
            unimplemented!()
        }

        fn explode_by_offsets(&self, _offsets: &[i64]) -> Series {
            invalid_operation_panic!(self)
        }

        /// Apply a rolling mean to a Series. See:
        /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_mean).
        #[cfg(feature = "rolling_window")]
        fn _rolling_mean(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }
        /// Apply a rolling sum to a Series. See:
        #[cfg(feature = "rolling_window")]
        fn _rolling_sum(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }
        /// Apply a rolling min to a Series. See:
        #[cfg(feature = "rolling_window")]
        fn _rolling_min(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }
        /// Apply a rolling max to a Series. See:
        #[cfg(feature = "rolling_window")]
        fn _rolling_max(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }

        /// Apply a rolling variance to a Series. See:
        #[cfg(feature = "rolling_window")]
        fn _rolling_var(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }

        /// Apply a rolling std_dev to a Series. See:
        #[cfg(feature = "rolling_window")]
        fn _rolling_std(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }

        /// Get an array with the cumulative max computed at every element
        #[cfg(feature = "cum_agg")]
        fn _cummax(&self, _reverse: bool) -> Series {
            panic!("operation cummax not supported for this dtype")
        }

        /// Get an array with the cumulative min computed at every element
        #[cfg(feature = "cum_agg")]
        fn _cummin(&self, _reverse: bool) -> Series {
            panic!("operation cummin not supported for this dtype")
        }

        /// Get an array with the cumulative sum computed at every element
        #[cfg(feature = "cum_agg")]
        fn _cumsum(&self, _reverse: bool) -> Series {
            panic!("operation cumsum not supported for this dtype")
        }

        /// Get an array with the cumulative sum computed at every element
        #[cfg(feature = "cum_agg")]
        fn _cumprod(&self, _reverse: bool) -> Series {
            panic!("operation cumprod not supported for this dtype")
        }

        #[cfg(feature = "asof_join")]
        fn join_asof(&self, _other: &Series) -> Result<Vec<Option<u32>>> {
            invalid_operation!(self)
        }

        fn set_sorted(&mut self, _reverse: bool) {
            invalid_operation_panic!(self)
        }

        unsafe fn equal_element(
            &self,
            _idx_self: usize,
            _idx_other: usize,
            _other: &Series,
        ) -> bool {
            invalid_operation_panic!(self)
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
            invalid_operation_panic!(self)
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
            invalid_operation_panic!(self)
        }
        fn vec_hash(&self, _build_hasher: RandomState) -> AlignedVec<u64> {
            invalid_operation_panic!(self)
        }
        fn vec_hash_combine(&self, _build_hasher: RandomState, _hashes: &mut [u64]) {
            invalid_operation_panic!(self)
        }
        fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        fn agg_min(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        fn agg_max(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
        /// first cast to `Int64` to prevent overflow issues.
        fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        fn agg_first(&self, _groups: &[(u32, Vec<u32>)]) -> Series {
            invalid_operation_panic!(self)
        }
        fn agg_last(&self, _groups: &[(u32, Vec<u32>)]) -> Series {
            invalid_operation_panic!(self)
        }
        fn agg_n_unique(&self, _groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
            None
        }
        fn agg_list(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        fn agg_quantile(
            &self,
            _groups: &[(u32, Vec<u32>)],
            _quantile: f64,
            _interpol: QuantileInterpolOptions,
        ) -> Option<Series> {
            None
        }
        fn agg_median(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        #[cfg(feature = "lazy")]
        fn agg_valid_count(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
            None
        }
        #[cfg(feature = "pivot")]
        fn pivot<'a>(
            &self,
            _pivot_series: &'a Series,
            _keys: Vec<Series>,
            _groups: &[(u32, Vec<u32>)],
            _agg_type: PivotAgg,
        ) -> Result<DataFrame> {
            invalid_operation_panic!(self)
        }

        #[cfg(feature = "pivot")]
        fn pivot_count<'a>(
            &self,
            _pivot_series: &'a Series,
            _keys: Vec<Series>,
            _groups: &[(u32, Vec<u32>)],
        ) -> Result<DataFrame> {
            invalid_operation_panic!(self)
        }

        fn hash_join_inner(&self, _other: &Series) -> Vec<(u32, u32)> {
            invalid_operation_panic!(self)
        }
        fn hash_join_left(&self, _other: &Series) -> Vec<(u32, Option<u32>)> {
            invalid_operation_panic!(self)
        }
        fn hash_join_outer(&self, _other: &Series) -> Vec<(Option<u32>, Option<u32>)> {
            invalid_operation_panic!(self)
        }
        fn zip_outer_join_column(
            &self,
            _right_column: &Series,
            _opt_join_tuples: &[(Option<u32>, Option<u32>)],
        ) -> Series {
            invalid_operation_panic!(self)
        }

        fn subtract(&self, _rhs: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        fn add_to(&self, _rhs: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        fn multiply(&self, _rhs: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        fn divide(&self, _rhs: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        fn remainder(&self, _rhs: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        fn group_tuples(&self, _multithreaded: bool) -> GroupTuples {
            invalid_operation_panic!(self)
        }
        fn zip_with_same_type(&self, _mask: &BooleanChunked, _other: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        #[cfg(feature = "sort_multiple")]
        fn argsort_multiple(&self, _by: &[Series], _reverse: &[bool]) -> Result<UInt32Chunked> {
            Err(PolarsError::InvalidOperation(
                "argsort_multiple is not implemented for this Series".into(),
            ))
        }
        /// Formatted string representation. Can used in formatting.
        fn str_value(&self, _index: usize) -> Cow<str> {
            invalid_operation_panic!(self)
        }
    }
}

pub trait SeriesTrait:
    Send + Sync + private::PrivateSeries + private::PrivateSeriesNumeric
{
    #[cfg(feature = "interpolate")]
    #[cfg_attr(docsrs, doc(cfg(feature = "interpolate")))]
    fn interpolate(&self) -> Series;

    /// Rename the Series.
    fn rename(&mut self, name: &str);

    fn bitand(&self, _other: &Series) -> Result<Series> {
        panic!(
            "bitwise and operation not supported for dtype {:?}",
            self.dtype()
        )
    }

    fn bitor(&self, _other: &Series) -> Result<Series> {
        panic!(
            "bitwise or operation not fit supported for dtype {:?}",
            self.dtype()
        )
    }

    fn bitxor(&self, _other: &Series) -> Result<Series> {
        panic!(
            "bitwise xor operation not fit supported for dtype {:?}",
            self.dtype()
        )
    }

    /// Get the lengths of the underlying chunks
    fn chunk_lengths(&self) -> ChunkIdIter {
        invalid_operation_panic!(self)
    }
    /// Name of series.
    fn name(&self) -> &str {
        invalid_operation_panic!(self)
    }

    /// Get field (used in schema)
    fn field(&self) -> Cow<Field> {
        self._field()
    }

    /// Get datatype of series.
    fn dtype(&self) -> &DataType {
        self._dtype()
    }

    /// Underlying chunks.
    fn chunks(&self) -> &Vec<ArrayRef> {
        invalid_operation_panic!(self)
    }

    /// Number of chunks in this Series
    fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    /// Shrink the capacity of this array to fit it's length.
    fn shrink_to_fit(&mut self) {
        panic!("shrink to fit not supported for dtype {:?}", self.dtype())
    }

    /// Unpack to ChunkedArray of dtype i8
    fn i8(&self) -> Result<&Int8Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != i8", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray i16
    fn i16(&self) -> Result<&Int16Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != i16", self.dtype()).into(),
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
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != i32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype i64
    fn i64(&self) -> Result<&Int64Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != i64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype f32
    fn f32(&self) -> Result<&Float32Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != f32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype f64
    fn f64(&self) -> Result<&Float64Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != f64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u8
    fn u8(&self) -> Result<&UInt8Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != u8", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u16
    fn u16(&self) -> Result<&UInt16Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != u16", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u32
    fn u32(&self) -> Result<&UInt32Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != u32", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype u64
    fn u64(&self) -> Result<&UInt64Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != u64", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype bool
    fn bool(&self) -> Result<&BooleanChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != bool", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype utf8
    fn utf8(&self) -> Result<&Utf8Chunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != utf8", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype Time
    fn time(&self) -> Result<&TimeChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != Time", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype Date
    fn date(&self) -> Result<&DateChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!(" Series dtype {:?} != Date", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype datetime
    fn datetime(&self) -> Result<&DatetimeChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != datetime", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype list
    fn list(&self) -> Result<&ListChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != list", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype categorical
    fn categorical(&self) -> Result<&CategoricalChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != categorical", self.dtype()).into(),
        ))
    }

    /// Append Arrow array of same dtype to this Series.
    fn append_array(&mut self, _other: ArrayRef) -> Result<()> {
        invalid_operation_panic!(self)
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
        invalid_operation_panic!(self)
    }

    /// Append a Series of the same type in place.
    fn append(&mut self, _other: &Series) -> Result<()> {
        invalid_operation_panic!(self)
    }

    /// Filter by boolean mask. This operation clones data.
    fn filter(&self, _filter: &BooleanChunked) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Take by index from an iterator. This operation clones the data.
    fn take_iter(&self, _iter: &mut dyn TakeIterator) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// This doesn't check any bounds.
    unsafe fn take_iter_unchecked(&self, _iter: &mut dyn TakeIterator) -> Series {
        invalid_operation_panic!(self)
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_unchecked(&self, _idx: &UInt32Chunked) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// This doesn't check any bounds.
    unsafe fn take_opt_iter_unchecked(&self, _iter: &mut dyn TakeIteratorNulls) -> Series {
        invalid_operation_panic!(self)
    }

    /// Take by index from an iterator. This operation clones the data.
    #[cfg(feature = "take_opt_iter")]
    #[cfg_attr(docsrs, doc(cfg(feature = "take_opt_iter")))]
    fn take_opt_iter(&self, _iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Take by index. This operation is clone.
    fn take(&self, _indices: &UInt32Chunked) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Get length of series.
    fn len(&self) -> usize {
        invalid_operation_panic!(self)
    }

    /// Check if Series is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Aggregate all chunks to a contiguous array of memory.
    fn rechunk(&self) -> Series {
        invalid_operation_panic!(self)
    }

    /// Get the head of the Series.
    fn head(&self, _length: Option<usize>) -> Series {
        invalid_operation_panic!(self)
    }

    /// Get the tail of the Series.
    fn tail(&self, _length: Option<usize>) -> Series {
        invalid_operation_panic!(self)
    }

    /// Take every nth value as a new Series
    fn take_every(&self, n: usize) -> Series;

    /// Drop all null values and return a new Series.
    fn drop_nulls(&self) -> Series {
        if !self.has_validity() {
            Series(self.clone_inner())
        } else {
            self.filter(&self.is_not_null()).unwrap()
        }
    }

    /// Returns the mean value in the array
    /// Returns an option because the array is nullable.
    fn mean(&self) -> Option<f64> {
        None
    }

    /// Returns the median value in the array
    /// Returns an option because the array is nullable.
    fn median(&self) -> Option<f64> {
        None
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
        invalid_operation_panic!(self)
    }

    fn cast(&self, _data_type: &DataType) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Create dummy variables. See [DataFrame](DataFrame::to_dummies)
    fn to_dummies(&self) -> Result<DataFrame> {
        invalid_operation_panic!(self)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        invalid_operation_panic!(self)
    }

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    fn get(&self, _index: usize) -> AnyValue {
        invalid_operation_panic!(self)
    }

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    ///
    /// # Safety
    /// Does not do any bounds checking
    unsafe fn get_unchecked(&self, _index: usize) -> AnyValue {
        invalid_operation_panic!(self)
    }

    fn sort_with(&self, _options: SortOptions) -> Series {
        invalid_operation_panic!(self)
    }

    /// Retrieve the indexes needed for a sort.
    fn argsort(&self, _reverse: bool) -> UInt32Chunked {
        invalid_operation_panic!(self)
    }

    /// Count the null values.
    fn null_count(&self) -> usize {
        invalid_operation_panic!(self)
    }

    /// Return if any the chunks in this `[ChunkedArray]` have a validity bitmap.
    /// no bitmap means no null values.
    fn has_validity(&self) -> bool;

    /// Get unique values in the Series.
    fn unique(&self) -> Result<Series> {
        invalid_operation!(self)
    }

    /// Get unique values in the Series.
    fn n_unique(&self) -> Result<usize> {
        invalid_operation_panic!(self)
    }

    /// Get first indexes of unique values.
    fn arg_unique(&self) -> Result<UInt32Chunked> {
        invalid_operation_panic!(self)
    }

    /// Get min index
    fn arg_min(&self) -> Option<usize> {
        None
    }

    /// Get max index
    fn arg_max(&self) -> Option<usize> {
        None
    }

    /// Get indexes that evaluate true
    fn arg_true(&self) -> Result<UInt32Chunked> {
        Err(PolarsError::InvalidOperation(
            "arg_true can only be called for boolean dtype".into(),
        ))
    }

    /// Get a mask of the null values.
    fn is_null(&self) -> BooleanChunked {
        invalid_operation_panic!(self)
    }

    /// Get a mask of the non-null values.
    fn is_not_null(&self) -> BooleanChunked {
        invalid_operation_panic!(self)
    }

    /// Get a mask of all the unique values.
    fn is_unique(&self) -> Result<BooleanChunked> {
        invalid_operation_panic!(self)
    }

    /// Get a mask of all the duplicated values.
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        invalid_operation_panic!(self)
    }

    /// return a Series in reversed order
    fn reverse(&self) -> Series {
        invalid_operation_panic!(self)
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
        invalid_operation_panic!(self)
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`fill_null` operation on `ChunkedArray<T>`](../chunked_array/ops/trait.ChunkFillNull.html)*.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example() -> Result<()> {
    ///     let s = Series::new("some_missing", &[Some(1), None, Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Forward)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Backward)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Min)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Max)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Mean)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    fn fill_null(&self, _strategy: FillNullStrategy) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Get the sum of the Series as a new Series of length 1.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    fn _sum_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the max of the Series as a new Series of length 1.
    fn max_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the min of the Series as a new Series of length 1.
    fn min_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the mean of the Series as a new Series of length 1.
    fn mean_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the median of the Series as a new Series of length 1.
    fn median_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the variance of the Series as a new Series of length 1.
    fn var_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the standard deviation of the Series as a new Series of length 1.
    fn std_as_series(&self) -> Series {
        invalid_operation_panic!(self)
    }
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    fn fmt_list(&self) -> String {
        "fmt implemented".into()
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.hour()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.hour()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.minute()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.minute()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.second()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.second()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.nanosecond()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.nanosecond()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.day()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.day()),
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
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.weekday()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.weekday()),
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
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.week()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.week()),
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
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.ordinal()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.ordinal()),
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
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.month()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.month()),
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
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.year()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(feature = "temporal")]
    #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
    /// Format Date/Datetimewith a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn strftime(&self, fmt: &str) -> Result<Series> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.strftime(fmt).into_series()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime => self.datetime().map(|ca| ca.strftime(fmt).into_series()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.strftime(fmt).into_series()),
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
            DataType::Date => self
                .cast(&DataType::Datetime)
                .unwrap()
                .datetime()
                .map(|ca| (ca.deref() * 1000)),
            DataType::Datetime => self.datetime().map(|ca| ca.deref().clone() / 1_000_000),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Clone inner ChunkedArray and wrap in a new Arc
    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        invalid_operation_panic!(self)
    }

    #[cfg(feature = "object")]
    #[cfg_attr(docsrs, doc(cfg(feature = "object")))]
    /// Get the value at this index as a downcastable Any trait ref.
    fn get_object(&self, _index: usize) -> Option<&dyn PolarsObjectSafe> {
        invalid_operation_panic!(self)
    }

    /// Get a hold to self as `Any` trait reference.
    /// Only implemented for ObjectType
    #[cfg(feature = "object")]
    #[cfg_attr(docsrs, doc(cfg(feature = "object")))]
    fn as_any(&self) -> &dyn Any {
        invalid_operation_panic!(self)
    }

    /// Raise a numeric series to the power of exponent.
    fn pow(&self, _exponent: f64) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!("power operation not supported on dtype {:?}", self.dtype()).into(),
        ))
    }

    /// Get a boolean mask of the local maximum peaks.
    fn peak_max(&self) -> BooleanChunked {
        invalid_operation_panic!(self)
    }

    /// Get a boolean mask of the local minimum peaks.
    fn peak_min(&self) -> BooleanChunked {
        invalid_operation_panic!(self)
    }

    /// Check if elements of this Series are in the right Series, or List values of the right Series.
    #[cfg(feature = "is_in")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_in")))]
    fn is_in(&self, _other: &Series) -> Result<BooleanChunked> {
        invalid_operation_panic!(self)
    }
    #[cfg(feature = "repeat_by")]
    #[cfg_attr(docsrs, doc(cfg(feature = "repeat_by")))]
    fn repeat_by(&self, _by: &UInt32Chunked) -> ListChunked {
        invalid_operation_panic!(self)
    }
    #[cfg(feature = "checked_arithmetic")]
    #[cfg_attr(docsrs, doc(cfg(feature = "checked_arithmetic")))]
    fn checked_div(&self, _rhs: &Series) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    #[cfg(feature = "is_first")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_first")))]
    /// Get a mask of the first unique values.
    fn is_first(&self) -> Result<BooleanChunked> {
        invalid_operation_panic!(self)
    }

    #[cfg(feature = "mode")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mode")))]
    /// Compute the most occurring element in the array.
    fn mode(&self) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    #[cfg(feature = "rolling_window")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    fn rolling_apply(&self, _window_size: usize, _f: &dyn Fn(&Series) -> Series) -> Result<Series> {
        panic!("rolling apply not implemented for this dtype. Only implemented for numeric data.")
    }
    #[cfg(feature = "concat_str")]
    #[cfg_attr(docsrs, doc(cfg(feature = "concat_str")))]
    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    fn str_concat(&self, _delimiter: &str) -> Utf8Chunked {
        invalid_operation_panic!(self)
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
            Err(PolarsError::SchemaMisMatch(
                "cannot unpack Series; data types don't match".into(),
            ))
        }
    }
}
