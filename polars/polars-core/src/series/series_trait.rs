#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::chunked_array::ChunkIdIter;
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::array::ArrayRef;
use polars_arrow::prelude::QuantileInterpolOptions;
use std::any::Any;
use std::borrow::Cow;
#[cfg(feature = "temporal")]
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
pub enum IsSorted {
    Ascending,
    Descending,
    Not,
}

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
    #[cfg(feature = "rows")]
    use crate::frame::groupby::GroupsProxy;

    use crate::chunked_array::ops::compare_inner::{PartialEqInner, PartialOrdInner};
    use ahash::RandomState;

    pub trait PrivateSeriesNumeric {
        fn bit_repr_is_large(&self) -> bool {
            false
        }
        fn bit_repr_large(&self) -> UInt64Chunked {
            unimplemented!()
        }
        fn bit_repr_small(&self) -> UInt32Chunked {
            unimplemented!()
        }
    }

    pub trait PrivateSeries {
        #[cfg(feature = "object")]
        fn get_list_builder(
            &self,
            _name: &str,
            _values_capacity: usize,
            _list_capacity: usize,
        ) -> Box<dyn ListBuilderTrait> {
            invalid_operation_panic!(self)
        }

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
        /// Apply a rolling sum to a Series.
        #[cfg(feature = "rolling_window")]
        fn _rolling_sum(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }
        /// Apply a rolling median to a Series.
        #[cfg(feature = "rolling_window")]
        fn _rolling_median(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }
        /// Apply a rolling quantile to a Series.
        #[cfg(feature = "rolling_window")]
        fn _rolling_quantile(
            &self,
            _quantile: f64,
            _interpolation: QuantileInterpolOptions,
            _options: RollingOptions,
        ) -> Result<Series> {
            invalid_operation!(self)
        }

        /// Apply a rolling min to a Series.
        #[cfg(feature = "rolling_window")]
        fn _rolling_min(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }
        /// Apply a rolling max to a Series.
        #[cfg(feature = "rolling_window")]
        fn _rolling_max(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }

        /// Apply a rolling variance to a Series.
        #[cfg(feature = "rolling_window")]
        fn _rolling_var(&self, _options: RollingOptions) -> Result<Series> {
            invalid_operation!(self)
        }

        /// Apply a rolling std_dev to a Series.
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
        fn vec_hash(&self, _build_hasher: RandomState) -> Vec<u64> {
            invalid_operation_panic!(self)
        }
        fn vec_hash_combine(&self, _build_hasher: RandomState, _hashes: &mut [u64]) {
            invalid_operation_panic!(self)
        }
        fn agg_mean(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn agg_min(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn agg_max(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
        /// first cast to `Int64` to prevent overflow issues.
        fn agg_sum(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn agg_std(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn agg_var(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn agg_list(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn agg_quantile(
            &self,
            _groups: &GroupsProxy,
            _quantile: f64,
            _interpol: QuantileInterpolOptions,
        ) -> Option<Series> {
            None
        }
        fn agg_median(&self, _groups: &GroupsProxy) -> Option<Series> {
            None
        }
        fn zip_outer_join_column(
            &self,
            _right_column: &Series,
            _opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
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
        fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> GroupsProxy {
            invalid_operation_panic!(self)
        }
        fn zip_with_same_type(&self, _mask: &BooleanChunked, _other: &Series) -> Result<Series> {
            invalid_operation_panic!(self)
        }
        #[cfg(feature = "sort_multiple")]
        fn argsort_multiple(&self, _by: &[Series], _reverse: &[bool]) -> Result<IdxCa> {
            Err(PolarsError::InvalidOperation(
                "argsort_multiple is not implemented for this Series".into(),
            ))
        }
    }
}

pub trait SeriesTrait:
    Send + Sync + private::PrivateSeries + private::PrivateSeriesNumeric
{
    /// Check if [`Series`] is sorted.
    fn is_sorted(&self) -> IsSorted {
        IsSorted::Not
    }

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
    #[cfg(feature = "dtype-time")]
    fn time(&self) -> Result<&TimeChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != Time", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype Date
    #[cfg(feature = "dtype-date")]
    fn date(&self) -> Result<&DateChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!(" Series dtype {:?} != Date", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype datetime
    #[cfg(feature = "dtype-datetime")]
    fn datetime(&self) -> Result<&DatetimeChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != datetime", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype duration
    #[cfg(feature = "dtype-duration")]
    fn duration(&self) -> Result<&DurationChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != duration", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype list
    fn list(&self) -> Result<&ListChunked> {
        Err(PolarsError::SchemaMisMatch(
            format!("Series dtype {:?} != list", self.dtype()).into(),
        ))
    }

    /// Unpack to ChunkedArray of dtype categorical
    #[cfg(feature = "dtype-categorical")]
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

    #[doc(hidden)]
    fn append(&mut self, _other: &Series) -> Result<()> {
        invalid_operation_panic!(self)
    }

    #[doc(hidden)]
    fn extend(&mut self, _other: &Series) -> Result<()> {
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
    /// - This doesn't check any bounds.
    /// - Iterator must be TrustedLen
    unsafe fn take_iter_unchecked(&self, _iter: &mut dyn TakeIterator) -> Series {
        invalid_operation_panic!(self)
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> Result<Series> {
        invalid_operation_panic!(self)
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// - This doesn't check any bounds.
    /// - Iterator must be TrustedLen
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
    fn take(&self, _indices: &IdxCa) -> Result<Series> {
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

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    fn get(&self, _index: usize) -> AnyValue {
        invalid_operation_panic!(self)
    }

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    ///
    /// This may refer to physical types
    ///
    /// # Safety
    /// Does not do any bounds checking
    #[cfg(feature = "private")]
    unsafe fn get_unchecked(&self, _index: usize) -> AnyValue {
        invalid_operation_panic!(self)
    }

    fn sort_with(&self, _options: SortOptions) -> Series {
        invalid_operation_panic!(self)
    }

    /// Retrieve the indexes needed for a sort.
    #[allow(unused)]
    fn argsort(&self, options: SortOptions) -> IdxCa {
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
    fn arg_unique(&self) -> Result<IdxCa> {
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
    fn arg_true(&self) -> Result<IdxCa> {
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
    fn repeat_by(&self, _by: &IdxCa) -> ListChunked {
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
    fn rolling_apply(
        &self,
        _f: &dyn Fn(&Series) -> Series,
        _options: RollingOptions,
    ) -> Result<Series> {
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
