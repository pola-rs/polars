use std::any::Any;
use std::borrow::Cow;
#[cfg(feature = "temporal")]
use std::sync::Arc;

use polars_arrow::prelude::QuantileInterpolOptions;

#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IsSorted {
    Ascending,
    Descending,
    Not,
}

impl IsSorted {
    pub(crate) fn reverse(self) -> Self {
        use IsSorted::*;
        match self {
            Ascending => Descending,
            Descending => Ascending,
            Not => Not,
        }
    }
}

macro_rules! invalid_operation_panic {
    ($op:ident, $s:expr) => {
        panic!(
            "`{}` operation not supported for dtype `{}`",
            stringify!($op),
            $s._dtype()
        )
    };
}

pub(crate) mod private {
    use ahash::RandomState;

    use super::*;
    use crate::chunked_array::ops::compare_inner::{PartialEqInner, PartialOrdInner};
    #[cfg(feature = "rows")]
    use crate::frame::groupby::GroupsProxy;

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
            invalid_operation_panic!(get_list_builder, self)
        }

        /// Get field (used in schema)
        fn _field(&self) -> Cow<Field>;

        fn _dtype(&self) -> &DataType;

        fn compute_len(&mut self);

        fn explode_by_offsets(&self, _offsets: &[i64]) -> Series {
            invalid_operation_panic!(explode_by_offsets, self)
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

        fn _set_sorted_flag(&mut self, _is_sorted: IsSorted) {
            // ignore
        }

        unsafe fn equal_element(
            &self,
            _idx_self: usize,
            _idx_other: usize,
            _other: &Series,
        ) -> bool {
            invalid_operation_panic!(equal_element, self)
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
            invalid_operation_panic!(into_partial_eq_inner, self)
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
            invalid_operation_panic!(into_partial_ord_inner, self)
        }
        fn vec_hash(&self, _build_hasher: RandomState, _buf: &mut Vec<u64>) -> PolarsResult<()> {
            polars_bail!(opq = vec_hash, self._dtype());
        }
        fn vec_hash_combine(
            &self,
            _build_hasher: RandomState,
            _hashes: &mut [u64],
        ) -> PolarsResult<()> {
            polars_bail!(opq = vec_hash_combine, self._dtype());
        }
        unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
        /// first cast to `Int64` to prevent overflow issues.
        unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        unsafe fn agg_std(&self, groups: &GroupsProxy, _ddof: u8) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        unsafe fn agg_var(&self, groups: &GroupsProxy, _ddof: u8) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }

        fn zip_outer_join_column(
            &self,
            _right_column: &Series,
            _opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
        ) -> Series {
            invalid_operation_panic!(zip_outer_join_column, self)
        }

        fn subtract(&self, _rhs: &Series) -> PolarsResult<Series> {
            invalid_operation_panic!(sub, self)
        }
        fn add_to(&self, _rhs: &Series) -> PolarsResult<Series> {
            invalid_operation_panic!(add, self)
        }
        fn multiply(&self, _rhs: &Series) -> PolarsResult<Series> {
            invalid_operation_panic!(mul, self)
        }
        fn divide(&self, _rhs: &Series) -> PolarsResult<Series> {
            invalid_operation_panic!(div, self)
        }
        fn remainder(&self, _rhs: &Series) -> PolarsResult<Series> {
            invalid_operation_panic!(rem, self)
        }
        fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> PolarsResult<GroupsProxy> {
            invalid_operation_panic!(group_tuples, self)
        }
        #[cfg(feature = "zip_with")]
        fn zip_with_same_type(
            &self,
            _mask: &BooleanChunked,
            _other: &Series,
        ) -> PolarsResult<Series> {
            invalid_operation_panic!(zip_with_same_type, self)
        }

        fn arg_sort_multiple(&self, _options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
            polars_bail!(opq = arg_sort_multiple, self._dtype());
        }
    }
}

pub trait SeriesTrait:
    Send + Sync + private::PrivateSeries + private::PrivateSeriesNumeric
{
    /// Check if [`Series`] is sorted.
    fn is_sorted_flag(&self) -> IsSorted {
        IsSorted::Not
    }

    /// Rename the Series.
    fn rename(&mut self, name: &str);

    fn bitand(&self, _other: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = bitand, self._dtype());
    }

    fn bitor(&self, _other: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = bitor, self._dtype());
    }

    fn bitxor(&self, _other: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = bitxor, self._dtype());
    }

    /// Get the lengths of the underlying chunks
    fn chunk_lengths(&self) -> ChunkIdIter;

    /// Name of series.
    fn name(&self) -> &str;

    /// Get field (used in schema)
    fn field(&self) -> Cow<Field> {
        self._field()
    }

    /// Get datatype of series.
    fn dtype(&self) -> &DataType {
        self._dtype()
    }

    /// Underlying chunks.
    fn chunks(&self) -> &Vec<ArrayRef>;

    /// Number of chunks in this Series
    fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    /// Shrink the capacity of this array to fit its length.
    fn shrink_to_fit(&mut self) {
        invalid_operation_panic!(shrink_to_fit, self);
    }

    /// Take `num_elements` from the top as a zero copy view.
    fn limit(&self, num_elements: usize) -> Series {
        self.slice(0, num_elements)
    }

    /// Get a zero copy view of the data.
    ///
    /// When offset is negative the offset is counted from the
    /// end of the array
    fn slice(&self, _offset: i64, _length: usize) -> Series;

    #[doc(hidden)]
    fn append(&mut self, _other: &Series) -> PolarsResult<()>;

    #[doc(hidden)]
    fn extend(&mut self, _other: &Series) -> PolarsResult<()>;

    /// Filter by boolean mask. This operation clones data.
    fn filter(&self, _filter: &BooleanChunked) -> PolarsResult<Series>;

    #[doc(hidden)]
    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series;

    #[doc(hidden)]
    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series;

    /// Take by index from an iterator. This operation clones the data.
    fn take_iter(&self, _iter: &mut dyn TakeIterator) -> PolarsResult<Series>;

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// - This doesn't check any bounds.
    /// - Iterator must be TrustedLen
    unsafe fn take_iter_unchecked(&self, _iter: &mut dyn TakeIterator) -> Series;

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> PolarsResult<Series>;

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// - This doesn't check any bounds.
    /// - Iterator must be TrustedLen
    unsafe fn take_opt_iter_unchecked(&self, _iter: &mut dyn TakeIteratorNulls) -> Series;

    /// Take by index from an iterator. This operation clones the data.
    /// todo! remove?
    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, _iter: &mut dyn TakeIteratorNulls) -> PolarsResult<Series> {
        invalid_operation_panic!(take_opt_iter, self)
    }

    /// Take by index. This operation is clone.
    fn take(&self, _indices: &IdxCa) -> PolarsResult<Series>;

    /// Get length of series.
    fn len(&self) -> usize;

    /// Check if Series is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Aggregate all chunks to a contiguous array of memory.
    fn rechunk(&self) -> Series;

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
        None
    }

    /// Returns the median value in the array
    /// Returns an option because the array is nullable.
    fn median(&self) -> Option<f64> {
        None
    }

    /// Create a new Series filled with values from the given index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// let s = Series::new("a", [0i32, 1, 8]);
    /// let s2 = s.new_from_index(2, 4);
    /// assert_eq!(Vec::from(s2.i32().unwrap()), &[Some(8), Some(8), Some(8), Some(8)])
    /// ```
    fn new_from_index(&self, _index: usize, _length: usize) -> Series;

    fn cast(&self, _data_type: &DataType) -> PolarsResult<Series>;

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    fn get(&self, _index: usize) -> PolarsResult<AnyValue>;

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    ///
    /// This may refer to physical types
    ///
    /// # Safety
    /// Does not do any bounds checking
    unsafe fn get_unchecked(&self, _index: usize) -> AnyValue {
        invalid_operation_panic!(get_unchecked, self)
    }

    fn sort_with(&self, _options: SortOptions) -> Series {
        invalid_operation_panic!(sort_with, self)
    }

    /// Retrieve the indexes needed for a sort.
    #[allow(unused)]
    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        invalid_operation_panic!(arg_sort, self)
    }

    /// Count the null values.
    fn null_count(&self) -> usize;

    /// Return if any the chunks in this `[ChunkedArray]` have a validity bitmap.
    /// no bitmap means no null values.
    fn has_validity(&self) -> bool;

    /// Get unique values in the Series.
    fn unique(&self) -> PolarsResult<Series> {
        polars_bail!(opq = unique, self._dtype());
    }

    /// Get unique values in the Series.
    fn n_unique(&self) -> PolarsResult<usize> {
        polars_bail!(opq = n_unique, self._dtype());
    }

    /// Get first indexes of unique values.
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        polars_bail!(opq = arg_unique, self._dtype());
    }

    /// Get a mask of the null values.
    fn is_null(&self) -> BooleanChunked;

    /// Get a mask of the non-null values.
    fn is_not_null(&self) -> BooleanChunked;

    /// return a Series in reversed order
    fn reverse(&self) -> Series;

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        polars_bail!(opq = as_single_ptr, self._dtype());
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
    /// fn example() -> PolarsResult<()> {
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
    fn shift(&self, _periods: i64) -> Series;

    /// Get the sum of the Series as a new Series of length 1.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    fn _sum_as_series(&self) -> Series {
        Series::full_null(self.name(), 1, self.dtype())
    }
    /// Get the max of the Series as a new Series of length 1.
    fn max_as_series(&self) -> Series {
        Series::full_null(self.name(), 1, self.dtype())
    }
    /// Get the min of the Series as a new Series of length 1.
    fn min_as_series(&self) -> Series {
        Series::full_null(self.name(), 1, self.dtype())
    }
    /// Get the median of the Series as a new Series of length 1.
    fn median_as_series(&self) -> Series {
        Series::full_null(self.name(), 1, self.dtype())
    }
    /// Get the variance of the Series as a new Series of length 1.
    fn var_as_series(&self, _ddof: u8) -> Series {
        Series::full_null(self.name(), 1, self.dtype())
    }
    /// Get the standard deviation of the Series as a new Series of length 1.
    fn std_as_series(&self, _ddof: u8) -> Series {
        Series::full_null(self.name(), 1, self.dtype())
    }
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        Ok(Series::full_null(self.name(), 1, self.dtype()))
    }

    /// Clone inner ChunkedArray and wrap in a new Arc
    fn clone_inner(&self) -> Arc<dyn SeriesTrait>;

    #[cfg(feature = "object")]
    /// Get the value at this index as a downcastable Any trait ref.
    fn get_object(&self, _index: usize) -> Option<&dyn PolarsObjectSafe> {
        invalid_operation_panic!(get_object, self)
    }

    /// Get a hold to self as `Any` trait reference.
    /// Only implemented for ObjectType
    fn as_any(&self) -> &dyn Any {
        invalid_operation_panic!(as_any, self)
    }

    /// Get a hold to self as `Any` trait reference.
    /// Only implemented for ObjectType
    fn as_any_mut(&mut self) -> &mut dyn Any {
        invalid_operation_panic!(as_any_mut, self)
    }

    /// Get a boolean mask of the local maximum peaks.
    fn peak_max(&self) -> BooleanChunked {
        invalid_operation_panic!(peak_max, self)
    }

    /// Get a boolean mask of the local minimum peaks.
    fn peak_min(&self) -> BooleanChunked {
        invalid_operation_panic!(peak_min, self)
    }

    /// Check if elements of this Series are in the right Series, or List values of the right Series.
    #[cfg(feature = "is_in")]
    fn is_in(&self, _other: &Series) -> PolarsResult<BooleanChunked> {
        polars_bail!(opq = is_in, self._dtype());
    }
    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, _by: &IdxCa) -> PolarsResult<ListChunked> {
        polars_bail!(opq = repeat_by, self._dtype());
    }
    #[cfg(feature = "checked_arithmetic")]
    fn checked_div(&self, _rhs: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = checked_div, self._dtype());
    }

    #[cfg(feature = "mode")]
    /// Compute the most occurring element in the array.
    fn mode(&self) -> PolarsResult<Series> {
        polars_bail!(opq = mode, self._dtype());
    }

    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    fn rolling_apply(
        &self,
        _f: &dyn Fn(&Series) -> Series,
        _options: RollingOptionsFixedWindow,
    ) -> PolarsResult<Series> {
        polars_bail!(opq = rolling_apply, self._dtype());
    }
    #[cfg(feature = "concat_str")]
    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    fn str_concat(&self, _delimiter: &str) -> Utf8Chunked {
        invalid_operation_panic!(str_concat, self);
    }

    fn tile(&self, _n: usize) -> Series {
        invalid_operation_panic!(tile, self);
    }
}

impl<'a> (dyn SeriesTrait + 'a) {
    pub fn unpack<N: 'static>(&self) -> PolarsResult<&ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        polars_ensure!(&N::get_dtype() == self.dtype(), unpack);
        Ok(self.as_ref())
    }
}
