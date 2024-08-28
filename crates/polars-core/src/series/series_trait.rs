use std::any::Any;
use std::borrow::Cow;
use std::sync::RwLockReadGuard;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::metadata::MetadataTrait;
#[cfg(feature = "object")]
use crate::chunked_array::object::PolarsObjectSafe;
use crate::prelude::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IsSorted {
    Ascending,
    Descending,
    Not,
}

impl IsSorted {
    pub fn reverse(self) -> Self {
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

pub enum BitRepr {
    Small(UInt32Chunked),
    Large(UInt64Chunked),
}

pub(crate) mod private {
    use super::*;
    use crate::chunked_array::metadata::MetadataFlags;
    use crate::chunked_array::ops::compare_inner::{TotalEqInner, TotalOrdInner};

    pub trait PrivateSeriesNumeric {
        /// Return a bit representation
        ///
        /// If there is no available bit representation this returns `None`.
        fn bit_repr(&self) -> Option<BitRepr>;
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

        fn _get_flags(&self) -> MetadataFlags;

        fn _set_flags(&mut self, flags: MetadataFlags);

        unsafe fn equal_element(
            &self,
            _idx_self: usize,
            _idx_other: usize,
            _other: &Series,
        ) -> bool {
            invalid_operation_panic!(equal_element, self)
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
            invalid_operation_panic!(into_total_eq_inner, self)
        }
        #[allow(clippy::wrong_self_convention)]
        fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
            invalid_operation_panic!(into_total_ord_inner, self)
        }
        fn vec_hash(&self, _build_hasher: PlRandomState, _buf: &mut Vec<u64>) -> PolarsResult<()> {
            polars_bail!(opq = vec_hash, self._dtype());
        }
        fn vec_hash_combine(
            &self,
            _build_hasher: PlRandomState,
            _hashes: &mut [u64],
        ) -> PolarsResult<()> {
            polars_bail!(opq = vec_hash_combine, self._dtype());
        }
        #[cfg(feature = "algorithm_group_by")]
        unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        #[cfg(feature = "algorithm_group_by")]
        unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
        /// first cast to `Int64` to prevent overflow issues.
        #[cfg(feature = "algorithm_group_by")]
        unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        #[cfg(feature = "algorithm_group_by")]
        unsafe fn agg_std(&self, groups: &GroupsProxy, _ddof: u8) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        #[cfg(feature = "algorithm_group_by")]
        unsafe fn agg_var(&self, groups: &GroupsProxy, _ddof: u8) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }
        #[cfg(feature = "algorithm_group_by")]
        unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
            Series::full_null(self._field().name(), groups.len(), self._dtype())
        }

        fn subtract(&self, _rhs: &Series) -> PolarsResult<Series> {
            polars_bail!(opq = subtract, self._dtype());
        }
        fn add_to(&self, _rhs: &Series) -> PolarsResult<Series> {
            polars_bail!(opq = add, self._dtype());
        }
        fn multiply(&self, _rhs: &Series) -> PolarsResult<Series> {
            polars_bail!(opq = multiply, self._dtype());
        }
        fn divide(&self, _rhs: &Series) -> PolarsResult<Series> {
            polars_bail!(opq = divide, self._dtype());
        }
        fn remainder(&self, _rhs: &Series) -> PolarsResult<Series> {
            polars_bail!(opq = remainder, self._dtype());
        }
        #[cfg(feature = "algorithm_group_by")]
        fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> PolarsResult<GroupsProxy> {
            polars_bail!(opq = group_tuples, self._dtype());
        }
        #[cfg(feature = "zip_with")]
        fn zip_with_same_type(
            &self,
            _mask: &BooleanChunked,
            _other: &Series,
        ) -> PolarsResult<Series> {
            polars_bail!(opq = zip_with_same_type, self._dtype());
        }

        #[allow(unused_variables)]
        fn arg_sort_multiple(
            &self,
            by: &[Series],
            _options: &SortMultipleOptions,
        ) -> PolarsResult<IdxCa> {
            polars_bail!(opq = arg_sort_multiple, self._dtype());
        }
    }
}

pub trait SeriesTrait:
    Send + Sync + private::PrivateSeries + private::PrivateSeriesNumeric
{
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

    fn get_metadata(&self) -> Option<RwLockReadGuard<dyn MetadataTrait>> {
        None
    }

    /// Get the lengths of the underlying chunks
    fn chunk_lengths(&self) -> ChunkLenIter;

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

    /// Underlying chunks.
    ///
    /// # Safety
    /// The caller must ensure the length and the data types of `ArrayRef` does not change.
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef>;

    /// Number of chunks in this Series
    fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    /// Shrink the capacity of this array to fit its length.
    fn shrink_to_fit(&mut self) {
        // no-op
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

    /// Get a zero copy view of the data.
    ///
    /// When offset is negative the offset is counted from the
    /// end of the array
    fn split_at(&self, _offset: i64) -> (Series, Series);

    #[doc(hidden)]
    fn append(&mut self, _other: &Series) -> PolarsResult<()>;

    #[doc(hidden)]
    fn extend(&mut self, _other: &Series) -> PolarsResult<()>;

    /// Filter by boolean mask. This operation clones data.
    fn filter(&self, _filter: &BooleanChunked) -> PolarsResult<Series>;

    /// Take by index. This operation is clone.
    fn take(&self, _indices: &IdxCa) -> PolarsResult<Series>;

    /// Take by index.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> Series;

    /// Take by index. This operation is clone.
    fn take_slice(&self, _indices: &[IdxSize]) -> PolarsResult<Series>;

    /// Take by index.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_slice_unchecked(&self, _idx: &[IdxSize]) -> Series;

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

    /// Returns the std value in the array
    /// Returns an option because the array is nullable.
    fn std(&self, _ddof: u8) -> Option<f64> {
        None
    }

    /// Returns the var value in the array
    /// Returns an option because the array is nullable.
    fn var(&self, _ddof: u8) -> Option<f64> {
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

    fn cast(&self, _data_type: &DataType, options: CastOptions) -> PolarsResult<Series>;

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

    fn sort_with(&self, _options: SortOptions) -> PolarsResult<Series> {
        polars_bail!(opq = sort_with, self._dtype());
    }

    /// Retrieve the indexes needed for a sort.
    #[allow(unused)]
    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        invalid_operation_panic!(arg_sort, self)
    }

    /// Count the null values.
    fn null_count(&self) -> usize;

    /// Return if any the chunks in this `[ChunkedArray]` have nulls.
    fn has_nulls(&self) -> bool;

    /// Get unique values in the Series.
    fn unique(&self) -> PolarsResult<Series> {
        polars_bail!(opq = unique, self._dtype());
    }

    /// Get unique values in the Series.
    ///
    /// A `null` value also counts as a unique value.
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

    /// Get the sum of the Series as a new Scalar.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    fn sum_reduce(&self) -> PolarsResult<Scalar> {
        polars_bail!(opq = sum, self._dtype());
    }
    /// Get the max of the Series as a new Series of length 1.
    fn max_reduce(&self) -> PolarsResult<Scalar> {
        polars_bail!(opq = max, self._dtype());
    }
    /// Get the min of the Series as a new Series of length 1.
    fn min_reduce(&self) -> PolarsResult<Scalar> {
        polars_bail!(opq = min, self._dtype());
    }
    /// Get the median of the Series as a new Series of length 1.
    fn median_reduce(&self) -> PolarsResult<Scalar> {
        polars_bail!(opq = median, self._dtype());
    }
    /// Get the variance of the Series as a new Series of length 1.
    fn var_reduce(&self, _ddof: u8) -> PolarsResult<Scalar> {
        polars_bail!(opq = var, self._dtype());
    }
    /// Get the standard deviation of the Series as a new Series of length 1.
    fn std_reduce(&self, _ddof: u8) -> PolarsResult<Scalar> {
        polars_bail!(opq = std, self._dtype());
    }
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_reduce(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Scalar> {
        polars_bail!(opq = quantile, self._dtype());
    }

    /// Clone inner ChunkedArray and wrap in a new Arc
    fn clone_inner(&self) -> Arc<dyn SeriesTrait>;

    #[cfg(feature = "object")]
    /// Get the value at this index as a downcastable Any trait ref.
    fn get_object(&self, _index: usize) -> Option<&dyn PolarsObjectSafe> {
        invalid_operation_panic!(get_object, self)
    }

    #[cfg(feature = "object")]
    /// Get the value at this index as a downcastable Any trait ref.
    ///
    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn get_object_chunked_unchecked(
        &self,
        _chunk: usize,
        _index: usize,
    ) -> Option<&dyn PolarsObjectSafe> {
        invalid_operation_panic!(get_object_chunked_unchecked, self)
    }

    /// Get a hold to self as `Any` trait reference.
    fn as_any(&self) -> &dyn Any;

    /// Get a hold to self as `Any` trait reference.
    /// Only implemented for ObjectType
    fn as_any_mut(&mut self) -> &mut dyn Any {
        invalid_operation_panic!(as_any_mut, self)
    }

    #[cfg(feature = "checked_arithmetic")]
    fn checked_div(&self, _rhs: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = checked_div, self._dtype());
    }

    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    fn rolling_map(
        &self,
        _f: &dyn Fn(&Series) -> Series,
        _options: RollingOptionsFixedWindow,
    ) -> PolarsResult<Series> {
        polars_bail!(opq = rolling_map, self._dtype());
    }
}

impl<'a> (dyn SeriesTrait + 'a) {
    pub fn unpack<N>(&self) -> PolarsResult<&ChunkedArray<N>>
    where
        N: 'static + PolarsDataType,
    {
        polars_ensure!(&N::get_dtype() == self.dtype(), unpack);
        Ok(self.as_ref())
    }
}
