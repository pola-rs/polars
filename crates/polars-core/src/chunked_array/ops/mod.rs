//! Traits for miscellaneous operations on ChunkedArray
use arrow::offset::OffsetsBuffer;

use crate::prelude::*;

pub(crate) mod aggregate;
pub(crate) mod any_value;
pub(crate) mod append;
mod apply;
pub mod arity;
mod bit_repr;
pub(crate) mod chunkops;
pub(crate) mod compare_inner;
#[cfg(feature = "dtype-decimal")]
mod decimal;
pub(crate) mod downcast;
pub(crate) mod explode;
mod explode_and_offsets;
mod extend;
pub mod fill_null;
mod filter;
pub mod float_sorted_arg_max;
mod for_each;
pub mod full;
pub mod gather;
#[cfg(feature = "zip_with")]
pub(crate) mod min_max_binary;
pub(crate) mod nulls;
mod reverse;
#[cfg(feature = "rolling_window")]
pub(crate) mod rolling_window;
pub mod search_sorted;
mod set;
mod shift;
pub mod sort;
#[cfg(feature = "algorithm_group_by")]
pub(crate) mod unique;
#[cfg(feature = "zip_with")]
pub mod zip;

use polars_utils::no_call_const;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
pub use sort::options::*;

use crate::chunked_array::cast::CastOptions;
use crate::series::{BitRepr, IsSorted};
#[cfg(feature = "reinterpret")]
pub trait Reinterpret {
    fn reinterpret_signed(&self) -> Series {
        unimplemented!()
    }

    fn reinterpret_unsigned(&self) -> Series {
        unimplemented!()
    }
}

/// Transmute [`ChunkedArray`] to bit representation.
/// This is useful in hashing context and reduces no.
/// of compiled code paths.
pub(crate) trait ToBitRepr {
    fn to_bit_repr(&self) -> BitRepr;
}

pub trait ChunkAnyValue {
    /// Get a single value. Beware this is slow.
    /// If you need to use this slightly performant, cast Categorical to UInt32
    ///
    /// # Safety
    /// Does not do any bounds checking.
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue;

    /// Get a single value. Beware this is slow.
    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue>;
}

/// Explode/flatten a List or String Series
pub trait ChunkExplode {
    fn explode(&self) -> PolarsResult<Series> {
        self.explode_and_offsets().map(|t| t.0)
    }
    fn offsets(&self) -> PolarsResult<OffsetsBuffer<i64>>;
    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)>;
}

pub trait ChunkBytes {
    fn to_byte_slices(&self) -> Vec<&[u8]>;
}

/// This differs from ChunkWindowCustom and ChunkWindow
/// by not using a fold aggregator, but reusing a `Series` wrapper and calling `Series` aggregators.
/// This likely is a bit slower than ChunkWindow
#[cfg(feature = "rolling_window")]
pub trait ChunkRollApply: AsRefDataType {
    fn rolling_map(
        &self,
        _f: &dyn Fn(&Series) -> Series,
        _options: RollingOptionsFixedWindow,
    ) -> PolarsResult<Series>
    where
        Self: Sized,
    {
        polars_bail!(opq = rolling_map, self.as_ref_dtype());
    }
}

pub trait ChunkTake<Idx: ?Sized>: ChunkTakeUnchecked<Idx> {
    /// Gather values from ChunkedArray by index.
    fn take(&self, indices: &Idx) -> PolarsResult<Self>
    where
        Self: Sized;
}

pub trait ChunkTakeUnchecked<Idx: ?Sized> {
    /// Gather values from ChunkedArray by index.
    ///
    /// # Safety
    /// The non-null indices must be valid.
    unsafe fn take_unchecked(&self, indices: &Idx) -> Self;
}

/// Create a `ChunkedArray` with new values by index or by boolean mask.
///
/// Note that these operations clone data. This is however the only way we can modify at mask or
/// index level as the underlying Arrow arrays are immutable.
pub trait ChunkSet<'a, A, B> {
    /// Set the values at indexes `idx` to some optional value `Option<T>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = UInt32Chunked::new("a", &[1, 2, 3]);
    /// let new = ca.scatter_single(vec![0, 1], Some(10)).unwrap();
    ///
    /// assert_eq!(Vec::from(&new), &[Some(10), Some(10), Some(3)]);
    /// ```
    fn scatter_single<I: IntoIterator<Item = IdxSize>>(
        &'a self,
        idx: I,
        opt_value: Option<A>,
    ) -> PolarsResult<Self>
    where
        Self: Sized;

    /// Set the values at indexes `idx` by applying a closure to these values.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = Int32Chunked::new("a", &[1, 2, 3]);
    /// let new = ca.scatter_with(vec![0, 1], |opt_v| opt_v.map(|v| v - 5)).unwrap();
    ///
    /// assert_eq!(Vec::from(&new), &[Some(-4), Some(-3), Some(3)]);
    /// ```
    fn scatter_with<I: IntoIterator<Item = IdxSize>, F>(
        &'a self,
        idx: I,
        f: F,
    ) -> PolarsResult<Self>
    where
        Self: Sized,
        F: Fn(Option<A>) -> Option<B>;
    /// Set the values where the mask evaluates to `true` to some optional value `Option<T>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = Int32Chunked::new("a", &[1, 2, 3]);
    /// let mask = BooleanChunked::new("mask", &[false, true, false]);
    /// let new = ca.set(&mask, Some(5)).unwrap();
    /// assert_eq!(Vec::from(&new), &[Some(1), Some(5), Some(3)]);
    /// ```
    fn set(&'a self, mask: &BooleanChunked, opt_value: Option<A>) -> PolarsResult<Self>
    where
        Self: Sized;
}

/// Cast `ChunkedArray<T>` to `ChunkedArray<N>`
pub trait ChunkCast {
    /// Cast a [`ChunkedArray`] to [`DataType`]
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(data_type, CastOptions::NonStrict)
    }

    /// Cast a [`ChunkedArray`] to [`DataType`]
    fn cast_with_options(&self, data_type: &DataType, options: CastOptions)
        -> PolarsResult<Series>;

    /// Does not check if the cast is a valid one and may over/underflow
    ///
    /// # Safety
    /// - This doesn't do utf8 validation checking when casting from binary
    /// - This doesn't do categorical bound checking when casting from UInt32
    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series>;
}

/// Fastest way to do elementwise operations on a [`ChunkedArray<T>`] when the operation is cheaper than
/// branching due to null checking.
pub trait ChunkApply<'a, T> {
    type FuncRet;

    /// Apply a closure elementwise. This is fastest when the null check branching is more expensive
    /// than the closure application. Often it is.
    ///
    /// Null values remain null.
    ///
    /// # Example
    ///
    /// ```
    /// use polars_core::prelude::*;
    /// fn double(ca: &UInt32Chunked) -> UInt32Chunked {
    ///     ca.apply_values(|v| v * 2)
    /// }
    /// ```
    #[must_use]
    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(T) -> Self::FuncRet + Copy;

    /// Apply a closure elementwise including null values.
    #[must_use]
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T>) -> Option<Self::FuncRet> + Copy;

    /// Apply a closure elementwise and write results to a mutable slice.
    fn apply_to_slice<F, S>(&'a self, f: F, slice: &mut [S])
    // (value of chunkedarray, value of slice) -> value of slice
    where
        F: Fn(Option<T>, &S) -> S;
}

/// Aggregation operations.
pub trait ChunkAgg<T> {
    /// Aggregate the sum of the ChunkedArray.
    /// Returns `None` if not implemented for `T`.
    /// If the array is empty, `0` is returned
    fn sum(&self) -> Option<T> {
        None
    }

    fn min(&self) -> Option<T> {
        None
    }

    /// Returns the maximum value in the array, according to the natural order.
    /// Returns `None` if the array is empty or only contains null values.
    fn max(&self) -> Option<T> {
        None
    }

    fn min_max(&self) -> Option<(T, T)> {
        Some((self.min()?, self.max()?))
    }

    /// Returns the mean value in the array.
    /// Returns `None` if the array is empty or only contains null values.
    fn mean(&self) -> Option<f64> {
        None
    }
}

/// Quantile and median aggregation.
pub trait ChunkQuantile<T> {
    /// Returns the mean value in the array.
    /// Returns `None` if the array is empty or only contains null values.
    fn median(&self) -> Option<T> {
        None
    }
    /// Aggregate a given quantile of the ChunkedArray.
    /// Returns `None` if the array is empty or only contains null values.
    fn quantile(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<T>> {
        Ok(None)
    }
}

/// Variance and standard deviation aggregation.
pub trait ChunkVar {
    /// Compute the variance of this ChunkedArray/Series.
    fn var(&self, _ddof: u8) -> Option<f64> {
        None
    }

    /// Compute the standard deviation of this ChunkedArray/Series.
    fn std(&self, _ddof: u8) -> Option<f64> {
        None
    }
}

/// Compare [`Series`] and [`ChunkedArray`]'s and get a `boolean` mask that
/// can be used to filter rows.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// fn filter_all_ones(df: &DataFrame) -> PolarsResult<DataFrame> {
///     let mask = df
///     .column("column_a")?
///     .equal(1)?;
///
///     df.filter(&mask)
/// }
/// ```
pub trait ChunkCompare<Rhs> {
    type Item;

    /// Check for equality.
    fn equal(&self, rhs: Rhs) -> Self::Item;

    /// Check for equality where `None == None`.
    fn equal_missing(&self, rhs: Rhs) -> Self::Item;

    /// Check for inequality.
    fn not_equal(&self, rhs: Rhs) -> Self::Item;

    /// Check for inequality where `None == None`.
    fn not_equal_missing(&self, rhs: Rhs) -> Self::Item;

    /// Greater than comparison.
    #[allow(unused_variables)]
    fn gt(&self, rhs: Rhs) -> Self::Item {
        no_call_const!()
    }

    /// Greater than or equal comparison.
    #[allow(unused_variables)]
    fn gt_eq(&self, rhs: Rhs) -> Self::Item {
        no_call_const!()
    }

    /// Less than comparison.
    #[allow(unused_variables)]
    fn lt(&self, rhs: Rhs) -> Self::Item {
        no_call_const!()
    }

    /// Less than or equal comparison
    #[allow(unused_variables)]
    fn lt_eq(&self, rhs: Rhs) -> Self::Item {
        no_call_const!()
    }
}

/// Get unique values in a `ChunkedArray`
pub trait ChunkUnique {
    // We don't return Self to be able to use AutoRef specialization
    /// Get unique values of a ChunkedArray
    fn unique(&self) -> PolarsResult<Self>
    where
        Self: Sized;

    /// Get first index of the unique values in a `ChunkedArray`.
    /// This Vec is sorted.
    fn arg_unique(&self) -> PolarsResult<IdxCa>;

    /// Number of unique values in the `ChunkedArray`
    fn n_unique(&self) -> PolarsResult<usize> {
        self.arg_unique().map(|v| v.len())
    }
}

/// Sort operations on `ChunkedArray`.
pub trait ChunkSort<T: PolarsDataType> {
    #[allow(unused_variables)]
    fn sort_with(&self, options: SortOptions) -> ChunkedArray<T>;

    /// Returned a sorted `ChunkedArray`.
    fn sort(&self, descending: bool) -> ChunkedArray<T>;

    /// Retrieve the indexes needed to sort this array.
    fn arg_sort(&self, options: SortOptions) -> IdxCa;

    /// Retrieve the indexes need to sort this and the other arrays.
    #[allow(unused_variables)]
    fn arg_sort_multiple(
        &self,
        by: &[Series],
        _options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        polars_bail!(opq = arg_sort_multiple, T::get_dtype());
    }
}

pub type FillNullLimit = Option<IdxSize>;

#[derive(Copy, Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub enum FillNullStrategy {
    /// previous value in array
    Backward(FillNullLimit),
    /// next value in array
    Forward(FillNullLimit),
    /// mean value of array
    Mean,
    /// minimal value in array
    Min,
    /// maximum value in array
    Max,
    /// replace with the value zero
    Zero,
    /// replace with the value one
    One,
    /// replace with the maximum value of that data type
    MaxBound,
    /// replace with the minimal value of that data type
    MinBound,
}
/// Replace None values with a value
pub trait ChunkFillNullValue<T> {
    /// Replace None values with a give value `T`.
    fn fill_null_with_values(&self, value: T) -> PolarsResult<Self>
    where
        Self: Sized;
}

/// Fill a ChunkedArray with one value.
pub trait ChunkFull<T> {
    /// Create a ChunkedArray with a single value.
    fn full(name: &str, value: T, length: usize) -> Self
    where
        Self: Sized;
}

pub trait ChunkFullNull {
    fn full_null(_name: &str, _length: usize) -> Self
    where
        Self: Sized;
}

/// Reverse a [`ChunkedArray<T>`]
pub trait ChunkReverse {
    /// Return a reversed version of this array.
    fn reverse(&self) -> Self;
}

/// Filter values by a boolean mask.
pub trait ChunkFilter<T: PolarsDataType> {
    /// Filter values in the ChunkedArray with a boolean mask.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let array = Int32Chunked::new("array", &[1, 2, 3]);
    /// let mask = BooleanChunked::new("mask", &[true, false, true]);
    ///
    /// let filtered = array.filter(&mask).unwrap();
    /// assert_eq!(Vec::from(&filtered), [Some(1), Some(3)])
    /// ```
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<T>>
    where
        Self: Sized;
}

/// Create a new ChunkedArray filled with values at that index.
pub trait ChunkExpandAtIndex<T: PolarsDataType> {
    /// Create a new ChunkedArray filled with values at that index.
    fn new_from_index(&self, index: usize, length: usize) -> ChunkedArray<T>;
}

macro_rules! impl_chunk_expand {
    ($self:ident, $length:ident, $index:ident) => {{
        if $self.is_empty() {
            return $self.clone();
        }
        let opt_val = $self.get($index);
        match opt_val {
            Some(val) => ChunkedArray::full($self.name(), val, $length),
            None => ChunkedArray::full_null($self.name(), $length),
        }
    }};
}

impl<T: PolarsNumericType> ChunkExpandAtIndex<T> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkFull<T::Native>,
{
    fn new_from_index(&self, index: usize, length: usize) -> ChunkedArray<T> {
        let mut out = impl_chunk_expand!(self, length, index);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkExpandAtIndex<BooleanType> for BooleanChunked {
    fn new_from_index(&self, index: usize, length: usize) -> BooleanChunked {
        let mut out = impl_chunk_expand!(self, length, index);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkExpandAtIndex<StringType> for StringChunked {
    fn new_from_index(&self, index: usize, length: usize) -> StringChunked {
        let mut out = impl_chunk_expand!(self, length, index);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkExpandAtIndex<BinaryType> for BinaryChunked {
    fn new_from_index(&self, index: usize, length: usize) -> BinaryChunked {
        let mut out = impl_chunk_expand!(self, length, index);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkExpandAtIndex<BinaryOffsetType> for BinaryOffsetChunked {
    fn new_from_index(&self, index: usize, length: usize) -> BinaryOffsetChunked {
        let mut out = impl_chunk_expand!(self, length, index);
        out.set_sorted_flag(IsSorted::Ascending);
        out
    }
}

impl ChunkExpandAtIndex<ListType> for ListChunked {
    fn new_from_index(&self, index: usize, length: usize) -> ListChunked {
        let opt_val = self.get_as_series(index);
        match opt_val {
            Some(val) => {
                let mut ca = ListChunked::full(self.name(), &val, length);
                unsafe { ca.to_logical(self.inner_dtype().clone()) };
                ca
            },
            None => ListChunked::full_null_with_dtype(self.name(), length, self.inner_dtype()),
        }
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkExpandAtIndex<StructType> for StructChunked {
    fn new_from_index(&self, index: usize, length: usize) -> ChunkedArray<StructType> {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let chunk = self.downcast_chunks().get(chunk_idx).unwrap();
        let chunk = if chunk.is_null(idx) {
            new_null_array(chunk.data_type().clone(), length)
        } else {
            let values = chunk
                .values()
                .iter()
                .map(|arr| {
                    let s = Series::try_from(("", arr.clone())).unwrap();
                    let s = s.new_from_index(idx, length);
                    s.chunks()[0].clone()
                })
                .collect::<Vec<_>>();

            StructArray::new(chunk.data_type().clone(), values, None).boxed()
        };

        // SAFETY: chunks are from self.
        unsafe { self.copy_with_chunks(vec![chunk]) }
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkExpandAtIndex<FixedSizeListType> for ArrayChunked {
    fn new_from_index(&self, index: usize, length: usize) -> ArrayChunked {
        let opt_val = self.get_as_series(index);
        match opt_val {
            Some(val) => {
                let mut ca = ArrayChunked::full(self.name(), &val, length);
                unsafe { ca.to_logical(self.inner_dtype().clone()) };
                ca
            },
            None => ArrayChunked::full_null_with_dtype(
                self.name(),
                length,
                self.inner_dtype(),
                self.width(),
            ),
        }
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkExpandAtIndex<ObjectType<T>> for ObjectChunked<T> {
    fn new_from_index(&self, index: usize, length: usize) -> ObjectChunked<T> {
        let opt_val = self.get(index);
        match opt_val {
            Some(val) => ObjectChunked::<T>::full(self.name(), val.clone(), length),
            None => ObjectChunked::<T>::full_null(self.name(), length),
        }
    }
}

/// Shift the values of a [`ChunkedArray`] by a number of periods.
pub trait ChunkShiftFill<T: PolarsDataType, V> {
    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `fill_value`.
    fn shift_and_fill(&self, periods: i64, fill_value: V) -> ChunkedArray<T>;
}

pub trait ChunkShift<T: PolarsDataType> {
    fn shift(&self, periods: i64) -> ChunkedArray<T>;
}

/// Combine two [`ChunkedArray`] based on some predicate.
pub trait ChunkZip<T: PolarsDataType> {
    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &ChunkedArray<T>,
    ) -> PolarsResult<ChunkedArray<T>>;
}

/// Apply kernels on the arrow array chunks in a ChunkedArray.
pub trait ChunkApplyKernel<A: Array> {
    /// Apply kernel and return result as a new ChunkedArray.
    #[must_use]
    fn apply_kernel(&self, f: &dyn Fn(&A) -> ArrayRef) -> Self;

    /// Apply a kernel that outputs an array of different type.
    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&A) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType;
}

#[cfg(feature = "is_first_distinct")]
/// Mask the first unique values as `true`
pub trait IsFirstDistinct<T: PolarsDataType> {
    fn is_first_distinct(&self) -> PolarsResult<BooleanChunked> {
        polars_bail!(opq = is_first_distinct, T::get_dtype());
    }
}

#[cfg(feature = "is_last_distinct")]
/// Mask the last unique values as `true`
pub trait IsLastDistinct<T: PolarsDataType> {
    fn is_last_distinct(&self) -> PolarsResult<BooleanChunked> {
        polars_bail!(opq = is_last_distinct, T::get_dtype());
    }
}
