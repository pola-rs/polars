//! Traits for miscellaneous operations on ChunkedArray
use std::marker::Sized;

use arrow::array::ArrayRef;

pub use self::take::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectType;
use crate::prelude::*;
use arrow::buffer::Buffer;
#[cfg(feature = "dtype-categorical")]
use std::ops::Deref;

pub(crate) mod aggregate;
mod any_value;
mod append;
mod apply;
mod bit_repr;
pub(crate) mod chunkops;
pub(crate) mod compare_inner;
#[cfg(feature = "concat_str")]
mod concat_str;
#[cfg(feature = "cum_agg")]
mod cum_agg;
pub(crate) mod downcast;
pub(crate) mod explode;
mod fill_null;
mod filter;
pub mod full;
#[cfg(feature = "interpolate")]
mod interpolate;
#[cfg(feature = "is_in")]
mod is_in;
mod peaks;
#[cfg(feature = "repeat_by")]
mod repeat_by;
mod reverse;
pub(crate) mod rolling_window;
mod set;
mod shift;
pub(crate) mod sort;
pub(crate) mod take;
pub(crate) mod unique;
#[cfg(feature = "zip_with")]
pub mod zip;

#[cfg(feature = "to_list")]
pub trait ToList<T: PolarsDataType> {
    fn to_list(&self) -> Result<ListChunked> {
        Err(PolarsError::InvalidOperation(
            format!("to_list not supported for dtype: {:?}", T::get_dtype()).into(),
        ))
    }
}

#[cfg(feature = "interpolate")]
pub trait Interpolate {
    fn interpolate(&self) -> Self;
}

#[cfg(feature = "reinterpret")]
pub trait Reinterpret {
    fn reinterpret_signed(&self) -> Series {
        unimplemented!()
    }

    fn reinterpret_unsigned(&self) -> Series {
        unimplemented!()
    }
}

/// Transmute ChunkedArray to bit representation.
/// This is useful in hashing context and reduces no.
/// of compiled code paths.
pub(crate) trait ToBitRepr {
    fn bit_repr_is_large() -> bool;

    fn bit_repr_large(&self) -> UInt64Chunked;
    fn bit_repr_small(&self) -> UInt32Chunked;
}

pub trait ChunkAnyValue {
    /// Get a single value. Beware this is slow.
    /// If you need to use this slightly performant, cast Categorical to UInt32
    ///
    /// # Safety
    /// Does not do any bounds checking.
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue;

    /// Get a single value. Beware this is slow.
    fn get_any_value(&self, index: usize) -> AnyValue;
}

#[cfg(feature = "cum_agg")]
pub trait ChunkCumAgg<T> {
    /// Get an array with the cumulative max computed at every element
    fn cummax(&self, _reverse: bool) -> ChunkedArray<T> {
        panic!("operation cummax not supported for this dtype")
    }
    /// Get an array with the cumulative min computed at every element
    fn cummin(&self, _reverse: bool) -> ChunkedArray<T> {
        panic!("operation cummin not supported for this dtype")
    }
    /// Get an array with the cumulative sum computed at every element
    fn cumsum(&self, _reverse: bool) -> ChunkedArray<T> {
        panic!("operation cumsum not supported for this dtype")
    }
}

/// Traverse and collect every nth element
pub trait ChunkTakeEvery<T> {
    /// Traverse and collect every nth element in a new array.
    fn take_every(&self, n: usize) -> ChunkedArray<T>;
}

/// Explode/ flatten a
pub trait ChunkExplode {
    fn explode(&self) -> Result<Series> {
        self.explode_and_offsets().map(|t| t.0)
    }
    fn explode_and_offsets(&self) -> Result<(Series, Buffer<i64>)>;
}

pub trait ChunkBytes {
    fn to_byte_slices(&self) -> Vec<&[u8]>;
}

/// This differs from ChunkWindowCustom and ChunkWindow
/// by not using a fold aggregator, but reusing a `Series` wrapper and calling `Series` aggregators.
/// This likely is a bit slower than ChunkWindow
#[cfg(feature = "rolling_window")]
pub trait ChunkRollApply {
    fn rolling_apply(&self, _window_size: usize, _f: &dyn Fn(&Series) -> Series) -> Result<Self>
    where
        Self: Sized,
    {
        Err(PolarsError::InvalidOperation(
            "rolling mean not supported for this datatype".into(),
        ))
    }
}

/// Random access
pub trait TakeRandom {
    type Item;

    /// Get a nullable value by index.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Get a value by index and ignore the null bit.
    ///
    /// # Safety
    ///
    /// Does not do bound checks.
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.get(index)
    }
}
// Utility trait because associated type needs a lifetime
pub trait TakeRandomUtf8 {
    type Item;

    /// Get a nullable value by index.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`
    fn get(self, index: usize) -> Option<Self::Item>;

    /// Get a value by index and ignore the null bit.
    ///
    /// # Safety
    ///
    /// Does not do bound checks.
    unsafe fn get_unchecked(self, index: usize) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.get(index)
    }
}

/// Fast access by index.
pub trait ChunkTake {
    /// Take values from ChunkedArray by index.
    ///
    /// # Safety
    ///
    /// Doesn't do any bound checking.
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls;

    /// Take values from ChunkedArray by index.
    /// Note that the iterator will be cloned, so prefer an iterator that takes the owned memory
    /// by reference.
    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Result<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls;
}

/// Create a `ChunkedArray` with new values by index or by boolean mask.
/// Note that these operations clone data. This is however the only way we can modify at mask or
/// index level as the underlying Arrow arrays are immutable.
pub trait ChunkSet<'a, A, B> {
    /// Set the values at indexes `idx` to some optional value `Option<T>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);
    /// let new = ca.set_at_idx(vec![0, 1], Some(10)).unwrap();
    ///
    /// assert_eq!(Vec::from(&new), &[Some(10), Some(10), Some(3)]);
    /// ```
    fn set_at_idx<I: IntoIterator<Item = usize>>(
        &'a self,
        idx: I,
        opt_value: Option<A>,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Set the values at indexes `idx` by applying a closure to these values.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
    /// let new = ca.set_at_idx_with(vec![0, 1], |opt_v| opt_v.map(|v| v - 5)).unwrap();
    ///
    /// assert_eq!(Vec::from(&new), &[Some(-4), Some(-3), Some(3)]);
    /// ```
    fn set_at_idx_with<I: IntoIterator<Item = usize>, F>(&'a self, idx: I, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<A>) -> Option<B>;
    /// Set the values where the mask evaluates to `true` to some optional value `Option<T>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
    /// let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
    /// let new = ca.set(&mask, Some(5)).unwrap();
    /// assert_eq!(Vec::from(&new), &[Some(1), Some(5), Some(3)]);
    /// ```
    fn set(&'a self, mask: &BooleanChunked, opt_value: Option<A>) -> Result<Self>
    where
        Self: Sized;

    /// Set the values where the mask evaluates to `true` by applying a closure to these values.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let ca = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);
    /// let mask = BooleanChunked::new_from_slice("mask", &[false, true, false]);
    /// let new = ca.set_with(&mask, |opt_v| opt_v.map(
    ///     |v| v * 2
    /// )).unwrap();
    /// assert_eq!(Vec::from(&new), &[Some(1), Some(4), Some(3)]);
    /// ```
    fn set_with<F>(&'a self, mask: &BooleanChunked, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<A>) -> Option<B>;
}

/// Cast `ChunkedArray<T>` to `ChunkedArray<N>`
pub trait ChunkCast {
    /// Cast a `[ChunkedArray]` to `[DataType]`
    fn cast(&self, data_type: &DataType) -> Result<Series>;
}

/// Fastest way to do elementwise operations on a ChunkedArray<T> when the operation is cheaper than
/// branching due to null checking
pub trait ChunkApply<'a, A, B> {
    /// Apply a closure elementwise and cast to a Numeric ChunkedArray. This is fastest when the null check branching is more expensive
    /// than the closure application.
    ///
    /// Null values remain null.
    fn apply_cast_numeric<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(A) -> S::Native + Copy,
        S: PolarsNumericType;

    /// Apply a closure on optional values and cast to Numeric ChunkedArray without null values.
    fn branch_apply_cast_numeric_no_null<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<A>) -> S::Native + Copy,
        S: PolarsNumericType;

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
    ///     ca.apply(|v| v * 2)
    /// }
    /// ```
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(A) -> B + Copy;

    fn try_apply<F>(&'a self, f: F) -> Result<Self>
    where
        F: Fn(A) -> Result<B> + Copy,
        Self: Sized;

    /// Apply a closure elementwise including null values.
    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<A>) -> Option<B> + Copy;

    /// Apply a closure elementwise. The closure gets the index of the element as first argument.
    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, A)) -> B + Copy;

    /// Apply a closure elementwise. The closure gets the index of the element as first argument.
    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<A>)) -> Option<B> + Copy;

    /// Apply a closure elementwise and write results to a mutable slice.
    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    // (value of chunkedarray, value of slice) -> value of slice
    where
        F: Fn(Option<A>, &T) -> T;
}

/// Aggregation operations
pub trait ChunkAgg<T> {
    /// Aggregate the sum of the ChunkedArray.
    /// Returns `None` if the array is empty or only contains null values.
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

    /// Returns the mean value in the array.
    /// Returns `None` if the array is empty or only contains null values.
    fn mean(&self) -> Option<f64> {
        None
    }

    /// Returns the mean value in the array.
    /// Returns `None` if the array is empty or only contains null values.
    fn median(&self) -> Option<f64> {
        None
    }

    /// Aggregate a given quantile of the ChunkedArray.
    /// Returns `None` if the array is empty or only contains null values.
    fn quantile(&self, _quantile: f64) -> Result<Option<T>> {
        Ok(None)
    }
}

/// Variance and standard deviation aggregation.
pub trait ChunkVar<T> {
    /// Compute the variance of this ChunkedArray/Series.
    fn var(&self) -> Option<T> {
        None
    }

    /// Compute the standard deviation of this ChunkedArray/Series.
    fn std(&self) -> Option<T> {
        None
    }
}

/// Compare [Series](series/series/enum.Series.html)
/// and [ChunkedArray](series/chunked_array/struct.ChunkedArray.html)'s and get a `boolean` mask that
/// can be used to filter rows.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// fn filter_all_ones(df: &DataFrame) -> Result<DataFrame> {
///     let mask = df
///     .column("column_a")?
///     .eq(1);
///
///     df.filter(&mask)
/// }
/// ```
pub trait ChunkCompare<Rhs> {
    /// Check for equality and regard missing values as equal.
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked;

    /// Check for equality.
    fn eq(&self, rhs: Rhs) -> BooleanChunked;

    /// Check for inequality.
    fn neq(&self, rhs: Rhs) -> BooleanChunked;

    /// Greater than comparison.
    fn gt(&self, rhs: Rhs) -> BooleanChunked;

    /// Greater than or equal comparison.
    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked;

    /// Less than comparison.
    fn lt(&self, rhs: Rhs) -> BooleanChunked;

    /// Less than or equal comparison
    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked;
}

/// Get unique values in a `ChunkedArray`
pub trait ChunkUnique<T> {
    // We don't return Self to be able to use AutoRef specialization
    /// Get unique values of a ChunkedArray
    fn unique(&self) -> Result<ChunkedArray<T>>;

    /// Get first index of the unique values in a `ChunkedArray`.
    /// This Vec is sorted.
    fn arg_unique(&self) -> Result<UInt32Chunked>;

    /// Number of unique values in the `ChunkedArray`
    fn n_unique(&self) -> Result<usize> {
        self.arg_unique().map(|v| v.len())
    }

    /// Get a mask of all the unique values.
    fn is_unique(&self) -> Result<BooleanChunked> {
        Err(PolarsError::InvalidOperation(
            "is_unique is not implemented for this dtype".into(),
        ))
    }

    /// Get a mask of all the duplicated values.
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        Err(PolarsError::InvalidOperation(
            "is_duplicated is not implemented for this dtype".into(),
        ))
    }

    /// Count the unique values.
    fn value_counts(&self) -> Result<DataFrame> {
        Err(PolarsError::InvalidOperation(
            "is_duplicated is not implemented for this dtype".into(),
        ))
    }

    /// The most occurring value(s). Can return multiple Values
    #[cfg(feature = "mode")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mode")))]
    fn mode(&self) -> Result<ChunkedArray<T>> {
        Err(PolarsError::InvalidOperation(
            "mode is not implemented for this dtype".into(),
        ))
    }
}

pub trait ToDummies<T>: ChunkUnique<T> {
    fn to_dummies(&self) -> Result<DataFrame> {
        Err(PolarsError::InvalidOperation(
            "is_duplicated is not implemented for this dtype".into(),
        ))
    }
}

/// Sort operations on `ChunkedArray`.
pub trait ChunkSort<T> {
    /// Returned a sorted `ChunkedArray`.
    fn sort(&self, reverse: bool) -> ChunkedArray<T>;

    /// Sort this array in place.
    fn sort_in_place(&mut self, reverse: bool);

    /// Retrieve the indexes needed to sort this array.
    fn argsort(&self, reverse: bool) -> UInt32Chunked;

    /// Retrieve the indexes need to sort this and the other arrays.
    fn argsort_multiple(&self, _other: &[Series], _reverse: &[bool]) -> Result<UInt32Chunked> {
        Err(PolarsError::InvalidOperation(
            "argsort_multiple not implemented for this dtype".into(),
        ))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FillNullStrategy {
    /// previous value in array
    Backward,
    /// next value in array
    Forward,
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

/// Replace None values with various strategies
pub trait ChunkFillNull {
    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self>
    where
        Self: Sized;
}
/// Replace None values with a value
pub trait ChunkFillNullValue<T> {
    /// Replace None values with a give value `T`.
    fn fill_null_with_values(&self, value: T) -> Result<Self>
    where
        Self: Sized;
}

/// Fill a ChunkedArray with one value.
pub trait ChunkFull<T> {
    /// Create a ChunkedArray with a single value.
    fn full(name: &str, value: T, length: usize) -> Self
    where
        Self: std::marker::Sized;
}

pub trait ChunkFullNull {
    fn full_null(_name: &str, _length: usize) -> Self
    where
        Self: std::marker::Sized;
}

/// Reverse a ChunkedArray<T>
pub trait ChunkReverse<T> {
    /// Return a reversed version of this array.
    fn reverse(&self) -> ChunkedArray<T>;
}

/// Filter values by a boolean mask.
pub trait ChunkFilter<T> {
    /// Filter values in the ChunkedArray with a boolean mask.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let array = Int32Chunked::new_from_slice("array", &[1, 2, 3]);
    /// let mask = BooleanChunked::new_from_slice("mask", &[true, false, true]);
    ///
    /// let filtered = array.filter(&mask).unwrap();
    /// assert_eq!(Vec::from(&filtered), [Some(1), Some(3)])
    /// ```
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<T>>
    where
        Self: Sized;
}

/// Create a new ChunkedArray filled with values at that index.
pub trait ChunkExpandAtIndex<T> {
    /// Create a new ChunkedArray filled with values at that index.
    fn expand_at_index(&self, length: usize, index: usize) -> ChunkedArray<T>;
}

macro_rules! impl_chunk_expand {
    ($self:ident, $length:ident, $index:ident) => {{
        let opt_val = $self.get($index);
        match opt_val {
            Some(val) => ChunkedArray::full($self.name(), val, $length),
            None => ChunkedArray::full_null($self.name(), $length),
        }
    }};
}

impl<T> ChunkExpandAtIndex<T> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkFull<T::Native> + TakeRandom<Item = T::Native>,
    T: PolarsNumericType,
{
    fn expand_at_index(&self, index: usize, length: usize) -> ChunkedArray<T> {
        impl_chunk_expand!(self, length, index)
    }
}

impl ChunkExpandAtIndex<BooleanType> for BooleanChunked {
    fn expand_at_index(&self, index: usize, length: usize) -> BooleanChunked {
        impl_chunk_expand!(self, length, index)
    }
}

impl ChunkExpandAtIndex<Utf8Type> for Utf8Chunked {
    fn expand_at_index(&self, index: usize, length: usize) -> Utf8Chunked {
        impl_chunk_expand!(self, length, index)
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkExpandAtIndex<CategoricalType> for CategoricalChunked {
    fn expand_at_index(&self, index: usize, length: usize) -> CategoricalChunked {
        let ca: CategoricalChunked = self.deref().expand_at_index(index, length).into();
        ca.set_state(self)
    }
}

impl ChunkExpandAtIndex<ListType> for ListChunked {
    fn expand_at_index(&self, _index: usize, _length: usize) -> ListChunked {
        unimplemented!()
    }
}

#[cfg(feature = "object")]
impl<T> ChunkExpandAtIndex<ObjectType<T>> for ObjectChunked<T> {
    fn expand_at_index(&self, _index: usize, _length: usize) -> ObjectChunked<T> {
        todo!()
    }
}

/// Shift the values of a ChunkedArray by a number of periods.
pub trait ChunkShiftFill<T, V> {
    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `fill_value`.
    fn shift_and_fill(&self, periods: i64, fill_value: V) -> ChunkedArray<T>;
}

pub trait ChunkShift<T> {
    fn shift(&self, periods: i64) -> ChunkedArray<T>;
}

/// Combine 2 ChunkedArrays based on some predicate.
pub trait ChunkZip<T> {
    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    fn zip_with(&self, mask: &BooleanChunked, other: &ChunkedArray<T>) -> Result<ChunkedArray<T>>;
}

/// Apply kernels on the arrow array chunks in a ChunkedArray.
pub trait ChunkApplyKernel<A: Array> {
    /// Apply kernel and return result as a new ChunkedArray.
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&A) -> ArrayRef;

    /// Apply a kernel that outputs an array of different type.
    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&A) -> ArrayRef,
        S: PolarsDataType;
}

/// Find local minima/ maxima
pub trait ChunkPeaks {
    /// Get a boolean mask of the local maximum peaks.
    fn peak_max(&self) -> BooleanChunked {
        unimplemented!()
    }

    /// Get a boolean mask of the local minimum peaks.
    fn peak_min(&self) -> BooleanChunked {
        unimplemented!()
    }
}

/// Check if element is member of list array
#[cfg(feature = "is_in")]
#[cfg_attr(docsrs, doc(cfg(feature = "is_in")))]
pub trait IsIn {
    /// Check if elements of this array are in the right Series, or List values of the right Series.
    fn is_in(&self, _other: &Series) -> Result<BooleanChunked> {
        unimplemented!()
    }
}

/// Argmin/ Argmax
pub trait ArgAgg {
    /// Get the index of the minimal value
    fn arg_min(&self) -> Option<usize> {
        None
    }
    /// Get the index of the maximal value
    fn arg_max(&self) -> Option<usize> {
        None
    }
}

/// Repeat the values `n` times.
#[cfg(feature = "repeat_by")]
#[cfg_attr(docsrs, doc(cfg(feature = "repeat_by")))]
pub trait RepeatBy {
    /// Repeat the values `n` times, where `n` is determined by the values in `by`.
    fn repeat_by(&self, _by: &UInt32Chunked) -> ListChunked {
        unimplemented!()
    }
}

#[cfg(feature = "is_first")]
#[cfg_attr(docsrs, doc(cfg(feature = "is_first")))]
/// Mask the first unique values as `true`
pub trait IsFirst<T: PolarsDataType> {
    fn is_first(&self) -> Result<BooleanChunked> {
        Err(PolarsError::InvalidOperation(
            format!("operation not supported by {:?}", T::get_dtype()).into(),
        ))
    }
}

#[cfg(feature = "is_first")]
#[cfg_attr(docsrs, doc(cfg(feature = "is_first")))]
/// Mask the last unique values as `true`
pub trait IsLast<T: PolarsDataType> {
    fn is_last(&self) -> Result<BooleanChunked> {
        Err(PolarsError::InvalidOperation(
            format!("operation not supported by {:?}", T::get_dtype()).into(),
        ))
    }
}

#[cfg(feature = "concat_str")]
#[cfg_attr(docsrs, doc(cfg(feature = "concat_str")))]
/// Concat the values into a string array.
pub trait StrConcat {
    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    fn str_concat(&self, delimiter: &str) -> Utf8Chunked;
}
