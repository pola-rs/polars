//! Traits for miscellaneous operations on ChunkedArray
use std::marker::Sized;

use arrow::array::ArrayRef;

use crate::chunked_array::builder::get_list_builder;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectType;
use crate::prelude::*;
use crate::utils::NoNull;

pub use self::take::*;

pub(crate) mod aggregate;
pub(crate) mod any_value;
pub(crate) mod apply;
pub(crate) mod bit_repr;
pub(crate) mod chunkops;
pub(crate) mod compare_inner;
pub(crate) mod cum_agg;
pub(crate) mod downcast;
pub(crate) mod explode;
pub(crate) mod fill_none;
pub(crate) mod filter;
#[cfg(feature = "is_in")]
pub(crate) mod is_in;
pub(crate) mod peaks;
#[cfg(feature = "repeat_by")]
pub(crate) mod repeat_by;
pub(crate) mod set;
pub(crate) mod shift;
pub(crate) mod sort;
pub(crate) mod take;
pub(crate) mod unique;
pub(crate) mod window;
#[cfg(feature = "zip_with")]
pub(crate) mod zip;

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

pub trait ChunkCumAgg<T> {
    /// Get an array with the cumulative max computed at every element
    fn cum_max(&self, _reverse: bool) -> ChunkedArray<T> {
        panic!("operation cum_max not supported for this dtype")
    }
    /// Get an array with the cumulative min computed at every element
    fn cum_min(&self, _reverse: bool) -> ChunkedArray<T> {
        panic!("operation cum_min not supported for this dtype")
    }
    /// Get an array with the cumulative sum computed at every element
    fn cum_sum(&self, _reverse: bool) -> ChunkedArray<T> {
        panic!("operation cum_sum not supported for this dtype")
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
    fn explode_and_offsets(&self) -> Result<(Series, &[i64])>;
}

pub trait ChunkBytes {
    fn to_byte_slices(&self) -> Vec<&[u8]>;
}

/// Rolling window functions
pub trait ChunkWindow {
    /// apply a rolling sum (moving sum) over the values in this array.
    /// a window of length `window_size` will traverse the array. the values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weight` vector. the resulting
    /// values will be aggregated to their sum.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The length of the window.
    /// * `weight` - An optional slice with the same length of the window that will be multiplied
    ///              elementwise with the values in the window.
    /// * `ignore_null` - Toggle behavior of aggregation regarding null values in the window.
    ///                     `true` -> Null values will be ignored.
    ///                     `false` -> Any Null in the window leads to a Null in the aggregation result.
    fn rolling_sum(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(PolarsError::InvalidOperation(
            "rolling sum not supported for this datatype".into(),
        ))
    }
    /// Apply a rolling mean (moving mean) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
    /// values will be aggregated to their mean.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The length of the window.
    /// * `weight` - An optional slice with the same length of the window that will be multiplied
    ///              elementwise with the values in the window.
    /// * `ignore_null` - Toggle behavior of aggregation regarding null values in the window.
    ///                     `true` -> Null values will be ignored.
    ///                     `false` -> Any Null in the window leads to a Null in the aggregation result.
    /// * `min_periods` -  Amount of elements in the window that should be filled before computing a result.
    fn rolling_mean(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(PolarsError::InvalidOperation(
            "rolling mean not supported for this datatype".into(),
        ))
    }

    /// Apply a rolling min (moving min) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
    /// values will be aggregated to their min.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The length of the window.
    /// * `weight` - An optional slice with the same length of the window that will be multiplied
    ///              elementwise with the values in the window.
    /// * `ignore_null` - Toggle behavior of aggregation regarding null values in the window.
    ///                     `true` -> Null values will be ignored.
    ///                     `false` -> Any Null in the window leads to a Null in the aggregation result.
    /// * `min_periods` -  Amount of elements in the window that should be filled before computing a result.
    fn rolling_min(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(PolarsError::InvalidOperation(
            "rolling mean not supported for this datatype".into(),
        ))
    }

    /// Apply a rolling max (moving max) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
    /// values will be aggregated to their max.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The length of the window.
    /// * `weight` - An optional slice with the same length of the window that will be multiplied
    ///              elementwise with the values in the window.
    /// * `ignore_null` - Toggle behavior of aggregation regarding null values in the window.
    ///                     `true` -> Null values will be ignored.
    ///                     `false` -> Any Null in the window leads to a Null in the aggregation result.
    /// * `min_periods` -  Amount of elements in the window that should be filled before computing a result.
    fn rolling_max(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
        _min_periods: u32,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(PolarsError::InvalidOperation(
            "rolling mean not supported for this datatype".into(),
        ))
    }
}

/// Custom rolling window functions
pub trait ChunkWindowCustom<T> {
    /// Apply a rolling aggregation over the values in this array.
    ///
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
    /// values will be aggregated to their max.
    ///
    /// You can pass a custom closure that will be used in the `fold` operation to aggregate the window.
    /// The closure/fn of type `Fn(Option<T>, Option<T>) -> Option<T>` takes an `accumulator` and
    /// a `value` as argument.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The length of the window.
    /// * `weight` - An optional slice with the same length of the window that will be multiplied
    ///              elementwise with the values in the window.
    /// * `min_periods` -  Amount of elements in the window that should be filled before computing a result.
    fn rolling_custom<F>(
        &self,
        _window_size: u32,
        _weight: Option<&[f64]>,
        _fold_fn: F,
        _init_fold: InitFold,
        _min_periods: u32,
    ) -> Result<Self>
    where
        F: Fn(Option<T>, Option<T>) -> Option<T> + Copy,
        Self: std::marker::Sized,
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
    /// Out of bounds access doesn't Error but will return a Null value
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
    /// Out of bounds access doesn't Error but will return a Null value
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
    /// let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
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
    /// let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
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
    /// Cast `ChunkedArray<T>` to `ChunkedArray<N>`
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
        Self: Sized;

    fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series>;
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
pub enum FillNoneStrategy {
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
pub trait ChunkFillNone {
    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self>
    where
        Self: Sized;
}
/// Replace None values with a value
pub trait ChunkFillNoneValue<T> {
    /// Replace None values with a give value `T`.
    fn fill_none_with_value(&self, value: T) -> Result<Self>
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

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn full(name: &str, value: T::Native, length: usize) -> Self
    where
        T::Native: Copy,
    {
        let mut ca = (0..length)
            .map(|_| value)
            .collect::<NoNull<ChunkedArray<T>>>()
            .into_inner();
        ca.rename(name);
        ca
    }
}

impl<T> ChunkFullNull for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn full_null(name: &str, length: usize) -> Self {
        let mut ca = (0..length).map(|_| None).collect::<Self>();
        ca.rename(name);
        ca
    }
}
impl ChunkFull<bool> for BooleanChunked {
    fn full(name: &str, value: bool, length: usize) -> Self {
        let mut ca = (0..length).map(|_| value).collect::<BooleanChunked>();
        ca.rename(name);
        ca
    }
}

impl ChunkFullNull for BooleanChunked {
    fn full_null(name: &str, length: usize) -> Self {
        let mut ca = (0..length).map(|_| None).collect::<Self>();
        ca.rename(name);
        ca
    }
}

impl<'a> ChunkFull<&'a str> for Utf8Chunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, length, length * value.len());

        for _ in 0..length {
            builder.append_value(value);
        }
        builder.finish()
    }
}

impl ChunkFullNull for Utf8Chunked {
    fn full_null(name: &str, length: usize) -> Self {
        let mut ca = (0..length)
            .map::<Option<String>, _>(|_| None)
            .collect::<Self>();
        ca.rename(name);
        ca
    }
}

impl ChunkFull<&Series> for ListChunked {
    fn full(name: &str, value: &Series, length: usize) -> ListChunked {
        let mut builder = get_list_builder(value.dtype(), value.len() * length, length, name);
        for _ in 0..length {
            builder.append_series(value)
        }
        builder.finish()
    }
}

impl ChunkFullNull for ListChunked {
    fn full_null(name: &str, length: usize) -> ListChunked {
        let mut ca = (0..length)
            .map::<Option<Series>, _>(|_| None)
            .collect::<Self>();
        ca.rename(name);
        ca
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFullNull for ObjectChunked<T> {
    fn full_null(name: &str, length: usize) -> ObjectChunked<T> {
        let mut ca: Self = (0..length).map(|_| None).collect();
        ca.rename(name);
        ca
    }
}

/// Reverse a ChunkedArray<T>
pub trait ChunkReverse<T> {
    /// Return a reversed version of this array.
    fn reverse(&self) -> ChunkedArray<T>;
}

impl<T> ChunkReverse<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkOps,
{
    fn reverse(&self) -> ChunkedArray<T> {
        if let Ok(slice) = self.cont_slice() {
            let ca: NoNull<ChunkedArray<T>> = slice.iter().rev().copied().collect();
            let mut ca = ca.into_inner();
            ca.rename(self.name());
            ca
        } else {
            self.into_iter().rev().collect()
        }
    }
}

impl ChunkReverse<CategoricalType> for CategoricalChunked {
    fn reverse(&self) -> ChunkedArray<CategoricalType> {
        self.cast::<UInt32Type>().unwrap().reverse().cast().unwrap()
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse<$arrow_type> for $ca_type {
            fn reverse(&self) -> Self {
                unsafe { self.take_unchecked((0..self.len()).rev().into()) }
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(Utf8Type, Utf8Chunked);
impl_reverse!(ListType, ListChunked);
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkReverse<ObjectType<T>> for ObjectChunked<T> {
    fn reverse(&self) -> Self {
        // Safety
        // we we know we don't get out of bounds
        unsafe { self.take_unchecked((0..self.len()).rev().into()) }
    }
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
    T: PolarsPrimitiveType,
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

impl ChunkExpandAtIndex<CategoricalType> for CategoricalChunked {
    fn expand_at_index(&self, index: usize, length: usize) -> CategoricalChunked {
        self.cast::<UInt32Type>()
            .unwrap()
            .expand_at_index(index, length)
            .cast()
            .unwrap()
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
pub trait ChunkApplyKernel<A> {
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
pub trait IsFirst<T: PolarsDataType> {
    fn is_first(&self) -> Result<BooleanChunked> {
        Err(PolarsError::InvalidOperation(
            format!("operation not supported by {:?}", T::get_dtype()).into(),
        ))
    }
}

#[cfg(feature = "is_first")]
#[cfg_attr(docsrs, doc(cfg(feature = "is_first")))]
pub trait IsLast<T: PolarsDataType> {
    fn is_last(&self) -> Result<BooleanChunked> {
        Err(PolarsError::InvalidOperation(
            format!("operation not supported by {:?}", T::get_dtype()).into(),
        ))
    }
}
