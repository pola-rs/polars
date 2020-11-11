//! Traits for miscellaneous operations on ChunkedArray
use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::utils::Xob;
use arrow::{array::ArrayRef, compute::kernels::filter::filter_primitive_array};
use itertools::Itertools;
use std::cmp::Ordering;
use std::marker::Sized;
use std::sync::Arc;

pub(crate) mod aggregate;
pub(crate) mod apply;
pub(crate) mod chunkops;
pub(crate) mod fill_none;
pub(crate) mod set;
pub(crate) mod shift;
pub(crate) mod take;
pub(crate) mod unique;
pub(crate) mod window;
pub(crate) mod zip;

pub trait ChunkBytes {
    fn to_byte_slices(&self) -> Vec<&[u8]>;
}

pub trait ChunkWindow<T> {
    /// Apply a rolling sum (moving sum) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
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
    fn rolling_mean(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
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
    fn rolling_min(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
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
    fn rolling_max(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _ignore_null: bool,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(PolarsError::InvalidOperation(
            "rolling mean not supported for this datatype".into(),
        ))
    }

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
    fn rolling_custom<F>(
        &self,
        _window_size: usize,
        _weight: Option<&[f64]>,
        _fold_fn: F,
        _init_fold: InitFold,
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
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Get a value by index and ignore the null bit.
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item;
}
// Utility trait because associated type needs a lifetime
pub trait TakeRandomUtf8 {
    type Item;

    /// Get a nullable value by index.
    fn get(self, index: usize) -> Option<Self::Item>;

    /// Get a value by index and ignore the null bit.
    unsafe fn get_unchecked(self, index: usize) -> Self::Item;
}

/// Fast access by index.
pub trait ChunkTake {
    /// Take values from ChunkedArray by index.
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// Take values from ChunkedArray by index without checking bounds.
    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized;

    /// Take values from ChunkedArray by Option<index>.
    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// Take values from ChunkedArray by Option<index>.
    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized;
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
    /// # use polars::prelude::*;
    /// let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
    /// let new = ca.set_at_idx(&[0, 1], Some(10)).unwrap();
    ///
    /// assert_eq!(Vec::from(&new), &[Some(10), Some(10), Some(3)]);
    /// ```
    fn set_at_idx<T: AsTakeIndex>(&'a self, idx: &T, opt_value: Option<A>) -> Result<Self>
    where
        Self: Sized;

    /// Set the values at indexes `idx` by applying a closure to these values.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars::prelude::*;
    /// let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
    /// let new = ca.set_at_idx_with(&[0, 1], |opt_v| opt_v.map(|v| v - 5)).unwrap();
    ///
    /// assert_eq!(Vec::from(&new), &[Some(-4), Some(-3), Some(3)]);
    /// ```
    fn set_at_idx_with<T: AsTakeIndex, F>(&'a self, idx: &T, f: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(Option<A>) -> Option<B>;
    /// Set the values where the mask evaluates to `true` to some optional value `Option<T>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars::prelude::*;
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
    /// # use polars::prelude::*;
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
        N: PolarsDataType;
}

/// Fastest way to do elementwise operations on a ChunkedArray<T>
pub trait ChunkApply<'a, A, B> {
    /// Apply a closure `F` elementwise.
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(A) -> B + Copy;
}

/// Aggregation operations
pub trait ChunkAgg<T> {
    /// Aggregate the sum of the ChunkedArray.
    /// Returns `None` if the array is empty or only contains null values.
    fn sum(&self) -> Option<T>;

    fn min(&self) -> Option<T>;
    /// Returns the maximum value in the array, according to the natural order.
    /// Returns `None` if the array is empty or only contains null values.
    fn max(&self) -> Option<T>;

    /// Returns the mean value in the array.
    /// Returns `None` if the array is empty or only contains null values.
    fn mean(&self) -> Option<T>;

    /// Returns the mean value in the array.
    /// Returns `None` if the array is empty or only contains null values.
    fn median(&self) -> Option<T>;

    /// Aggregate a given quantile of the ChunkedArray.
    /// Returns `None` if the array is empty or only contains null values.
    fn quantile(&self, quantile: f64) -> Result<Option<T>>;
}

/// Compare [Series](series/series/enum.Series.html)
/// and [ChunkedArray](series/chunked_array/struct.ChunkedArray.html)'s and get a `boolean` mask that
/// can be used to filter rows.
///
/// # Example
///
/// ```
/// use polars::prelude::*;
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
    fn arg_unique(&self) -> Result<Vec<usize>>;

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
}

/// Sort operations on `ChunkedArray`.
pub trait ChunkSort<T> {
    /// Returned a sorted `ChunkedArray`.
    fn sort(&self, reverse: bool) -> ChunkedArray<T>;

    /// Sort this array in place.
    fn sort_in_place(&mut self, reverse: bool);

    /// Retrieve the indexes needed to sort this array.
    fn argsort(&self, reverse: bool) -> Vec<usize>;
}

fn sort_partial<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.partial_cmp(b).expect("could not compare"),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: std::cmp::PartialOrd,
{
    fn sort(&self, reverse: bool) -> ChunkedArray<T> {
        if reverse {
            self.into_iter()
                .sorted_by(|a, b| sort_partial(b, a))
                .collect()
        } else {
            self.into_iter()
                .sorted_by(|a, b| sort_partial(a, b))
                .collect()
        }
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            self.into_iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| sort_partial(b, a))
                .map(|(idx, _v)| idx)
                .collect()
        } else {
            self.into_iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| sort_partial(a, b))
                .map(|(idx, _v)| idx)
                .collect()
        }
    }
}

macro_rules! argsort {
    ($self:ident, $closure:expr) => {{
        $self
            .into_iter()
            .enumerate()
            .sorted_by($closure)
            .map(|(idx, _v)| idx)
            .collect()
    }};
}

macro_rules! sort {
    ($self:ident, $reverse:ident) => {{
        if $reverse {
            $self.into_iter().sorted_by(|a, b| b.cmp(a)).collect()
        } else {
            $self.into_iter().sorted_by(|a, b| a.cmp(b)).collect()
        }
    }};
}

impl ChunkSort<Utf8Type> for Utf8Chunked {
    fn sort(&self, reverse: bool) -> Utf8Chunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| b.cmp(a))
        } else {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| a.cmp(b))
        }
    }
}

impl ChunkSort<ListType> for ListChunked {
    fn sort(&self, _reverse: bool) -> Self {
        println!("A ListChunked cannot be sorted. Doing nothing");
        self.clone()
    }

    fn sort_in_place(&mut self, _reverse: bool) {
        println!("A ListChunked cannot be sorted. Doing nothing");
    }

    fn argsort(&self, _reverse: bool) -> Vec<usize> {
        println!("A ListChunked cannot be sorted. Doing nothing");
        (0..self.len()).collect()
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort(&self, reverse: bool) -> BooleanChunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| b.cmp(a))
        } else {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| a.cmp(b))
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FillNoneStrategy {
    Backward,
    Forward,
    Mean,
    Min,
    Max,
}

/// Replace None values with various strategies
pub trait ChunkFillNone<T> {
    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self>
    where
        Self: Sized;

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

    fn full_null(_name: &str, _length: usize) -> Self
    where
        Self: std::marker::Sized;
}

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn full(name: &str, value: T::Native, length: usize) -> Self
    where
        T::Native: Copy,
    {
        let mut builder = PrimitiveChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_value(value)
        }
        builder.finish()
    }

    fn full_null(name: &str, length: usize) -> Self {
        let mut builder = PrimitiveChunkedBuilder::new(name, length);

        // todo: faster with null arrays or in one go allocation
        for _ in 0..length {
            builder.append_null()
        }
        builder.finish()
    }
}

impl<'a> ChunkFull<&'a str> for Utf8Chunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_value(value);
        }
        builder.finish()
    }

    fn full_null(name: &str, length: usize) -> Self {
        // todo: faster with null arrays or in one go allocation
        let mut builder = Utf8ChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_null()
        }
        builder.finish()
    }
}

impl ChunkFull<Series> for ListChunked {
    fn full(_name: &str, _value: Series, _length: usize) -> ListChunked {
        unimplemented!()
    }

    fn full_null(_name: &str, _length: usize) -> ListChunked {
        unimplemented!()
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
            let ca: Xob<ChunkedArray<T>> = slice.iter().rev().copied().collect();
            let mut ca = ca.into_inner();
            ca.rename(self.name());
            ca
        } else {
            self.take((0..self.len()).rev(), None)
                .expect("implementation error, should not fail")
        }
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse<$arrow_type> for $ca_type {
            fn reverse(&self) -> Self {
                self.take((0..self.len()).rev(), None)
                    .expect("implementation error, should not fail")
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(Utf8Type, Utf8Chunked);
impl_reverse!(ListType, ListChunked);

/// Filter values by a boolean mask.
pub trait ChunkFilter<T> {
    /// Filter values in the ChunkedArray with a boolean mask.
    ///
    /// ```rust
    /// # use polars::prelude::*;
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

macro_rules! impl_filter_with_nulls_in_both {
    ($self:expr, $filter:expr) => {{
        let ca = $self
            .into_iter()
            .zip($filter)
            .filter_map(|(val, valid)| match valid {
                Some(valid) => {
                    if valid {
                        Some(val)
                    } else {
                        None
                    }
                }
                None => None,
            })
            .collect();
        Ok(ca)
    }};
}

macro_rules! impl_filter_no_nulls_in_mask {
    ($self:expr, $filter:expr) => {{
        let ca = $self
            .into_iter()
            .zip($filter.into_no_null_iter())
            .filter_map(|(val, valid)| if valid { Some(val) } else { None })
            .collect();
        Ok(ca)
    }};
}

macro_rules! check_filter_len {
    ($self:expr, $filter:expr) => {{
        if $self.len() != $filter.len() {
            return Err(PolarsError::ShapeMisMatch(
                "Filter's length differs from that of the ChunkedArray/ Series.".into(),
            ));
        }
    }};
}

macro_rules! impl_filter_no_nulls {
    ($self:expr, $filter:expr) => {{
        $self
            .into_no_null_iter()
            .zip($filter.into_no_null_iter())
            .filter_map(|(val, valid)| if valid { Some(val) } else { None })
            .collect()
    }};
}

macro_rules! impl_filter_no_nulls_in_self {
    ($self:expr, $filter:expr) => {{
        $self
            .into_no_null_iter()
            .zip($filter)
            .filter_map(|(val, valid)| match valid {
                Some(valid) => {
                    if valid {
                        Some(val)
                    } else {
                        None
                    }
                }
                None => None,
            })
            .collect()
    }};
}

impl<T> ChunkFilter<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkOps,
{
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<T>> {
        check_filter_len!(self, filter);
        if self.chunk_id == filter.chunk_id {
            let chunks = self
                .downcast_chunks()
                .iter()
                .zip(filter.downcast_chunks())
                .map(|(&left, mask)| {
                    Arc::new(filter_primitive_array(left, mask).unwrap()) as ArrayRef
                })
                .collect::<Vec<_>>();
            return Ok(ChunkedArray::new_from_chunks(self.name(), chunks));
        }
        let out = match (self.null_count(), filter.null_count()) {
            (0, 0) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls!(self, filter);
                Ok(ca.into_inner())
            }
            (0, _) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls_in_self!(self, filter);
                Ok(ca.into_inner())
            }
            (_, 0) => impl_filter_no_nulls_in_mask!(self, filter),
            (_, _) => impl_filter_with_nulls_in_both!(self, filter),
        };
        out.map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

impl ChunkFilter<BooleanType> for BooleanChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<BooleanType>> {
        check_filter_len!(self, filter);
        let out = match (self.null_count(), filter.null_count()) {
            (0, 0) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls!(self, filter);
                Ok(ca.into_inner())
            }
            (0, _) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls_in_self!(self, filter);
                Ok(ca.into_inner())
            }
            (_, 0) => impl_filter_no_nulls_in_mask!(self, filter),
            (_, _) => impl_filter_with_nulls_in_both!(self, filter),
        };
        out.map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}
impl ChunkFilter<Utf8Type> for Utf8Chunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<Utf8Type>> {
        check_filter_len!(self, filter);
        let out: Result<Utf8Chunked> = match (self.null_count(), filter.null_count()) {
            (0, 0) => {
                let ca = impl_filter_no_nulls!(self, filter);
                Ok(ca)
            }
            (0, _) => {
                let ca = impl_filter_no_nulls_in_self!(self, filter);
                Ok(ca)
            }
            (_, 0) => impl_filter_no_nulls_in_mask!(self, filter),
            (_, _) => impl_filter_with_nulls_in_both!(self, filter),
        };

        out.map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

impl ChunkFilter<ListType> for ListChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ListChunked> {
        let dt = self.get_inner_dtype();
        let mut builder = get_list_builder(dt, self.len(), self.name());
        filter
            .into_iter()
            .zip(self.into_iter())
            .for_each(|(opt_bool_val, opt_series)| {
                let bool_val = opt_bool_val.unwrap_or(false);
                let opt_val = match bool_val {
                    true => opt_series,
                    false => None,
                };
                builder.append_opt_series(&opt_val)
            });
        Ok(builder.finish())
    }
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
    T: ArrowPrimitiveType,
{
    fn expand_at_index(&self, index: usize, length: usize) -> ChunkedArray<T> {
        impl_chunk_expand!(self, length, index)
    }
}

impl ChunkExpandAtIndex<Utf8Type> for Utf8Chunked {
    fn expand_at_index(&self, index: usize, length: usize) -> Utf8Chunked {
        impl_chunk_expand!(self, length, index)
    }
}

impl ChunkExpandAtIndex<ListType> for ListChunked {
    fn expand_at_index(&self, index: usize, length: usize) -> ListChunked {
        impl_chunk_expand!(self, length, index)
    }
}

/// Shift the values of a ChunkedArray by a number of periods.
pub trait ChunkShift<T, V> {
    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `fill_value`.
    fn shift(&self, periods: i32, fill_value: &Option<V>) -> Result<ChunkedArray<T>>;
}

/// Combine 2 ChunkedArrays based on some predicate.
pub trait ChunkZip<T> {
    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    fn zip_with(&self, mask: &BooleanChunked, other: &ChunkedArray<T>) -> Result<ChunkedArray<T>>;

    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    fn zip_with_series(&self, mask: &BooleanChunked, other: &Series) -> Result<ChunkedArray<T>>;
}

/// Aggregations that return Series of unit length. Those can be used in broadcasting operations.
pub trait ChunkAggSeries {
    /// Get the sum of the ChunkedArray as a new Series of length 1.
    fn sum_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the max of the ChunkedArray as a new Series of length 1.
    fn max_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the min of the ChunkedArray as a new Series of length 1.
    fn min_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the mean of the ChunkedArray as a new Series of length 1.
    fn mean_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the median of the ChunkedArray as a new Series of length 1.
    fn median_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_as_series(&self, _quantile: f64) -> Result<Series> {
        unimplemented!()
    }
}

/// Apply kernels on the arrow array chunks in a ChunkedArray.
pub trait ChunkApplyKernel<A> {
    /// Apply kernel and return result as a new ChunkedArray.
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&A) -> ArrayRef;
    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&A) -> ArrayRef,
        S: PolarsDataType;
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_shift() {
        let ca = Int32Chunked::new_from_slice("", &[1, 2, 3]);
        let shifted = ca.shift(1, &Some(0)).unwrap();
        assert_eq!(shifted.cont_slice().unwrap(), &[0, 1, 2]);
        let shifted = ca.shift(1, &None).unwrap();
        assert_eq!(Vec::from(&shifted), &[None, Some(1), Some(2)]);
        let shifted = ca.shift(-1, &None).unwrap();
        assert_eq!(Vec::from(&shifted), &[Some(2), Some(3), None]);
        assert!(ca.shift(3, &None).is_err());

        let s = Series::new("a", ["a", "b", "c"]);
        let shifted = s.shift(-1).unwrap();
        assert_eq!(
            Vec::from(shifted.utf8().unwrap()),
            &[Some("b"), Some("c"), None]
        );
    }

    #[test]
    fn test_fill_none() {
        let ca =
            Int32Chunked::new_from_opt_slice("", &[None, Some(2), Some(3), None, Some(4), None]);
        let filled = ca.fill_none(FillNoneStrategy::Forward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[None, Some(2), Some(3), Some(3), Some(4), Some(4)]
        );
        let filled = ca.fill_none(FillNoneStrategy::Backward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(4), Some(4), None]
        );
        let filled = ca.fill_none(FillNoneStrategy::Min).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(2), Some(4), Some(2)]
        );
        let filled = ca.fill_none_with_value(10).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(10), Some(2), Some(3), Some(10), Some(4), Some(10)]
        );
        let filled = ca.fill_none(FillNoneStrategy::Mean).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(3), Some(2), Some(3), Some(3), Some(4), Some(3)]
        );
        println!("{:?}", filled);
    }
}
