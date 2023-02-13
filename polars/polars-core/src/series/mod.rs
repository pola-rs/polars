//! Type agnostic columnar data structure.
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;

mod any_value;
pub mod arithmetic;
mod comparison;
mod from;
pub mod implementations;
mod into;
pub(crate) mod iterator;
pub mod ops;
mod series_trait;
#[cfg(feature = "private")]
pub mod unstable;

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use ahash::RandomState;
use arrow::compute::aggregate::estimated_bytes_size;
pub use from::*;
pub use iterator::SeriesIter;
use num::NumCast;
use rayon::prelude::*;
pub use series_trait::{IsSorted, *};

#[cfg(feature = "rank")]
use crate::prelude::unique::rank::rank;
#[cfg(feature = "zip_with")]
use crate::series::arithmetic::coerce_lhs_rhs;
use crate::utils::{_split_offsets, split_ca, split_series, Wrap};
use crate::POOL;

/// # Series
/// The columnar data type for a DataFrame.
///
/// Most of the available functions are defined in the [SeriesTrait trait](crate::series::SeriesTrait).
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
/// let s = Series::new("a", [1 , 2, 3]);
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
/// let s = Series::new("dollars", &[1, 2, 3]);
/// let mask = s.equal(1).unwrap();
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
/// // Series can be created from Vec's, slices and arrays
/// Series::new("boolean series", &[true, false, true]);
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
#[must_use]
pub struct Series(pub Arc<dyn SeriesTrait>);

impl PartialEq for Wrap<Series> {
    fn eq(&self, other: &Self) -> bool {
        self.0.series_equal_missing(other)
    }
}

impl Eq for Wrap<Series> {}

impl Hash for Wrap<Series> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let rs = RandomState::with_seeds(0, 0, 0, 0);
        let mut h = vec![];
        self.0.vec_hash(rs, &mut h).unwrap();
        let h = UInt64Chunked::from_vec("", h).sum();
        h.hash(state)
    }
}

impl Series {
    /// Create a new empty Series
    pub fn new_empty(name: &str, dtype: &DataType) -> Series {
        Series::full_null(name, 0, dtype)
    }

    #[doc(hidden)]
    #[cfg(feature = "private")]
    pub fn _get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
        if Arc::weak_count(&self.0) + Arc::strong_count(&self.0) != 1 {
            self.0 = self.0.clone_inner();
        }
        Arc::get_mut(&mut self.0).expect("implementation error")
    }

    /// # Safety
    /// The caller must ensure the length and the data types of `ArrayRef` does not change.
    pub(crate) unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        #[allow(unused_mut)]
        let mut ca = self._get_inner_mut();
        let chunks = ca.chunks() as *const Vec<ArrayRef> as *mut Vec<ArrayRef>;
        // Safety
        // ca is the owner of `chunks` and this we do not break aliasing rules
        &mut *chunks
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        let inner = self._get_inner_mut();
        inner._set_sorted_flag(sorted)
    }

    pub fn into_frame(self) -> DataFrame {
        DataFrame::new_no_checks(vec![self])
    }

    /// Rename series.
    pub fn rename(&mut self, name: &str) -> &mut Series {
        self._get_inner_mut().rename(name);
        self
    }

    /// Shrink the capacity of this array to fit its length.
    pub fn shrink_to_fit(&mut self) {
        self._get_inner_mut().shrink_to_fit()
    }

    /// Append in place. This is done by adding the chunks of `other` to this [`Series`].
    ///
    /// See [`ChunkedArray::append`] and [`ChunkedArray::extend`].
    pub fn append(&mut self, other: &Series) -> PolarsResult<&mut Self> {
        self._get_inner_mut().append(other)?;
        Ok(self)
    }

    /// Extend the memory backed by this array with the values from `other`.
    ///
    /// See [`ChunkedArray::extend`] and [`ChunkedArray::append`].
    pub fn extend(&mut self, other: &Series) -> PolarsResult<&mut Self> {
        self._get_inner_mut().extend(other)?;
        Ok(self)
    }

    pub fn sort(&self, reverse: bool) -> Self {
        self.sort_with(SortOptions {
            descending: reverse,
            ..Default::default()
        })
    }

    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self._get_inner_mut().as_single_ptr()
    }

    /// Cast `[Series]` to another `[DataType]`
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        match self.0.cast(dtype) {
            Ok(out) => Ok(out),
            Err(err) => {
                let len = self.len();
                if self.null_count() == len {
                    Ok(Series::full_null(self.name(), len, dtype))
                } else {
                    Err(err)
                }
            }
        }
    }

    /// Cast from physical to logical types without any checks on the validity of the cast.
    ///
    /// # Safety
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        match self.dtype() {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let ca: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                        ca.cast_unchecked(dtype)
                })
            }
            _ => self.cast(dtype),
        }
    }

    /// Compute the sum of all values in this Series.
    /// Returns `Some(0)` if the array is empty, and `None` if the array only
    /// contains null values.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("days", &[1, 2, 3]);
    /// assert_eq!(s.sum(), Some(6));
    /// ```
    pub fn sum<T>(&self) -> Option<T>
    where
        T: NumCast,
    {
        self.sum_as_series()
            .cast(&DataType::Float64)
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
            .cast(&DataType::Float64)
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
            .cast(&DataType::Float64)
            .ok()
            .and_then(|s| s.f64().unwrap().get(0).and_then(T::from))
    }

    /// Explode a list or utf8 Series. This expands every item to a new row..
    pub fn explode(&self) -> PolarsResult<Series> {
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
    pub fn is_nan(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_nan()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "'is_nan' not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is NaN (note this is different than missing/ null)
    pub fn is_not_nan(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_not_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_not_nan()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "'is_not_nan' not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is finite
    pub fn is_finite(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_finite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_finite()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "'is_finite' not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is infinite
    pub fn is_infinite(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_infinite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_infinite()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "'is_infinite' not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    #[cfg(feature = "zip_with")]
    pub fn zip_with(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let (lhs, rhs) = coerce_lhs_rhs(self, other)?;
        lhs.zip_with_same_type(mask, rhs.as_ref())
    }

    /// Cast a datelike Series to their physical representation.
    /// Primitives remain unchanged
    ///
    /// * Date -> Int32
    /// * Datetime-> Int64
    /// * Time -> Int64
    /// * Categorical -> UInt32
    ///
    pub fn to_physical_repr(&self) -> Cow<Series> {
        use DataType::*;
        match self.dtype() {
            Date => Cow::Owned(self.cast(&DataType::Int32).unwrap()),
            Datetime(_, _) | Duration(_) | Time => Cow::Owned(self.cast(&DataType::Int64).unwrap()),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => Cow::Owned(self.cast(&DataType::UInt32).unwrap()),
            _ => Cow::Borrowed(self),
        }
    }

    fn finish_take_threaded(&self, s: Vec<Series>, rechunk: bool) -> Series {
        let s = s
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

    // take a function pointer to reduce bloat
    fn threaded_op(
        &self,
        rechunk: bool,
        len: usize,
        func: &(dyn Fn(usize, usize) -> PolarsResult<Series> + Send + Sync),
    ) -> PolarsResult<Series> {
        let n_threads = POOL.current_num_threads();
        let offsets = _split_offsets(len, n_threads);

        let series: PolarsResult<Vec<_>> = POOL.install(|| {
            offsets
                .into_par_iter()
                .map(|(offset, len)| func(offset, len))
                .collect()
        });

        Ok(self.finish_take_threaded(series?, rechunk))
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_unchecked_from_slice(&self, idx: &[IdxSize]) -> PolarsResult<Series> {
        let idx = IdxCa::mmap_slice("", idx);
        self.take_unchecked(&idx)
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_unchecked_threaded(
        &self,
        idx: &IdxCa,
        rechunk: bool,
    ) -> PolarsResult<Series> {
        self.threaded_op(rechunk, idx.len(), &|offset, len| {
            let idx = idx.slice(offset as i64, len);
            self.take_unchecked(&idx)
        })
    }

    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    #[cfg(feature = "chunked_ids")]
    pub(crate) unsafe fn _take_chunked_unchecked_threaded(
        &self,
        chunk_ids: &[ChunkId],
        sorted: IsSorted,
        rechunk: bool,
    ) -> Series {
        self.threaded_op(rechunk, chunk_ids.len(), &|offset, len| {
            let chunk_ids = &chunk_ids[offset..offset + len];
            Ok(self._take_chunked_unchecked(chunk_ids, sorted))
        })
        .unwrap()
    }

    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    #[cfg(feature = "chunked_ids")]
    pub(crate) unsafe fn _take_opt_chunked_unchecked_threaded(
        &self,
        chunk_ids: &[Option<ChunkId>],
        rechunk: bool,
    ) -> Series {
        self.threaded_op(rechunk, chunk_ids.len(), &|offset, len| {
            let chunk_ids = &chunk_ids[offset..offset + len];
            Ok(self._take_opt_chunked_unchecked(chunk_ids))
        })
        .unwrap()
    }

    /// Take by index. This operation is clone.
    ///
    /// # Notes
    /// Out of bounds access doesn't Error but will return a Null value
    pub fn take_threaded(&self, idx: &IdxCa, rechunk: bool) -> PolarsResult<Series> {
        self.threaded_op(rechunk, idx.len(), &|offset, len| {
            let idx = idx.slice(offset as i64, len);
            self.take(&idx)
        })
    }

    /// Filter by boolean mask. This operation clones data.
    pub fn filter_threaded(&self, filter: &BooleanChunked, rechunk: bool) -> PolarsResult<Series> {
        // this would fail if there is a broadcasting filter.
        // because we cannot split that filter over threads
        // besides they are a no-op, so we do the standard filter.
        if filter.len() == 1 {
            return self.filter(filter);
        }
        let n_threads = POOL.current_num_threads();
        let filters = split_ca(filter, n_threads).unwrap();
        let series = split_series(self, n_threads).unwrap();

        let series: PolarsResult<Vec<_>> = POOL.install(|| {
            filters
                .par_iter()
                .zip(series)
                .map(|(filter, s)| s.filter(filter))
                .collect()
        });

        Ok(self.finish_take_threaded(series?, rechunk))
    }

    #[cfg(feature = "dot_product")]
    pub fn dot(&self, other: &Series) -> Option<f64> {
        (self * other).sum::<f64>()
    }

    /// Get the sum of the Series as a new Series of length 1.
    /// Returns a Series with a single zeroed entry if self is an empty numeric series.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    pub fn sum_as_series(&self) -> Series {
        use DataType::*;
        if self.is_empty() && self.dtype().is_numeric() {
            return Series::new("", [0])
                .cast(self.dtype())
                .unwrap()
                .sum_as_series();
        }
        match self.dtype() {
            Int8 | UInt8 | Int16 | UInt16 => self.cast(&Int64).unwrap().sum_as_series(),
            _ => self._sum_as_series(),
        }
    }

    /// Get an array with the cumulative max computed at every element
    pub fn cummax(&self, _reverse: bool) -> Series {
        #[cfg(feature = "cum_agg")]
        {
            self._cummax(_reverse)
        }
        #[cfg(not(feature = "cum_agg"))]
        {
            panic!("activate 'cum_agg' feature")
        }
    }

    /// Get an array with the cumulative min computed at every element
    pub fn cummin(&self, _reverse: bool) -> Series {
        #[cfg(feature = "cum_agg")]
        {
            self._cummin(_reverse)
        }
        #[cfg(not(feature = "cum_agg"))]
        {
            panic!("activate 'cum_agg' feature")
        }
    }

    /// Get an array with the cumulative sum computed at every element
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    #[allow(unused_variables)]
    pub fn cumsum(&self, reverse: bool) -> Series {
        #[cfg(feature = "cum_agg")]
        {
            use DataType::*;
            match self.dtype() {
                Boolean => self.cast(&DataType::UInt32).unwrap().cumsum(reverse),
                Int8 | UInt8 | Int16 | UInt16 => {
                    let s = self.cast(&Int64).unwrap();
                    s.cumsum(reverse)
                }
                Int32 => {
                    let ca = self.i32().unwrap();
                    ca.cumsum(reverse).into_series()
                }
                UInt32 => {
                    let ca = self.u32().unwrap();
                    ca.cumsum(reverse).into_series()
                }
                UInt64 => {
                    let ca = self.u64().unwrap();
                    ca.cumsum(reverse).into_series()
                }
                Int64 => {
                    let ca = self.i64().unwrap();
                    ca.cumsum(reverse).into_series()
                }
                Float32 => {
                    let ca = self.f32().unwrap();
                    ca.cumsum(reverse).into_series()
                }
                Float64 => {
                    let ca = self.f64().unwrap();
                    ca.cumsum(reverse).into_series()
                }
                dt => panic!("cumsum not supported for dtype: {dt:?}"),
            }
        }
        #[cfg(not(feature = "cum_agg"))]
        {
            panic!("activate 'cum_agg' feature")
        }
    }

    /// Get an array with the cumulative product computed at every element
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16, Int32, UInt32}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    #[allow(unused_variables)]
    pub fn cumprod(&self, reverse: bool) -> Series {
        #[cfg(feature = "cum_agg")]
        {
            use DataType::*;
            match self.dtype() {
                Boolean => self.cast(&DataType::Int64).unwrap().cumprod(reverse),
                Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 => {
                    let s = self.cast(&Int64).unwrap();
                    s.cumprod(reverse)
                }
                Int64 => {
                    let ca = self.i64().unwrap();
                    ca.cumprod(reverse).into_series()
                }
                UInt64 => {
                    let ca = self.u64().unwrap();
                    ca.cumprod(reverse).into_series()
                }
                Float32 => {
                    let ca = self.f32().unwrap();
                    ca.cumprod(reverse).into_series()
                }
                Float64 => {
                    let ca = self.f64().unwrap();
                    ca.cumprod(reverse).into_series()
                }
                dt => panic!("cumprod not supported for dtype: {dt:?}"),
            }
        }
        #[cfg(not(feature = "cum_agg"))]
        {
            panic!("activate 'cum_agg' feature")
        }
    }

    /// Get the product of an array.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    pub fn product(&self) -> Series {
        #[cfg(feature = "product")]
        {
            use DataType::*;
            match self.dtype() {
                Boolean => self.cast(&DataType::Int64).unwrap().product(),
                Int8 | UInt8 | Int16 | UInt16 => {
                    let s = self.cast(&Int64).unwrap();
                    s.product()
                }
                Int64 => {
                    let ca = self.i64().unwrap();
                    ca.prod_as_series()
                }
                Float32 => {
                    let ca = self.f32().unwrap();
                    ca.prod_as_series()
                }
                Float64 => {
                    let ca = self.f64().unwrap();
                    ca.prod_as_series()
                }
                dt => panic!("cumprod not supported for dtype: {dt:?}"),
            }
        }
        #[cfg(not(feature = "product"))]
        {
            panic!("activate 'product' feature")
        }
    }

    #[cfg(feature = "rank")]
    pub fn rank(&self, options: RankOptions) -> Series {
        rank(self, options.method, options.descending)
    }

    /// Cast throws an error if conversion had overflows
    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        let null_count = self.null_count();
        let len = self.len();
        if null_count == len {
            return Ok(Series::full_null(self.name(), len, dtype));
        }
        let s = self.0.cast(dtype)?;
        if null_count != s.null_count() {
            let failure_mask = !self.is_null() & s.is_null();
            let failures = self.filter_threaded(&failure_mask, false)?.unique()?;
            Err(PolarsError::ComputeError(
                format!(
                    "Strict conversion from {:?} to {:?} failed for values {}. \
                    If you were trying to cast Utf8 to Date, Time, or Datetime, \
                    consider using `strptime`.",
                    self.dtype(),
                    dtype,
                    failures.fmt_list(),
                )
                .into(),
            ))
        } else {
            Ok(s)
        }
    }

    #[cfg(feature = "dtype-time")]
    pub(crate) fn into_time(self) -> Series {
        #[cfg(not(feature = "dtype-time"))]
        {
            panic!("activate feature dtype-time")
        }
        match self.dtype() {
            DataType::Int64 => self.i64().unwrap().clone().into_time().into_series(),
            DataType::Time => self
                .time()
                .unwrap()
                .as_ref()
                .clone()
                .into_time()
                .into_series(),
            dt => panic!("date not implemented for {dt:?}"),
        }
    }

    pub(crate) fn into_date(self) -> Series {
        #[cfg(not(feature = "dtype-date"))]
        {
            panic!("activate feature dtype-date")
        }
        #[cfg(feature = "dtype-date")]
        match self.dtype() {
            DataType::Int32 => self.i32().unwrap().clone().into_date().into_series(),
            DataType::Date => self
                .date()
                .unwrap()
                .as_ref()
                .clone()
                .into_date()
                .into_series(),
            dt => panic!("date not implemented for {dt:?}"),
        }
    }
    pub(crate) fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> Series {
        #[cfg(not(feature = "dtype-datetime"))]
        {
            panic!("activate feature dtype-datetime")
        }

        #[cfg(feature = "dtype-datetime")]
        match self.dtype() {
            DataType::Int64 => self
                .i64()
                .unwrap()
                .clone()
                .into_datetime(timeunit, tz)
                .into_series(),
            DataType::Datetime(_, _) => self
                .datetime()
                .unwrap()
                .as_ref()
                .clone()
                .into_datetime(timeunit, tz)
                .into_series(),
            dt => panic!("into_datetime not implemented for {dt:?}"),
        }
    }

    pub(crate) fn into_duration(self, timeunit: TimeUnit) -> Series {
        #[cfg(not(feature = "dtype-duration"))]
        {
            panic!("activate feature dtype-duration")
        }
        #[cfg(feature = "dtype-duration")]
        match self.dtype() {
            DataType::Int64 => self
                .i64()
                .unwrap()
                .clone()
                .into_duration(timeunit)
                .into_series(),
            DataType::Duration(_) => self
                .duration()
                .unwrap()
                .as_ref()
                .clone()
                .into_duration(timeunit)
                .into_series(),
            dt => panic!("into_duration not implemented for {dt:?}"),
        }
    }

    #[cfg(feature = "abs")]
    /// convert numerical values to their absolute value
    pub fn abs(&self) -> PolarsResult<Series> {
        let a = self.to_physical_repr();
        use DataType::*;
        let out = match a.dtype() {
            #[cfg(feature = "dtype-i8")]
            Int8 => a.i8().unwrap().abs().into_series(),
            #[cfg(feature = "dtype-i16")]
            Int16 => a.i16().unwrap().abs().into_series(),
            Int32 => a.i32().unwrap().abs().into_series(),
            Int64 => a.i64().unwrap().abs().into_series(),
            UInt8 | UInt16 | UInt32 | UInt64 => self.clone(),
            Float32 => a.f32().unwrap().abs().into_series(),
            Float64 => a.f64().unwrap().abs().into_series(),
            dt => {
                return Err(PolarsError::InvalidOperation(
                    format!("abs not supported for series of type {dt:?}").into(),
                ));
            }
        };
        Ok(out)
    }

    #[cfg(feature = "private")]
    // used for formatting
    pub fn str_value(&self, index: usize) -> PolarsResult<Cow<str>> {
        let out = match self.0.get(index)? {
            AnyValue::Utf8(s) => Cow::Borrowed(s),
            AnyValue::Null => Cow::Borrowed("null"),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(idx, rev, arr) => {
                if arr.is_null() {
                    Cow::Borrowed(rev.get(idx))
                } else {
                    unsafe { Cow::Borrowed(arr.deref_unchecked().value(idx as usize)) }
                }
            }
            av => Cow::Owned(format!("{av}")),
        };
        Ok(out)
    }
    /// Get the head of the Series.
    pub fn head(&self, length: Option<usize>) -> Series {
        match length {
            Some(len) => self.slice(0, std::cmp::min(len, self.len())),
            None => self.slice(0, std::cmp::min(10, self.len())),
        }
    }

    /// Get the tail of the Series.
    pub fn tail(&self, length: Option<usize>) -> Series {
        let len = match length {
            Some(len) => std::cmp::min(len, self.len()),
            None => std::cmp::min(10, self.len()),
        };
        self.slice(-(len as i64), len)
    }

    pub fn mean_as_series(&self) -> Series {
        match self.dtype() {
            DataType::Float32 => {
                let val = &[self.mean().map(|m| m as f32)];
                Series::new(self.name(), val)
            }
            dt if dt.is_numeric() || matches!(dt, DataType::Boolean) => {
                let val = &[self.mean()];
                Series::new(self.name(), val)
            }
            dt @ DataType::Duration(_) => {
                Series::new(self.name(), &[self.mean().map(|v| v as i64)])
                    .cast(dt)
                    .unwrap()
            }
            _ => return Series::full_null(self.name(), 1, self.dtype()),
        }
    }

    /// Compute the unique elements, but maintain order. This requires more work
    /// than a naive [`Series::unique`](SeriesTrait::unique).
    pub fn unique_stable(&self) -> PolarsResult<Series> {
        let idx = self.arg_unique()?;
        // Safety:
        // Indices are in bounds.
        unsafe { self.take_unchecked(&idx) }
    }

    pub fn idx(&self) -> PolarsResult<&IdxCa> {
        #[cfg(feature = "bigidx")]
        {
            self.u64()
        }
        #[cfg(not(feature = "bigidx"))]
        {
            self.u32()
        }
    }

    /// Returns an estimation of the total (heap) allocated size of the `Series` in bytes.
    ///
    /// # Implementation
    /// This estimation is the sum of the size of its buffers, validity, including nested arrays.
    /// Multiple arrays may share buffers and bitmaps. Therefore, the size of 2 arrays is not the
    /// sum of the sizes computed from this function. In particular, [`StructArray`]'s size is an upper bound.
    ///
    /// When an array is sliced, its allocated size remains constant because the buffer unchanged.
    /// However, this function will yield a smaller number. This is because this function returns
    /// the visible size of the buffer, not its total capacity.
    ///
    /// FFI buffers are included in this estimation.
    pub fn estimated_size(&self) -> usize {
        #[allow(unused_mut)]
        let mut size = self
            .chunks()
            .iter()
            .map(|arr| estimated_bytes_size(&**arr))
            .sum();
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(Some(rv)) => match &**rv {
                RevMapping::Local(arr) => size += estimated_bytes_size(arr),
                RevMapping::Global(map, arr, _) => {
                    size +=
                        map.capacity() * std::mem::size_of::<u32>() * 2 + estimated_bytes_size(arr);
                }
            },
            _ => {}
        }

        size
    }
}

impl Deref for Series {
    type Target = dyn SeriesTrait;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl<'a> AsRef<(dyn SeriesTrait + 'a)> for Series {
    fn as_ref(&self) -> &(dyn SeriesTrait + 'a) {
        self.0.as_ref()
    }
}

impl Default for Series {
    fn default() -> Self {
        Int64Chunked::default().into_series()
    }
}

impl<'a, T> AsRef<ChunkedArray<T>> for dyn SeriesTrait + 'a
where
    T: 'static + PolarsDataType,
{
    fn as_ref(&self) -> &ChunkedArray<T> {
        if &T::get_dtype() == self.dtype() ||
            // needed because we want to get ref of List no matter what the inner type is.
            (matches!(T::get_dtype(), DataType::List(_)) && matches!(self.dtype(), DataType::List(_)))
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

impl<'a, T> AsMut<ChunkedArray<T>> for dyn SeriesTrait + 'a
where
    T: 'static + PolarsDataType,
{
    fn as_mut(&mut self) -> &mut ChunkedArray<T> {
        if &T::get_dtype() == self.dtype() ||
            // needed because we want to get ref of List no matter what the inner type is.
            (matches!(T::get_dtype(), DataType::List(_)) && matches!(self.dtype(), DataType::List(_)))
        {
            unsafe { &mut *(self as *mut dyn SeriesTrait as *mut ChunkedArray<T>) }
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
    use std::convert::TryFrom;

    use crate::prelude::*;
    use crate::series::*;

    #[test]
    fn cast() {
        let ar = UInt32Chunked::new("a", &[1, 2]);
        let s = ar.into_series();
        let s2 = s.cast(&DataType::Int64).unwrap();

        assert!(s2.i64().is_ok());
        let s2 = s.cast(&DataType::Float32).unwrap();
        assert!(s2.f32().is_ok());
    }

    #[test]
    fn new_series() {
        let _ = Series::new("boolean series", &vec![true, false, true]);
        let _ = Series::new("int series", &[1, 2, 3]);
        let ca = Int32Chunked::new("a", &[1, 2, 3]);
        let _ = ca.into_series();
    }

    #[test]
    #[cfg(feature = "dtype-struct")]
    fn new_series_from_empty_structs() {
        let dtype = DataType::Struct(vec![]);
        let empties = vec![AnyValue::StructOwned(Box::new((vec![], vec![]))); 3];
        let s = Series::from_any_values_and_dtype("", &empties, &dtype).unwrap();
        assert_eq!(s.len(), 3);
    }
    #[test]
    fn new_series_from_arrow_primitive_array() {
        let array = UInt32Array::from_slice(&[1, 2, 3, 4, 5]);
        let array_ref: ArrayRef = Box::new(array);

        let _ = Series::try_from(("foo", array_ref)).unwrap();
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

        assert_eq!(slice_1.get(0).unwrap(), AnyValue::Int64(3));
        assert_eq!(slice_2.get(0).unwrap(), AnyValue::Int64(1));
        assert_eq!(slice_3.get(0).unwrap(), AnyValue::Int64(1));
    }

    #[test]
    fn out_of_range_slice_does_not_panic() {
        let series = Series::new("a", &[1i64, 2, 3, 4, 5]);

        let _ = series.slice(-3, 4);
        let _ = series.slice(-6, 2);
        let _ = series.slice(4, 2);
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
