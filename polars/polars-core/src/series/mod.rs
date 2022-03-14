//! Type agnostic columnar data structure.
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::array::ArrayRef;
use polars_arrow::prelude::QuantileInterpolOptions;
#[cfg(any(feature = "dtype-struct", feature = "object"))]
use std::any::Any;

#[cfg(feature = "series_from_anyvalue")]
mod any_value;
pub(crate) mod arithmetic;
mod comparison;
mod from;
pub mod implementations;
mod into;
pub(crate) mod iterator;
pub mod ops;
mod series_trait;
#[cfg(feature = "private")]
pub mod unstable;

use crate::chunked_array::ops::rolling_window::RollingOptions;
#[cfg(feature = "rank")]
use crate::prelude::unique::rank::rank;
use crate::utils::Wrap;
use crate::utils::{split_ca, split_series};
use crate::{series::arithmetic::coerce_lhs_rhs, POOL};
use ahash::RandomState;
pub use from::*;
use num::NumCast;
use rayon::prelude::*;
pub use series_trait::*;
use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

pub use series_trait::IsSorted;

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
/// let mask = s.equal(1);
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
        let h = UInt64Chunked::from_vec("", self.0.vec_hash(rs)).sum();
        h.hash(state)
    }
}

impl Series {
    /// Create a new empty Series
    pub fn new_empty(name: &str, dtype: &DataType) -> Series {
        Series::full_null(name, 0, dtype)
    }

    pub(crate) fn get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
        if Arc::weak_count(&self.0) + Arc::strong_count(&self.0) != 1 {
            self.0 = self.0.clone_inner();
        }
        Arc::get_mut(&mut self.0).expect("implementation error")
    }

    pub fn into_frame(self) -> DataFrame {
        DataFrame::new_no_checks(vec![self])
    }

    /// Rename series.
    pub fn rename(&mut self, name: &str) -> &mut Series {
        self.get_inner_mut().rename(name);
        self
    }

    /// Shrink the capacity of this array to fit it's length.
    pub fn shrink_to_fit(&mut self) {
        self.get_inner_mut().shrink_to_fit()
    }

    /// Append arrow array of same datatype.
    pub fn append_array(&mut self, other: ArrayRef) -> Result<&mut Self> {
        self.get_inner_mut().append_array(other)?;
        Ok(self)
    }

    /// Append in place. This is done by adding the chunks of `other` to this [`Series`].
    ///
    /// See [`ChunkedArray::append`] and [`ChunkedArray::extend`].
    pub fn append(&mut self, other: &Series) -> Result<&mut Self> {
        self.get_inner_mut().append(other)?;
        Ok(self)
    }

    /// Extend the memory backed by this array with the values from `other`.
    ///
    /// See [`ChunkedArray::extend`] and [`ChunkedArray::append`].
    pub fn extend(&mut self, other: &Series) -> Result<&mut Self> {
        self.get_inner_mut().extend(other)?;
        Ok(self)
    }

    pub fn sort(&self, reverse: bool) -> Self {
        self.sort_with(SortOptions {
            descending: reverse,
            ..Default::default()
        })
    }

    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> Result<usize> {
        self.get_inner_mut().as_single_ptr()
    }

    /// Cast `[Series]` to another `[DataType]`
    pub fn cast(&self, dtype: &DataType) -> Result<Self> {
        self.0.cast(dtype)
    }

    /// Compute the sum of all values in this Series.
    /// Returns `None` if the array is empty or only contains null values.
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
    pub fn explode(&self) -> Result<Series> {
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
    pub fn is_nan(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_nan()),
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
        }
    }

    /// Check if float value is NaN (note this is different than missing/ null)
    pub fn is_not_nan(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_not_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_not_nan()),
            _ => Ok(BooleanChunked::full(self.name(), true, self.len())),
        }
    }

    /// Check if float value is finite
    pub fn is_finite(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_finite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_finite()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "is_nan not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Check if float value is finite
    pub fn is_infinite(&self) -> Result<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_infinite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_infinite()),
            _ => Err(PolarsError::InvalidOperation(
                format!(
                    "is_nan not supported for series with dtype {:?}",
                    self.dtype()
                )
                .into(),
            )),
        }
    }

    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`
    #[cfg(feature = "zip_with")]
    #[cfg_attr(docsrs, doc(cfg(feature = "zip_with")))]
    pub fn zip_with(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
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

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_unchecked_threaded(&self, idx: &IdxCa, rechunk: bool) -> Result<Series> {
        let n_threads = POOL.current_num_threads();
        let idx = split_ca(idx, n_threads)?;

        let series: Result<Vec<_>> =
            POOL.install(|| idx.par_iter().map(|idx| self.take_unchecked(idx)).collect());

        let s = series?
            .into_iter()
            .reduce(|mut s, s1| {
                s.append(&s1).unwrap();
                s
            })
            .unwrap();
        if rechunk {
            Ok(s.rechunk())
        } else {
            Ok(s)
        }
    }

    /// Take by index. This operation is clone.
    ///
    /// # Safety
    ///
    /// Out of bounds access doesn't Error but will return a Null value
    pub fn take_threaded(&self, idx: &IdxCa, rechunk: bool) -> Result<Series> {
        let n_threads = POOL.current_num_threads();
        let idx = split_ca(idx, n_threads).unwrap();

        let series = POOL.install(|| {
            idx.par_iter()
                .map(|idx| self.take(idx))
                .collect::<Result<Vec<_>>>()
        })?;

        let s = series
            .into_iter()
            .reduce(|mut s, s1| {
                s.append(&s1).unwrap();
                s
            })
            .unwrap();
        if rechunk {
            Ok(s.rechunk())
        } else {
            Ok(s)
        }
    }

    /// Filter by boolean mask. This operation clones data.
    pub fn filter_threaded(&self, filter: &BooleanChunked, rechunk: bool) -> Result<Series> {
        // this would fail if there is a broadcasting filter.
        // because we cannot split that filter over threads
        // besides they are a no-op, so we do the standard filter.
        if filter.len() == 1 {
            return self.filter(filter);
        }
        let n_threads = POOL.current_num_threads();
        let filters = split_ca(filter, n_threads).unwrap();
        let series = split_series(self, n_threads).unwrap();

        let series: Result<Vec<_>> = POOL.install(|| {
            filters
                .par_iter()
                .zip(series)
                .map(|(filter, s)| s.filter(filter))
                .collect()
        });

        let s = series?
            .into_iter()
            .reduce(|mut s, s1| {
                s.append(&s1).unwrap();
                s
            })
            .unwrap();
        if rechunk {
            Ok(s.rechunk())
        } else {
            Ok(s)
        }
    }

    #[cfg(feature = "dot_product")]
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_product")))]
    pub fn dot(&self, other: &Series) -> Option<f64> {
        (self * other).sum::<f64>()
    }

    #[cfg(feature = "row_hash")]
    #[cfg_attr(docsrs, doc(cfg(feature = "row_hash")))]
    /// Get a hash of this Series
    pub fn hash(&self, build_hasher: ahash::RandomState) -> UInt64Chunked {
        UInt64Chunked::from_vec(self.name(), self.0.vec_hash(build_hasher))
    }

    /// Get the sum of the Series as a new Series of length 1.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    pub fn sum_as_series(&self) -> Series {
        use DataType::*;
        match self.dtype() {
            Int8 | UInt8 | Int16 | UInt16 => self.cast(&Int64).unwrap().sum_as_series(),
            _ => self._sum_as_series(),
        }
    }

    /// Get an array with the cumulative max computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
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
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
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
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
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
                dt => panic!("cumsum not supported for dtype: {:?}", dt),
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
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
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
                dt => panic!("cumprod not supported for dtype: {:?}", dt),
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
    #[cfg_attr(docsrs, doc(cfg(feature = "product")))]
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
                dt => panic!("cumprod not supported for dtype: {:?}", dt),
            }
        }
        #[cfg(not(feature = "product"))]
        {
            panic!("activate 'product' feature")
        }
    }

    /// Apply a rolling variance to a Series. See:
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_var(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_var(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }

    /// Apply a rolling std to a Series. See:
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_std(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_std(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }

    /// Apply a rolling mean to a Series. See:
    /// [ChunkedArray::rolling_mean]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_mean(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_mean(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }
    /// Apply a rolling sum to a Series. See:
    /// [ChunkedArray::rolling_sum]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_sum(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_sum(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }
    /// Apply a rolling median to a Series. See:
    /// [`ChunkedArray::rolling_median`]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_median(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_median(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }
    /// Apply a rolling quantile to a Series. See:
    /// [`ChunkedArray::rolling_quantile`]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_quantile(
        &self,
        _quantile: f64,
        _interpolation: QuantileInterpolOptions,
        _options: RollingOptions,
    ) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_quantile(_quantile, _interpolation, _options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }
    /// Apply a rolling min to a Series. See:
    /// [ChunkedArray::rolling_min]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_min(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_min(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }
    /// Apply a rolling max to a Series. See:
    /// [ChunkedArray::rolling_max]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    pub fn rolling_max(&self, _options: RollingOptions) -> Result<Series> {
        #[cfg(feature = "rolling_window")]
        {
            self._rolling_max(_options)
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            panic!("activate 'rolling_window' feature")
        }
    }

    #[cfg(feature = "rank")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rank")))]
    pub fn rank(&self, options: RankOptions) -> Series {
        rank(self, options.method, options.descending)
    }

    /// Cast throws an error if conversion had overflows
    pub fn strict_cast(&self, data_type: &DataType) -> Result<Series> {
        let s = self.cast(data_type)?;
        if self.null_count() != s.null_count() {
            Err(PolarsError::ComputeError(
                format!(
                    "strict conversion of cast from {:?} to {:?} failed. consider non-strict cast.\n
                    If you were trying to cast Utf8 to Date,Time,Datetime, consider using `strptime`",
                    self.dtype(),
                    data_type
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
            dt => panic!("date not implemented for {:?}", dt),
        }
    }

    pub(crate) fn into_date(self) -> Series {
        #[cfg(not(feature = "dtype-date"))]
        {
            panic!("activate feature dtype-date")
        }
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Int32 => self.i32().unwrap().clone().into_date().into_series(),
            #[cfg(feature = "dtype-date")]
            DataType::Date => self
                .date()
                .unwrap()
                .as_ref()
                .clone()
                .into_date()
                .into_series(),
            dt => panic!("date not implemented for {:?}", dt),
        }
    }
    pub(crate) fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> Series {
        #[cfg(not(feature = "dtype-datetime"))]
        {
            panic!("activate feature dtype-datetime")
        }

        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Int64 => self
                .i64()
                .unwrap()
                .clone()
                .into_datetime(timeunit, tz)
                .into_series(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self
                .datetime()
                .unwrap()
                .as_ref()
                .clone()
                .into_datetime(timeunit, tz)
                .into_series(),
            dt => panic!("into_datetime not implemented for {:?}", dt),
        }
    }

    pub(crate) fn into_duration(self, timeunit: TimeUnit) -> Series {
        #[cfg(not(feature = "dtype-duration"))]
        {
            panic!("activate feature dtype-duration")
        }
        match self.dtype() {
            #[cfg(feature = "dtype-duration")]
            DataType::Int64 => self
                .i64()
                .unwrap()
                .clone()
                .into_duration(timeunit)
                .into_series(),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => self
                .duration()
                .unwrap()
                .as_ref()
                .clone()
                .into_duration(timeunit)
                .into_series(),
            dt => panic!("into_duration not implemented for {:?}", dt),
        }
    }

    /// Check if the underlying data is a logical type.
    pub fn is_logical(&self) -> bool {
        self.dtype().is_logical()
    }

    /// Check if underlying physical data is numeric.
    ///
    /// Date types and Categoricals are also considered numeric.
    pub fn is_numeric_physical(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        #[allow(clippy::match_like_matches_macro)]
        match self.dtype() {
            DataType::Utf8 | DataType::List(_) | DataType::Boolean | DataType::Null => false,
            #[cfg(feature = "object")]
            DataType::Object(_) => false,
            _ => true,
        }
    }

    #[cfg(feature = "abs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "abs")))]
    /// convert numerical values to their absolute value
    pub fn abs(&self) -> Result<Series> {
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
                    format!("abs not supportedd for series of type {:?}", dt).into(),
                ));
            }
        };
        Ok(out)
    }

    #[cfg(feature = "private")]
    // used for formatting
    pub fn str_value(&self, index: usize) -> Cow<str> {
        match self.0.get(index) {
            AnyValue::Utf8(s) => Cow::Borrowed(s),
            AnyValue::Null => Cow::Borrowed("null"),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(idx, rev) => Cow::Borrowed(rev.get(idx)),
            av => Cow::Owned(format!("{}", av)),
        }
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
        let val = [self.mean()];
        let s = Series::new(self.name(), val);
        if !self.dtype().is_numeric() {
            s.cast(self.dtype()).unwrap()
        } else {
            s
        }
    }

    /// Compute the unique elements, but maintain order. This requires more work
    /// than a naive [`Series::unique`](SeriesTrait::unique).
    pub fn unique_stable(&self) -> Result<Series> {
        let idx = self.arg_unique()?;
        // Safety:
        // Indices are in bounds.
        unsafe { self.take_unchecked(&idx) }
    }

    pub fn idx(&self) -> Result<&IdxCa> {
        #[cfg(feature = "bigidx")]
        {
            self.u64()
        }
        #[cfg(not(feature = "bigidx"))]
        {
            self.u32()
        }
    }

    /// Unpack to ChunkedArray of dtype struct
    #[cfg(feature = "dtype-struct")]
    pub fn struct_(&self) -> Result<&StructChunked> {
        #[cfg(feature = "dtype-struct")]
        match self.dtype() {
            DataType::Struct(_) => {
                let any = self.as_any();

                debug_assert!(any.is::<StructChunked>());
                // Safety
                // We just checked type
                Ok(unsafe { &*(any as *const dyn Any as *const StructChunked) })
            }
            _ => Err(PolarsError::SchemaMisMatch(
                format!("Series dtype {:?} != struct", self.dtype()).into(),
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
    use crate::prelude::*;
    use crate::series::*;
    use std::convert::TryFrom;

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
    fn new_series_from_arrow_primitive_array() {
        let array = UInt32Array::from_slice(&[1, 2, 3, 4, 5]);
        let array_ref: ArrayRef = Arc::new(array);

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

        assert_eq!(slice_1.get(0), AnyValue::Int64(3));
        assert_eq!(slice_2.get(0), AnyValue::Int64(1));
        assert_eq!(slice_3.get(0), AnyValue::Int64(1));
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
