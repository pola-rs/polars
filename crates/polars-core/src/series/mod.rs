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
pub mod unstable;

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use ahash::RandomState;
use arrow::compute::aggregate::estimated_bytes_size;
use arrow::offset::Offsets;
pub use from::*;
pub use iterator::{SeriesIter, SeriesPhysIter};
use num_traits::NumCast;
use rayon::prelude::*;
pub use series_trait::{IsSorted, *};

use crate::chunked_array::Settings;
#[cfg(feature = "zip_with")]
use crate::series::arithmetic::coerce_lhs_rhs;
use crate::utils::{_split_offsets, get_casting_failures, split_ca, split_series, Wrap};
use crate::POOL;

/// # Series
/// The columnar data type for a DataFrame.
///
/// Most of the available functions are defined in the [SeriesTrait trait](crate::series::SeriesTrait).
///
/// The `Series` struct consists
/// of typed [ChunkedArray]'s. To quickly cast
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
/// See all the comparison operators in the [CmpOps trait](crate::chunked_array::ops::ChunkCompare)
///
/// ## Iterators
/// The Series variants contain differently typed [ChunkedArray's](crate::chunked_array::ChunkedArray).
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
    /// Create a new empty Series.
    pub fn new_empty(name: &str, dtype: &DataType) -> Series {
        Series::full_null(name, 0, dtype)
    }

    pub fn clear(&self) -> Series {
        // Only the inner of objects know their type, so use this hack.
        #[cfg(feature = "object")]
        if matches!(self.dtype(), DataType::Object(_)) {
            return if self.is_empty() {
                self.clone()
            } else {
                let av = self.get(0).unwrap();
                Series::new(self.name(), [av]).slice(0, 0)
            };
        }
        Series::new_empty(self.name(), self.dtype())
    }

    #[doc(hidden)]
    pub fn _get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
        if Arc::weak_count(&self.0) + Arc::strong_count(&self.0) != 1 {
            self.0 = self.0.clone_inner();
        }
        Arc::get_mut(&mut self.0).expect("implementation error")
    }

    /// # Safety
    /// The caller must ensure the length and the data types of `ArrayRef` does not change.
    /// And that the null_count is updated (e.g. with a `compute_len()`)
    pub unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        #[allow(unused_mut)]
        let mut ca = self._get_inner_mut();
        ca.chunks_mut()
    }

    pub fn is_sorted_flag(&self) -> IsSorted {
        let flags = self.get_flags();
        if flags.contains(Settings::SORTED_DSC) {
            IsSorted::Descending
        } else if flags.contains(Settings::SORTED_ASC) {
            IsSorted::Ascending
        } else {
            IsSorted::Not
        }
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        let mut flags = self.get_flags();
        flags.set_sorted_flag(sorted);
        self.set_flags(flags);
    }

    pub(crate) fn clear_settings(&mut self) {
        self.set_flags(Settings::empty());
    }
    #[allow(dead_code)]
    pub fn get_flags(&self) -> Settings {
        self.0._get_flags()
    }

    pub(crate) fn set_flags(&mut self, flags: Settings) {
        self._get_inner_mut()._set_flags(flags)
    }

    pub fn into_frame(self) -> DataFrame {
        DataFrame::new_no_checks(vec![self])
    }

    /// Rename series.
    pub fn rename(&mut self, name: &str) -> &mut Series {
        self._get_inner_mut().rename(name);
        self
    }

    /// Return this Series with a new name.
    pub fn with_name(mut self, name: &str) -> Series {
        self.rename(name);
        self
    }

    pub fn from_arrow(name: &str, array: ArrayRef) -> PolarsResult<Series> {
        Self::try_from((name, array))
    }

    #[cfg(feature = "arrow_rs")]
    pub fn from_arrow_rs(name: &str, array: &dyn arrow_array::Array) -> PolarsResult<Series> {
        Self::from_arrow(name, array.into())
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

    /// Redo a length and null_count compute
    pub fn compute_len(&mut self) {
        self._get_inner_mut().compute_len()
    }

    /// Extend the memory backed by this array with the values from `other`.
    ///
    /// See [`ChunkedArray::extend`] and [`ChunkedArray::append`].
    pub fn extend(&mut self, other: &Series) -> PolarsResult<&mut Self> {
        self._get_inner_mut().extend(other)?;
        Ok(self)
    }

    pub fn sort(&self, descending: bool) -> Self {
        self.sort_with(SortOptions {
            descending,
            ..Default::default()
        })
    }

    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self._get_inner_mut().as_single_ptr()
    }

    /// Cast `[Series]` to another `[DataType]`.
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        // Best leave as is.
        if matches!(dtype, DataType::Unknown) {
            return Ok(self.clone());
        }
        let ret = self.0.cast(dtype);
        let len = self.len();
        if ret.is_err() && self.null_count() == len {
            return Ok(Series::full_null(self.name(), len, dtype));
        }
        ret
    }

    /// Cast from physical to logical types without any checks on the validity of the cast.
    ///
    /// # Safety
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        match self.dtype() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => self.struct_().unwrap().cast_unchecked(dtype),
            DataType::List(_) => self.list().unwrap().cast_unchecked(dtype),
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let ca: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                        ca.cast_unchecked(dtype)
                })
            },
            DataType::Binary => self.binary().unwrap().cast_unchecked(dtype),
            _ => self.cast(dtype),
        }
    }

    /// Cast numerical types to f64, and keep floats as is.
    pub fn to_float(&self) -> PolarsResult<Series> {
        match self.dtype() {
            DataType::Float32 | DataType::Float64 => Ok(self.clone()),
            _ => self.cast(&DataType::Float64),
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
        let sum = self.sum_as_series().cast(&DataType::Float64).ok()?;
        T::from(sum.f64().unwrap().get(0)?)
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
        let min = self.min_as_series().cast(&DataType::Float64).ok()?;
        T::from(min.f64().unwrap().get(0)?)
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
        let max = self.max_as_series().cast(&DataType::Float64).ok()?;
        T::from(max.f64().unwrap().get(0)?)
    }

    /// Explode a list Series. This expands every item to a new row..
    pub fn explode(&self) -> PolarsResult<Series> {
        match self.dtype() {
            DataType::List(_) => self.list().unwrap().explode(),
            #[cfg(feature = "dtype-array")]
            DataType::Array(_, _) => self.array().unwrap().explode(),
            _ => Ok(self.clone()),
        }
    }

    /// Check if float value is NaN (note this is different than missing/ null)
    pub fn is_nan(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_nan()),
            dt if dt.is_numeric() => Ok(BooleanChunked::full(self.name(), false, self.len())),
            _ => polars_bail!(opq = is_nan, self.dtype()),
        }
    }

    /// Check if float value is NaN (note this is different than missing/ null)
    pub fn is_not_nan(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_not_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_not_nan()),
            dt if dt.is_numeric() => Ok(BooleanChunked::full(self.name(), true, self.len())),
            _ => polars_bail!(opq = is_not_nan, self.dtype()),
        }
    }

    /// Check if numeric value is finite
    pub fn is_finite(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_finite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_finite()),
            dt if dt.is_numeric() => Ok(BooleanChunked::full(self.name(), true, self.len())),
            _ => polars_bail!(opq = is_finite, self.dtype()),
        }
    }

    /// Check if float value is infinite
    pub fn is_infinite(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_infinite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_infinite()),
            dt if dt.is_numeric() => Ok(BooleanChunked::full(self.name(), false, self.len())),
            _ => polars_bail!(opq = is_infinite, self.dtype()),
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
    /// * List(inner) -> List(physical of inner)
    ///
    pub fn to_physical_repr(&self) -> Cow<Series> {
        use DataType::*;
        match self.dtype() {
            Date => Cow::Owned(self.cast(&Int32).unwrap()),
            Datetime(_, _) | Duration(_) | Time => Cow::Owned(self.cast(&Int64).unwrap()),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => Cow::Owned(self.cast(&UInt32).unwrap()),
            List(inner) => Cow::Owned(self.cast(&List(Box::new(inner.to_physical()))).unwrap()),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let arr = self.struct_().unwrap();
                let fields: Vec<_> = arr
                    .fields()
                    .iter()
                    .map(|s| s.to_physical_repr().into_owned())
                    .collect();
                let ca = StructChunked::new(self.name(), &fields).unwrap();
                Cow::Owned(ca.into_series())
            },
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

    // Take a function pointer to reduce bloat.
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
    pub unsafe fn take_unchecked_from_slice(&self, idx: &[IdxSize]) -> Series {
        self.take_slice_unchecked(idx)
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_unchecked_threaded(&self, idx: &IdxCa, rechunk: bool) -> Series {
        self.threaded_op(rechunk, idx.len(), &|offset, len| {
            let idx = idx.slice(offset as i64, len);
            Ok(self.take_unchecked(&idx))
        })
        .unwrap()
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_slice_unchecked_threaded(&self, idx: &[IdxSize], rechunk: bool) -> Series {
        self.threaded_op(rechunk, idx.len(), &|offset, len| {
            Ok(self.take_slice_unchecked(&idx[offset..offset + len]))
        })
        .unwrap()
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

    /// Traverse and collect every nth element in a new array.
    pub fn take_every(&self, n: usize) -> Series {
        let idx = (0..self.len() as IdxSize).step_by(n).collect_ca("");
        // SAFETY: we stay in-bounds.
        unsafe { self.take_unchecked(&idx) }
    }

    /// Filter by boolean mask. This operation clones data.
    pub fn filter_threaded(&self, filter: &BooleanChunked, rechunk: bool) -> PolarsResult<Series> {
        // This would fail if there is a broadcasting filter, because we cannot
        // split that filter over threads besides they are a no-op, so we do the
        // standard filter.
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
        if self.is_empty()
            && (self.dtype().is_numeric() || matches!(self.dtype(), DataType::Boolean))
        {
            let zero = Series::new(self.name(), [0]);
            return zero.cast(self.dtype()).unwrap().sum_as_series();
        }
        match self.dtype() {
            Int8 | UInt8 | Int16 | UInt16 => self.cast(&Int64).unwrap().sum_as_series(),
            _ => self._sum_as_series(),
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
                Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 => {
                    let s = self.cast(&Int64).unwrap();
                    s.product()
                },
                Int64 => self.i64().unwrap().prod_as_series(),
                UInt64 => self.u64().unwrap().prod_as_series(),
                Float32 => self.f32().unwrap().prod_as_series(),
                Float64 => self.f64().unwrap().prod_as_series(),
                dt => panic!("product not supported for dtype: {dt:?}"),
            }
        }
        #[cfg(not(feature = "product"))]
        {
            panic!("activate 'product' feature")
        }
    }

    /// Cast throws an error if conversion had overflows
    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        let null_count = self.null_count();
        let len = self.len();

        match self.dtype() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => {},
            _ => {
                if null_count == len {
                    return Ok(Series::full_null(self.name(), len, dtype));
                }
            },
        }
        let s = self.0.cast(dtype)?;
        if null_count != s.null_count() {
            let failures = get_casting_failures(self, &s)?;
            polars_bail!(
                ComputeError:
                "strict conversion from `{}` to `{}` failed for column: {}, value(s) {}; \
                if you were trying to cast Utf8 to temporal dtypes, consider using `strptime`",
                self.dtype(), dtype, s.name(), failures.fmt_list(),
            );
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
            },
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
            },
            dt if dt.is_numeric() || matches!(dt, DataType::Boolean) => {
                let val = &[self.mean()];
                Series::new(self.name(), val)
            },
            dt @ DataType::Duration(_) => {
                Series::new(self.name(), &[self.mean().map(|v| v as i64)])
                    .cast(dt)
                    .unwrap()
            },
            _ => return Series::full_null(self.name(), 1, self.dtype()),
        }
    }

    /// Compute the unique elements, but maintain order. This requires more work
    /// than a naive [`Series::unique`](SeriesTrait::unique).
    pub fn unique_stable(&self) -> PolarsResult<Series> {
        let idx = self.arg_unique()?;
        // SAFETY: Indices are in bounds.
        unsafe { Ok(self.take_unchecked(&idx)) }
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
                },
            },
            _ => {},
        }

        size
    }

    /// Packs every element into a list.
    pub fn as_list(&self) -> ListChunked {
        let s = self.rechunk();
        // don't  use `to_arrow` as we need the physical types
        let values = s.chunks()[0].clone();
        let offsets = (0i64..(s.len() as i64 + 1)).collect::<Vec<_>>();
        let offsets = unsafe { Offsets::new_unchecked(offsets) };

        let new_arr = LargeListArray::new(
            DataType::List(Box::new(s.dtype().clone())).to_arrow(),
            offsets.into(),
            values,
            None,
        );
        ListChunked::with_chunk(s.name(), new_arr)
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
        match T::get_dtype() {
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(None, None) => panic!("impl error"),
            _ => {
                if &T::get_dtype() == self.dtype() ||
                    // Needed because we want to get ref of List no matter what the inner type is.
                    (matches!(T::get_dtype(), DataType::List(_)) && matches!(self.dtype(), DataType::List(_)))
                {
                    unsafe { &*(self as *const dyn SeriesTrait as *const ChunkedArray<T>) }
                } else {
                    panic!(
                        "implementation error, cannot get ref {:?} from {:?}",
                        T::get_dtype(),
                        self.dtype()
                    );
                }
            },
        }
    }
}

impl<'a, T> AsMut<ChunkedArray<T>> for dyn SeriesTrait + 'a
where
    T: 'static + PolarsDataType,
{
    fn as_mut(&mut self) -> &mut ChunkedArray<T> {
        if &T::get_dtype() == self.dtype() ||
            // Needed because we want to get ref of List no matter what the inner type is.
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
        let s = Series::from_any_values_and_dtype("", &empties, &dtype, false).unwrap();
        assert_eq!(s.len(), 3);
    }
    #[test]
    fn new_series_from_arrow_primitive_array() {
        let array = UInt32Array::from_slice([1, 2, 3, 4, 5]);
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
}
