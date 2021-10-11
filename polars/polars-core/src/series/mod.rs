//! Type agnostic columnar data structure.
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::array::ArrayRef;
pub(crate) mod arithmetic;
mod comparison;
mod from;
pub mod implementations;
mod into;
pub(crate) mod iterator;
pub mod ops;
mod series_trait;

use crate::chunked_array::ops::rolling_window::RollingOptions;
#[cfg(feature = "rank")]
use crate::prelude::unique::rank::{rank, RankMethod};
#[cfg(feature = "groupby_list")]
use crate::utils::Wrap;
use crate::utils::{split_ca, split_series};
use crate::{series::arithmetic::coerce_lhs_rhs, POOL};
#[cfg(feature = "groupby_list")]
use ahash::RandomState;
pub use from::*;
use num::NumCast;
use rayon::prelude::*;
pub use series_trait::*;
#[cfg(feature = "groupby_list")]
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// # Series
/// The columnar data type for a DataFrame.
///
/// Most of the available functions are definedin the [SeriesTrait trait](crate::series::SeriesTrait).
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
/// let s: Series = [1, 2, 3].iter().collect();
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
/// use itertools::Itertools;
/// let s = Series::new("dollars", &[1, 2, 3]);
/// let mask = s.eq(1);
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
/// // Series van be created from Vec's, slices and arrays
/// Series::new("boolean series", &vec![true, false, true]);
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
pub struct Series(pub Arc<dyn SeriesTrait>);

#[cfg(feature = "groupby_list")]
impl PartialEq for Wrap<Series> {
    fn eq(&self, other: &Self) -> bool {
        self.0.series_equal_missing(other)
    }
}

#[cfg(feature = "groupby_list")]
impl Eq for Wrap<Series> {}

#[cfg(feature = "groupby_list")]
impl Hash for Wrap<Series> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let rs = RandomState::with_seeds(0, 0, 0, 0);
        let h = UInt64Chunked::new_from_aligned_vec("", self.0.vec_hash(rs)).sum();
        h.hash(state)
    }
}

impl Series {
    pub(crate) fn get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
        if Arc::weak_count(&self.0) + Arc::strong_count(&self.0) != 1 {
            self.0 = self.0.clone_inner();
        }
        Arc::get_mut(&mut self.0).expect("implementation error")
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

    /// Append a Series of the same type in place.
    pub fn append(&mut self, other: &Series) -> Result<&mut Self> {
        self.get_inner_mut().append(other)?;
        Ok(self)
    }

    /// Sort in place.
    pub fn sort_in_place(&mut self, reverse: bool) -> &mut Self {
        self.get_inner_mut().sort_in_place(reverse);
        self
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> Result<usize> {
        self.get_inner_mut().as_single_ptr()
    }

    /// Cast `[Series]` to another `[DataType]`
    pub fn cast(&self, dtype: &DataType) -> Result<Self> {
        self.0.cast(dtype)
    }
    /// Returns `None` if the array is empty or only contains null values.
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("days", [1, 2, 3].as_ref());
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
    ///
    pub fn to_physical_repr(&self) -> Series {
        use DataType::*;
        let out = match self.dtype() {
            Date => self.cast(&DataType::Int32),
            Datetime => self.cast(&DataType::Int64),
            _ => return self.clone(),
        };
        out.unwrap()
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds. Null validity is checked.
    pub unsafe fn take_unchecked_threaded(
        &self,
        idx: &UInt32Chunked,
        rechunk: bool,
    ) -> Result<Series> {
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
    pub fn take_threaded(&self, idx: &UInt32Chunked, rechunk: bool) -> Result<Series> {
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

    /// Round underlying floating point array to given decimal.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn round(&self, decimals: u32) -> Result<Self> {
        use num::traits::Pow;
        if let Ok(ca) = self.f32() {
            let multiplier = 10.0.pow(decimals as f32) as f32;
            let s = ca
                .apply(|val| (val * multiplier).round() / multiplier)
                .into_series();
            return Ok(s);
        }
        if let Ok(ca) = self.f64() {
            let multiplier = 10.0.pow(decimals as f32) as f64;
            let s = ca
                .apply(|val| (val * multiplier).round() / multiplier)
                .into_series();
            return Ok(s);
        }
        Err(PolarsError::DataTypeMisMatch(
            format!("{:?} is not a floating point datatype", self.dtype()).into(),
        ))
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
        UInt64Chunked::new_from_aligned_vec(self.name(), self.0.vec_hash(build_hasher))
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
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cumsum(&self, _reverse: bool) -> Series {
        #[cfg(feature = "cum_agg")]
        {
            match self.dtype() {
                DataType::Boolean => self.cast(&DataType::UInt32).unwrap()._cumsum(_reverse),
                _ => self._cumsum(_reverse),
            }
        }
        #[cfg(not(feature = "cum_agg"))]
        {
            panic!("activate 'cum_agg' feature")
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
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_mean).
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
    /// [ChunkedArray::rolling_sum](crate::prelude::ChunkWindow::rolling_sum).
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
    /// Apply a rolling min to a Series. See:
    /// [ChunkedArray::rolling_min](crate::prelude::ChunkWindow::rolling_min).
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
    /// [ChunkedArray::rolling_max](crate::prelude::ChunkWindow::rolling_max).
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
    pub fn rank(&self, method: RankMethod) -> Series {
        rank(self, method)
    }

    /// Cast throws an error if conversion had overflows
    pub fn strict_cast(&self, data_type: &DataType) -> Result<Series> {
        let s = self.cast(data_type)?;
        if self.null_count() != s.null_count() {
            Err(PolarsError::ComputeError(
                format!(
                    "strict conversion of cast from {:?} to {:?} failed. consider non-strict cast.",
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
        self.i64()
            .expect("impl error")
            .clone()
            .into_time()
            .into_series()
    }

    pub(crate) fn into_date(self) -> Series {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Int32 => self.i32().unwrap().clone().into_date().into_series(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Int64 => self.i64().unwrap().clone().into_date().into_series(),
            _ => unreachable!(),
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
            (matches!(T::get_dtype(), DataType::List(_)) && matches!(self.dtype(), DataType::List(_)) )
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

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::series::*;
    use std::convert::TryFrom;

    #[test]
    fn cast() {
        let ar = UInt32Chunked::new_from_slice("a", &[1, 2]);
        let s = ar.into_series();
        let s2 = s.cast(&DataType::Int64).unwrap();

        assert!(s2.i64().is_ok());
        let s2 = s.cast(&DataType::Float32).unwrap();
        assert!(s2.f32().is_ok());
    }

    #[test]
    fn new_series() {
        Series::new("boolean series", &vec![true, false, true]);
        Series::new("int series", &[1, 2, 3]);
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        ca.into_series();
    }

    #[test]
    fn new_series_from_arrow_primitive_array() {
        let array = UInt32Array::from_slice(&[1, 2, 3, 4, 5]);
        let array_ref: ArrayRef = Arc::new(array);

        Series::try_from(("foo", array_ref)).unwrap();
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

        series.slice(-3, 4);
        series.slice(-6, 2);
        series.slice(4, 2);
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
