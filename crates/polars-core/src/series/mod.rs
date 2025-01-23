//! Type agnostic columnar data structure.
use crate::chunked_array::flags::StatisticsFlags;
pub use crate::prelude::ChunkCompareEq;
use crate::prelude::*;
use crate::{HEAD_DEFAULT_LENGTH, TAIL_DEFAULT_LENGTH};

macro_rules! invalid_operation_panic {
    ($op:ident, $s:expr) => {
        panic!(
            "`{}` operation not supported for dtype `{}`",
            stringify!($op),
            $s._dtype()
        )
    };
}

pub mod amortized_iter;
mod any_value;
pub mod arithmetic;
mod comparison;
mod from;
pub mod implementations;
mod into;
pub(crate) mod iterator;
pub mod ops;
mod series_trait;

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use arrow::compute::aggregate::estimated_bytes_size;
use arrow::offset::Offsets;
pub use from::*;
pub use iterator::{SeriesIter, SeriesPhysIter};
use num_traits::NumCast;
use polars_error::feature_gated;
pub use series_trait::{IsSorted, *};

use crate::chunked_array::cast::CastOptions;
#[cfg(feature = "zip_with")]
use crate::series::arithmetic::coerce_lhs_rhs;
use crate::utils::{handle_casting_failures, materialize_dyn_int, Wrap};
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
/// let s = Series::new("a".into(), [1 , 2, 3]);
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
/// let s = Series::new("dollars".into(), &[1, 2, 3]);
/// let mask = s.equal(1).unwrap();
/// let valid = [true, false, false].iter();
/// assert!(mask
///     .into_iter()
///     .map(|opt_bool| opt_bool.unwrap()) // option, because series can be null
///     .zip(valid)
///     .all(|(a, b)| a == *b))
/// ```
///
/// See all the comparison operators in the [ChunkCompareEq trait](crate::chunked_array::ops::ChunkCompareEq) and
/// [ChunkCompareIneq trait](crate::chunked_array::ops::ChunkCompareIneq).
///
/// ## Iterators
/// The Series variants contain differently typed [ChunkedArray](crate::chunked_array::ChunkedArray)s.
/// These structs can be turned into iterators, making it possible to use any function/ closure you want
/// on a Series.
///
/// These iterators return an `Option<T>` because the values of a series may be null.
///
/// ```
/// use polars_core::prelude::*;
/// let pi = 3.14;
/// let s = Series::new("angle".into(), [2f32 * pi, pi, 1.5 * pi].as_ref());
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
/// Series::new("boolean series".into(), &[true, false, true]);
/// Series::new("int series".into(), &[1, 2, 3]);
/// // And can be nullable
/// Series::new("got nulls".into(), &[Some(1), None, Some(2)]);
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
        self.0.equals_missing(other)
    }
}

impl Eq for Wrap<Series> {}

impl Hash for Wrap<Series> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let rs = PlRandomState::with_seeds(0, 0, 0, 0);
        let mut h = vec![];
        if self.0.vec_hash(rs, &mut h).is_ok() {
            let h = h.into_iter().fold(0, |a: u64, b| a.wrapping_add(b));
            h.hash(state)
        } else {
            self.len().hash(state);
            self.null_count().hash(state);
            self.dtype().hash(state);
        }
    }
}

impl Series {
    /// Create a new empty Series.
    pub fn new_empty(name: PlSmallStr, dtype: &DataType) -> Series {
        Series::full_null(name, 0, dtype)
    }

    pub fn clear(&self) -> Series {
        if self.is_empty() {
            self.clone()
        } else {
            match self.dtype() {
                #[cfg(feature = "object")]
                DataType::Object(_, _) => self
                    .take(&ChunkedArray::<IdxType>::new_vec(PlSmallStr::EMPTY, vec![]))
                    .unwrap(),
                dt => Series::new_empty(self.name().clone(), dt),
            }
        }
    }

    #[doc(hidden)]
    pub fn _get_inner_mut(&mut self) -> &mut dyn SeriesTrait {
        if Arc::weak_count(&self.0) + Arc::strong_count(&self.0) != 1 {
            self.0 = self.0.clone_inner();
        }
        Arc::get_mut(&mut self.0).expect("implementation error")
    }

    /// Take or clone a owned copy of the inner [`ChunkedArray`].
    pub fn take_inner<T>(self) -> ChunkedArray<T>
    where
        T: 'static + PolarsDataType<IsLogical = FalseT>,
    {
        let arc_any = self.0.as_arc_any();
        let downcast = arc_any
            .downcast::<implementations::SeriesWrap<ChunkedArray<T>>>()
            .unwrap();

        match Arc::try_unwrap(downcast) {
            Ok(ca) => ca.0,
            Err(ca) => ca.as_ref().as_ref().clone(),
        }
    }

    /// # Safety
    /// The caller must ensure the length and the data types of `ArrayRef` does not change.
    /// And that the null_count is updated (e.g. with a `compute_len()`)
    pub unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        #[allow(unused_mut)]
        let mut ca = self._get_inner_mut();
        ca.chunks_mut()
    }

    pub fn into_chunks(mut self) -> Vec<ArrayRef> {
        let ca = self._get_inner_mut();
        let chunks = std::mem::take(unsafe { ca.chunks_mut() });
        ca.compute_len();
        chunks
    }

    // TODO! this probably can now be removed, now we don't have special case for structs.
    pub fn select_chunk(&self, i: usize) -> Self {
        let mut new = self.clear();
        let mut flags = self.get_flags();

        use StatisticsFlags as F;
        flags &= F::IS_SORTED_ANY | F::CAN_FAST_EXPLODE_LIST;

        // Assign mut so we go through arc only once.
        let mut_new = new._get_inner_mut();
        let chunks = unsafe { mut_new.chunks_mut() };
        let chunk = self.chunks()[i].clone();
        chunks.clear();
        chunks.push(chunk);
        mut_new.compute_len();
        mut_new._set_flags(flags);
        new
    }

    pub fn is_sorted_flag(&self) -> IsSorted {
        if self.len() <= 1 {
            return IsSorted::Ascending;
        }
        self.get_flags().is_sorted()
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        let mut flags = self.get_flags();
        flags.set_sorted(sorted);
        self.set_flags(flags);
    }

    pub(crate) fn clear_flags(&mut self) {
        self.set_flags(StatisticsFlags::empty());
    }
    pub fn get_flags(&self) -> StatisticsFlags {
        self.0._get_flags()
    }

    pub(crate) fn set_flags(&mut self, flags: StatisticsFlags) {
        self._get_inner_mut()._set_flags(flags)
    }

    pub fn into_frame(self) -> DataFrame {
        // SAFETY: A single-column dataframe cannot have length mismatches or duplicate names
        unsafe { DataFrame::new_no_checks(self.len(), vec![self.into()]) }
    }

    /// Rename series.
    pub fn rename(&mut self, name: PlSmallStr) -> &mut Series {
        self._get_inner_mut().rename(name);
        self
    }

    /// Return this Series with a new name.
    pub fn with_name(mut self, name: PlSmallStr) -> Series {
        self.rename(name);
        self
    }

    pub fn from_arrow_chunks(name: PlSmallStr, arrays: Vec<ArrayRef>) -> PolarsResult<Series> {
        Self::try_from((name, arrays))
    }

    pub fn from_arrow(name: PlSmallStr, array: ArrayRef) -> PolarsResult<Series> {
        Self::try_from((name, array))
    }

    /// Shrink the capacity of this array to fit its length.
    pub fn shrink_to_fit(&mut self) {
        self._get_inner_mut().shrink_to_fit()
    }

    /// Append in place. This is done by adding the chunks of `other` to this [`Series`].
    ///
    /// See [`ChunkedArray::append`] and [`ChunkedArray::extend`].
    pub fn append(&mut self, other: &Series) -> PolarsResult<&mut Self> {
        let must_cast = other.dtype().matches_schema_type(self.dtype())?;
        if must_cast {
            let other = other.cast(self.dtype())?;
            self.append_owned(other)?;
        } else {
            self._get_inner_mut().append(other)?;
        }
        Ok(self)
    }

    /// Append in place. This is done by adding the chunks of `other` to this [`Series`].
    ///
    /// See [`ChunkedArray::append_owned`] and [`ChunkedArray::extend`].
    pub fn append_owned(&mut self, other: Series) -> PolarsResult<&mut Self> {
        let must_cast = other.dtype().matches_schema_type(self.dtype())?;
        if must_cast {
            let other = other.cast(self.dtype())?;
            self._get_inner_mut().append_owned(other)?;
        } else {
            self._get_inner_mut().append_owned(other)?;
        }
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
        let must_cast = other.dtype().matches_schema_type(self.dtype())?;
        if must_cast {
            let other = other.cast(self.dtype())?;
            self._get_inner_mut().extend(&other)?;
        } else {
            self._get_inner_mut().extend(other)?;
        }
        Ok(self)
    }

    /// Sort the series with specific options.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// # fn main() -> PolarsResult<()> {
    /// let s = Series::new("foo".into(), [2, 1, 3]);
    /// let sorted = s.sort(SortOptions::default())?;
    /// assert_eq!(sorted, Series::new("foo".into(), [1, 2, 3]));
    /// # Ok(())
    /// }
    /// ```
    ///
    /// See [`SortOptions`] for more options.
    pub fn sort(&self, sort_options: SortOptions) -> PolarsResult<Self> {
        self.sort_with(sort_options)
    }

    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self._get_inner_mut().as_single_ptr()
    }

    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        self.cast_with_options(dtype, CastOptions::NonStrict)
    }

    /// Cast [`Series`] to another [`DataType`].
    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        use DataType as D;

        let do_clone = match dtype {
            D::Unknown(UnknownKind::Any) => true,
            D::Unknown(UnknownKind::Int(_)) if self.dtype().is_integer() => true,
            D::Unknown(UnknownKind::Float) if self.dtype().is_float() => true,
            D::Unknown(UnknownKind::Str)
                if self.dtype().is_string() | self.dtype().is_categorical() =>
            {
                true
            },
            dt if dt.is_primitive() && dt == self.dtype() => true,
            _ => false,
        };

        if do_clone {
            return Ok(self.clone());
        }

        pub fn cast_dtype(dtype: &DataType) -> Option<DataType> {
            match dtype {
                D::Unknown(UnknownKind::Int(v)) => Some(materialize_dyn_int(*v).dtype()),
                D::Unknown(UnknownKind::Float) => Some(DataType::Float64),
                D::Unknown(UnknownKind::Str) => Some(DataType::String),
                // Best leave as is.
                D::List(inner) => cast_dtype(inner.as_ref()).map(Box::new).map(D::List),
                #[cfg(feature = "dtype-struct")]
                D::Struct(fields) => {
                    // @NOTE: We only allocate if we really need to.

                    let mut field_iter = fields.iter().enumerate();
                    let mut new_fields = loop {
                        let (i, field) = field_iter.next()?;

                        if let Some(dtype) = cast_dtype(&field.dtype) {
                            let mut new_fields = Vec::with_capacity(fields.len());
                            new_fields.extend(fields.iter().take(i).cloned());
                            new_fields.push(Field {
                                name: field.name.clone(),
                                dtype,
                            });
                            break new_fields;
                        }
                    };

                    new_fields.extend(fields.iter().skip(new_fields.len()).cloned().map(|field| {
                        let dtype = cast_dtype(&field.dtype).unwrap_or(field.dtype);
                        Field {
                            name: field.name.clone(),
                            dtype,
                        }
                    }));

                    Some(D::Struct(new_fields))
                },
                _ => None,
            }
        }

        let casted = cast_dtype(dtype);
        let dtype = match casted {
            None => dtype,
            Some(ref dtype) => dtype,
        };

        // Always allow casting all nulls to other all nulls.
        let len = self.len();
        if self.null_count() == len {
            return Ok(Series::full_null(self.name().clone(), len, dtype));
        }

        let new_options = match options {
            // Strictness is handled on this level to improve error messages.
            CastOptions::Strict => CastOptions::NonStrict,
            opt => opt,
        };

        let ret = self.0.cast(dtype, new_options);

        match options {
            CastOptions::NonStrict | CastOptions::Overflowing => ret,
            CastOptions::Strict => {
                let ret = ret?;
                if self.null_count() != ret.null_count() {
                    handle_casting_failures(self, &ret)?;
                }
                Ok(ret)
            },
        }
    }

    /// Cast from physical to logical types without any checks on the validity of the cast.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        match self.dtype() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => self.struct_().unwrap().cast_unchecked(dtype),
            DataType::List(_) => self.list().unwrap().cast_unchecked(dtype),
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let ca: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                        ca.cast_unchecked(dtype)
                })
            },
            DataType::Binary => self.binary().unwrap().cast_unchecked(dtype),
            _ => self.cast_with_options(dtype, CastOptions::Overflowing),
        }
    }

    /// Convert a non-logical series back into a logical series without casting.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn from_physical_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        debug_assert!(!self.dtype().is_logical());

        if self.dtype() == dtype {
            return Ok(self.clone());
        }

        use DataType as D;
        match (self.dtype(), dtype) {
            #[cfg(feature = "dtype-decimal")]
            (D::Int128, D::Decimal(precision, scale)) => {
                self.clone().into_decimal(*precision, scale.unwrap())
            },

            #[cfg(feature = "dtype-categorical")]
            (D::UInt32, D::Categorical(revmap, ordering)) => Ok(unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(
                    self.u32().unwrap().clone(),
                    revmap.as_ref().unwrap().clone(),
                    false,
                    *ordering,
                )
            }
            .into_series()),
            #[cfg(feature = "dtype-categorical")]
            (D::UInt32, D::Enum(revmap, ordering)) => Ok(unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(
                    self.u32().unwrap().clone(),
                    revmap.as_ref().unwrap().clone(),
                    true,
                    *ordering,
                )
            }
            .into_series()),

            (D::Int32, D::Date) => feature_gated!("dtype-time", Ok(self.clone().into_date())),
            (D::Int64, D::Datetime(tu, tz)) => feature_gated!(
                "dtype-datetime",
                Ok(self.clone().into_datetime(*tu, tz.clone()))
            ),
            (D::Int64, D::Duration(tu)) => {
                feature_gated!("dtype-duration", Ok(self.clone().into_duration(*tu)))
            },
            (D::Int64, D::Time) => feature_gated!("dtype-time", Ok(self.clone().into_time())),

            (D::List(_), D::List(to)) => unsafe {
                self.list()
                    .unwrap()
                    .from_physical_unchecked(to.as_ref().clone())
                    .map(|ca| ca.into_series())
            },
            #[cfg(feature = "dtype-array")]
            (D::Array(_, lw), D::Array(to, rw)) if lw == rw => unsafe {
                self.array()
                    .unwrap()
                    .from_physical_unchecked(to.as_ref().clone())
                    .map(|ca| ca.into_series())
            },
            #[cfg(feature = "dtype-struct")]
            (D::Struct(_), D::Struct(to)) => unsafe {
                self.struct_()
                    .unwrap()
                    .from_physical_unchecked(to.as_slice())
                    .map(|ca| ca.into_series())
            },

            _ => panic!("invalid from_physical({dtype:?}) for {:?}", self.dtype()),
        }
    }

    /// Cast numerical types to f64, and keep floats as is.
    pub fn to_float(&self) -> PolarsResult<Series> {
        match self.dtype() {
            DataType::Float32 | DataType::Float64 => Ok(self.clone()),
            _ => self.cast_with_options(&DataType::Float64, CastOptions::Overflowing),
        }
    }

    /// Compute the sum of all values in this Series.
    /// Returns `Some(0)` if the array is empty, and `None` if the array only
    /// contains null values.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    pub fn sum<T>(&self) -> PolarsResult<T>
    where
        T: NumCast,
    {
        let sum = self.sum_reduce()?;
        let sum = sum.value().extract().unwrap();
        Ok(sum)
    }

    /// Returns the minimum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    pub fn min<T>(&self) -> PolarsResult<Option<T>>
    where
        T: NumCast,
    {
        let min = self.min_reduce()?;
        let min = min.value().extract::<T>();
        Ok(min)
    }

    /// Returns the maximum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    pub fn max<T>(&self) -> PolarsResult<Option<T>>
    where
        T: NumCast,
    {
        let max = self.max_reduce()?;
        let max = max.value().extract::<T>();
        Ok(max)
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

    /// Check if numeric value is NaN (note this is different than missing/ null)
    pub fn is_nan(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_nan()),
            dt if dt.is_primitive_numeric() => {
                let arr = BooleanArray::full(self.len(), false, ArrowDataType::Boolean)
                    .with_validity(self.rechunk_validity());
                Ok(BooleanChunked::with_chunk(self.name().clone(), arr))
            },
            _ => polars_bail!(opq = is_nan, self.dtype()),
        }
    }

    /// Check if numeric value is NaN (note this is different than missing/null)
    pub fn is_not_nan(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_not_nan()),
            DataType::Float64 => Ok(self.f64().unwrap().is_not_nan()),
            dt if dt.is_primitive_numeric() => {
                let arr = BooleanArray::full(self.len(), true, ArrowDataType::Boolean)
                    .with_validity(self.rechunk_validity());
                Ok(BooleanChunked::with_chunk(self.name().clone(), arr))
            },
            _ => polars_bail!(opq = is_not_nan, self.dtype()),
        }
    }

    /// Check if numeric value is finite
    pub fn is_finite(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_finite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_finite()),
            dt if dt.is_primitive_numeric() => {
                let arr = BooleanArray::full(self.len(), true, ArrowDataType::Boolean)
                    .with_validity(self.rechunk_validity());
                Ok(BooleanChunked::with_chunk(self.name().clone(), arr))
            },
            _ => polars_bail!(opq = is_finite, self.dtype()),
        }
    }

    /// Check if numeric value is infinite
    pub fn is_infinite(&self) -> PolarsResult<BooleanChunked> {
        match self.dtype() {
            DataType::Float32 => Ok(self.f32().unwrap().is_infinite()),
            DataType::Float64 => Ok(self.f64().unwrap().is_infinite()),
            dt if dt.is_primitive_numeric() => {
                let arr = BooleanArray::full(self.len(), false, ArrowDataType::Boolean)
                    .with_validity(self.rechunk_validity());
                Ok(BooleanChunked::with_chunk(self.name().clone(), arr))
            },
            _ => polars_bail!(opq = is_infinite, self.dtype()),
        }
    }

    /// Create a new ChunkedArray with values from self where the mask evaluates `true` and values
    /// from `other` where the mask evaluates `false`. This function automatically broadcasts unit
    /// length inputs.
    #[cfg(feature = "zip_with")]
    pub fn zip_with(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let (lhs, rhs) = coerce_lhs_rhs(self, other)?;
        lhs.zip_with_same_type(mask, rhs.as_ref())
    }

    /// Converts a Series to their physical representation, if they have one,
    /// otherwise the series is left unchanged.
    ///
    /// * Date -> Int32
    /// * Datetime -> Int64
    /// * Duration -> Int64
    /// * Decimal -> Int128
    /// * Time -> Int64
    /// * Categorical -> UInt32
    /// * List(inner) -> List(physical of inner)
    /// * Array(inner) -> Array(physical of inner)
    /// * Struct -> Struct with physical repr of each struct column
    pub fn to_physical_repr(&self) -> Cow<Series> {
        use DataType::*;
        match self.dtype() {
            // NOTE: Don't use cast here, as it might rechunk (if all nulls)
            // which is not allowed in a phys repr.
            #[cfg(feature = "dtype-date")]
            Date => Cow::Owned(self.date().unwrap().0.clone().into_series()),
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, _) => Cow::Owned(self.datetime().unwrap().0.clone().into_series()),
            #[cfg(feature = "dtype-duration")]
            Duration(_) => Cow::Owned(self.duration().unwrap().0.clone().into_series()),
            #[cfg(feature = "dtype-time")]
            Time => Cow::Owned(self.time().unwrap().0.clone().into_series()),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => {
                let ca = self.categorical().unwrap();
                Cow::Owned(ca.physical().clone().into_series())
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => Cow::Owned(self.decimal().unwrap().0.clone().into_series()),
            List(_) => match self.list().unwrap().to_physical_repr() {
                Cow::Borrowed(_) => Cow::Borrowed(self),
                Cow::Owned(ca) => Cow::Owned(ca.into_series()),
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => match self.array().unwrap().to_physical_repr() {
                Cow::Borrowed(_) => Cow::Borrowed(self),
                Cow::Owned(ca) => Cow::Owned(ca.into_series()),
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => match self.struct_().unwrap().to_physical_repr() {
                Cow::Borrowed(_) => Cow::Borrowed(self),
                Cow::Owned(ca) => Cow::Owned(ca.into_series()),
            },
            _ => Cow::Borrowed(self),
        }
    }

    /// Traverse and collect every nth element in a new array.
    pub fn gather_every(&self, n: usize, offset: usize) -> Series {
        let idx = ((offset as IdxSize)..self.len() as IdxSize)
            .step_by(n)
            .collect_ca(PlSmallStr::EMPTY);
        // SAFETY: we stay in-bounds.
        unsafe { self.take_unchecked(&idx) }
    }

    #[cfg(feature = "dot_product")]
    pub fn dot(&self, other: &Series) -> PolarsResult<f64> {
        std::ops::Mul::mul(self, other)?.sum::<f64>()
    }

    /// Get the sum of the Series as a new Series of length 1.
    /// Returns a Series with a single zeroed entry if self is an empty numeric series.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    pub fn sum_reduce(&self) -> PolarsResult<Scalar> {
        use DataType::*;
        match self.dtype() {
            Int8 | UInt8 | Int16 | UInt16 => self.cast(&Int64).unwrap().sum_reduce(),
            _ => self.0.sum_reduce(),
        }
    }

    /// Get the product of an array.
    ///
    /// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
    /// first cast to `Int64` to prevent overflow issues.
    pub fn product(&self) -> PolarsResult<Scalar> {
        #[cfg(feature = "product")]
        {
            use DataType::*;
            match self.dtype() {
                Boolean => self.cast(&DataType::Int64).unwrap().product(),
                Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 => {
                    let s = self.cast(&Int64).unwrap();
                    s.product()
                },
                Int64 => Ok(self.i64().unwrap().prod_reduce()),
                UInt64 => Ok(self.u64().unwrap().prod_reduce()),
                #[cfg(feature = "dtype-i128")]
                Int128 => Ok(self.i128().unwrap().prod_reduce()),
                Float32 => Ok(self.f32().unwrap().prod_reduce()),
                Float64 => Ok(self.f64().unwrap().prod_reduce()),
                dt => {
                    polars_bail!(InvalidOperation: "`product` operation not supported for dtype `{dt}`")
                },
            }
        }
        #[cfg(not(feature = "product"))]
        {
            panic!("activate 'product' feature")
        }
    }

    /// Cast throws an error if conversion had overflows
    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Strict)
    }

    #[cfg(feature = "dtype-decimal")]
    pub(crate) fn into_decimal(
        self,
        precision: Option<usize>,
        scale: usize,
    ) -> PolarsResult<Series> {
        match self.dtype() {
            DataType::Int128 => Ok(self
                .i128()
                .unwrap()
                .clone()
                .into_decimal(precision, scale)?
                .into_series()),
            DataType::Decimal(cur_prec, cur_scale)
                if (cur_prec.is_none() || precision.is_none() || *cur_prec == precision)
                    && *cur_scale == Some(scale) =>
            {
                Ok(self)
            },
            dt => panic!("into_decimal({precision:?}, {scale}) not implemented for {dt:?}"),
        }
    }

    #[cfg(feature = "dtype-time")]
    pub(crate) fn into_time(self) -> Series {
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

    #[allow(unused_variables)]
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

    #[allow(unused_variables)]
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
        Ok(self.0.get(index)?.str_value())
    }
    /// Get the head of the Series.
    pub fn head(&self, length: Option<usize>) -> Series {
        let len = length.unwrap_or(HEAD_DEFAULT_LENGTH);
        self.slice(0, std::cmp::min(len, self.len()))
    }

    /// Get the tail of the Series.
    pub fn tail(&self, length: Option<usize>) -> Series {
        let len = length.unwrap_or(TAIL_DEFAULT_LENGTH);
        let len = std::cmp::min(len, self.len());
        self.slice(-(len as i64), len)
    }

    pub fn mean_reduce(&self) -> Scalar {
        crate::scalar::reduce::mean_reduce(self.mean(), self.dtype().clone())
    }

    /// Compute the unique elements, but maintain order. This requires more work
    /// than a naive [`Series::unique`](SeriesTrait::unique).
    pub fn unique_stable(&self) -> PolarsResult<Series> {
        let idx = self.arg_unique()?;
        // SAFETY: Indices are in bounds.
        unsafe { Ok(self.take_unchecked(&idx)) }
    }

    pub fn try_idx(&self) -> Option<&IdxCa> {
        #[cfg(feature = "bigidx")]
        {
            self.try_u64()
        }
        #[cfg(not(feature = "bigidx"))]
        {
            self.try_u32()
        }
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
        let mut size = 0;
        match self.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(Some(rv), _) | DataType::Enum(Some(rv), _) => match &**rv {
                RevMapping::Local(arr, _) => size += estimated_bytes_size(arr),
                RevMapping::Global(map, arr, _) => {
                    size += map.capacity() * size_of::<u32>() * 2 + estimated_bytes_size(arr);
                },
            },
            #[cfg(feature = "object")]
            DataType::Object(_, _) => {
                let ArrowDataType::FixedSizeBinary(size) = self.chunks()[0].dtype() else {
                    unreachable!()
                };
                // This is only the pointer size in python. So will be a huge underestimation.
                return self.len() * *size;
            },
            _ => {},
        }

        size += self
            .chunks()
            .iter()
            .map(|arr| estimated_bytes_size(&**arr))
            .sum::<usize>();

        size
    }

    /// Packs every element into a list.
    pub fn as_list(&self) -> ListChunked {
        let s = self.rechunk();
        // don't  use `to_arrow` as we need the physical types
        let values = s.chunks()[0].clone();
        let offsets = (0i64..(s.len() as i64 + 1)).collect::<Vec<_>>();
        let offsets = unsafe { Offsets::new_unchecked(offsets) };

        let dtype = LargeListArray::default_datatype(
            s.dtype().to_physical().to_arrow(CompatLevel::newest()),
        );
        let new_arr = LargeListArray::new(dtype, offsets.into(), values, None);
        let mut out = ListChunked::with_chunk(s.name().clone(), new_arr);
        out.set_inner_dtype(s.dtype().clone());
        out
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

impl<T> AsRef<ChunkedArray<T>> for dyn SeriesTrait + '_
where
    T: 'static + PolarsDataType<IsLogical = FalseT>,
{
    fn as_ref(&self) -> &ChunkedArray<T> {
        // @NOTE: SeriesTrait `as_any` returns a std::any::Any for the underlying ChunkedArray /
        // Logical (so not the SeriesWrap).
        let Some(ca) = self.as_any().downcast_ref::<ChunkedArray<T>>() else {
            panic!(
                "implementation error, cannot get ref {:?} from {:?}",
                T::get_dtype(),
                self.dtype()
            );
        };

        ca
    }
}

impl<T> AsMut<ChunkedArray<T>> for dyn SeriesTrait + '_
where
    T: 'static + PolarsDataType<IsLogical = FalseT>,
{
    fn as_mut(&mut self) -> &mut ChunkedArray<T> {
        if !self.as_any_mut().is::<ChunkedArray<T>>() {
            panic!(
                "implementation error, cannot get ref {:?} from {:?}",
                T::get_dtype(),
                self.dtype()
            );
        }

        // @NOTE: SeriesTrait `as_any` returns a std::any::Any for the underlying ChunkedArray /
        // Logical (so not the SeriesWrap).
        self.as_any_mut().downcast_mut::<ChunkedArray<T>>().unwrap()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::series::*;

    #[test]
    fn cast() {
        let ar = UInt32Chunked::new("a".into(), &[1, 2]);
        let s = ar.into_series();
        let s2 = s.cast(&DataType::Int64).unwrap();

        assert!(s2.i64().is_ok());
        let s2 = s.cast(&DataType::Float32).unwrap();
        assert!(s2.f32().is_ok());
    }

    #[test]
    fn new_series() {
        let _ = Series::new("boolean series".into(), &vec![true, false, true]);
        let _ = Series::new("int series".into(), &[1, 2, 3]);
        let ca = Int32Chunked::new("a".into(), &[1, 2, 3]);
        let _ = ca.into_series();
    }

    #[test]
    #[cfg(feature = "dtype-date")]
    fn roundtrip_list_logical_20311() {
        let list = ListChunked::from_chunk_iter(
            PlSmallStr::from_static("a"),
            [ListArray::new(
                ArrowDataType::LargeList(Box::new(ArrowField::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Int32,
                    true,
                ))),
                unsafe { Offsets::new_unchecked(vec![0, 1]) }.into(),
                PrimitiveArray::new(ArrowDataType::Int32, vec![1i32].into(), None).to_boxed(),
                None,
            )],
        );
        let list = unsafe { list.from_physical_unchecked(DataType::Date) }.unwrap();
        assert_eq!(list.dtype(), &DataType::List(Box::new(DataType::Date)));
    }

    #[test]
    #[cfg(feature = "dtype-struct")]
    fn new_series_from_empty_structs() {
        let dtype = DataType::Struct(vec![]);
        let empties = vec![AnyValue::StructOwned(Box::new((vec![], vec![]))); 3];
        let s = Series::from_any_values_and_dtype("".into(), &empties, &dtype, false).unwrap();
        assert_eq!(s.len(), 3);
    }
    #[test]
    fn new_series_from_arrow_primitive_array() {
        let array = UInt32Array::from_slice([1, 2, 3, 4, 5]);
        let array_ref: ArrayRef = Box::new(array);

        let _ = Series::try_new("foo".into(), array_ref).unwrap();
    }

    #[test]
    fn series_append() {
        let mut s1 = Series::new("a".into(), &[1, 2]);
        let s2 = Series::new("b".into(), &[3]);
        s1.append(&s2).unwrap();
        assert_eq!(s1.len(), 3);

        // add wrong type
        let s2 = Series::new("b".into(), &[3.0]);
        assert!(s1.append(&s2).is_err())
    }

    #[test]
    #[cfg(feature = "dtype-decimal")]
    fn series_append_decimal() {
        let s1 = Series::new("a".into(), &[1.1, 2.3])
            .cast(&DataType::Decimal(None, Some(2)))
            .unwrap();
        let s2 = Series::new("b".into(), &[3])
            .cast(&DataType::Decimal(None, Some(0)))
            .unwrap();

        {
            let mut s1 = s1.clone();
            s1.append(&s2).unwrap();
            assert_eq!(s1.len(), 3);
            assert_eq!(s1.get(2).unwrap(), AnyValue::Decimal(300, 2));
        }

        {
            let mut s2 = s2.clone();
            s2.extend(&s1).unwrap();
            assert_eq!(s2.get(2).unwrap(), AnyValue::Decimal(2, 0));
        }
    }

    #[test]
    fn series_slice_works() {
        let series = Series::new("a".into(), &[1i64, 2, 3, 4, 5]);

        let slice_1 = series.slice(-3, 3);
        let slice_2 = series.slice(-5, 5);
        let slice_3 = series.slice(0, 5);

        assert_eq!(slice_1.get(0).unwrap(), AnyValue::Int64(3));
        assert_eq!(slice_2.get(0).unwrap(), AnyValue::Int64(1));
        assert_eq!(slice_3.get(0).unwrap(), AnyValue::Int64(1));
    }

    #[test]
    fn out_of_range_slice_does_not_panic() {
        let series = Series::new("a".into(), &[1i64, 2, 3, 4, 5]);

        let _ = series.slice(-3, 4);
        let _ = series.slice(-6, 2);
        let _ = series.slice(4, 2);
    }
}
