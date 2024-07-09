use std::ops::Range;

use either::Either;

use super::{Array, Splitable};
use crate::array::iterator::NonNullValuesIter;
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::*;
use crate::trusted_len::TrustedLen;
use crate::types::{days_ms, f16, i256, months_days_ns, NativeType};

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod from_natural;
pub mod iterator;

mod mutable;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::index::{Bounded, Indexable, NullCount};
use polars_utils::slice::{GetSaferUnchecked, SliceAble};

/// A [`PrimitiveArray`] is Arrow's semantically equivalent of an immutable `Vec<Option<T>>` where
/// T is [`NativeType`] (e.g. [`i32`]). It implements [`Array`].
///
/// One way to think about a [`PrimitiveArray`] is `(DataType, Arc<Vec<T>>, Option<Arc<Vec<u8>>>)`
/// where:
/// * the first item is the array's logical type
/// * the second is the immutable values
/// * the third is the immutable validity (whether a value is null or not as a bitmap).
///
/// The size of this struct is `O(1)`, as all data is stored behind an [`std::sync::Arc`].
/// # Example
/// ```
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::bitmap::Bitmap;
/// use polars_arrow::buffer::Buffer;
///
/// let array = PrimitiveArray::from([Some(1i32), None, Some(10)]);
/// assert_eq!(array.value(0), 1);
/// assert_eq!(array.iter().collect::<Vec<_>>(), vec![Some(&1i32), None, Some(&10)]);
/// assert_eq!(array.values_iter().copied().collect::<Vec<_>>(), vec![1, 0, 10]);
/// // the underlying representation
/// assert_eq!(array.values(), &Buffer::from(vec![1i32, 0, 10]));
/// assert_eq!(array.validity(), Some(&Bitmap::from([true, false, true])));
///
/// ```
#[derive(Clone)]
pub struct PrimitiveArray<T: NativeType> {
    data_type: ArrowDataType,
    values: Buffer<T>,
    validity: Option<Bitmap>,
}

pub(super) fn check<T: NativeType>(
    data_type: &ArrowDataType,
    values: &[T],
    validity_len: Option<usize>,
) -> PolarsResult<()> {
    if validity_len.map_or(false, |len| len != values.len()) {
        polars_bail!(ComputeError: "validity mask length must match the number of values")
    }

    if data_type.to_physical_type() != PhysicalType::Primitive(T::PRIMITIVE) {
        polars_bail!(ComputeError: "PrimitiveArray can only be initialized with a DataType whose physical type is Primitive")
    }
    Ok(())
}

impl<T: NativeType> PrimitiveArray<T> {
    /// The canonical method to create a [`PrimitiveArray`] out of its internal components.
    /// # Implementation
    /// This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors iff:
    /// * The validity is not `None` and its length is different from `values`'s length
    /// * The `data_type`'s [`PhysicalType`] is not equal to [`PhysicalType::Primitive(T::PRIMITIVE)`]
    pub fn try_new(
        data_type: ArrowDataType,
        values: Buffer<T>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        check(&data_type, &values, validity.as_ref().map(|v| v.len()))?;
        Ok(Self {
            data_type,
            values,
            validity,
        })
    }

    /// # Safety
    /// Doesn't check invariants
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        values: Buffer<T>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self {
            data_type,
            values,
            validity,
        }
    }

    /// Returns a new [`PrimitiveArray`] with a different logical type.
    ///
    /// This function is useful to assign a different [`ArrowDataType`] to the array.
    /// Used to change the arrays' logical type (see example).
    /// # Example
    /// ```
    /// use polars_arrow::array::Int32Array;
    /// use polars_arrow::datatypes::ArrowDataType;
    ///
    /// let array = Int32Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Date32);
    /// assert_eq!(
    ///    format!("{:?}", array),
    ///    "Date32[1970-01-02, None, 1970-01-03]"
    /// );
    /// ```
    /// # Panics
    /// Panics iff the `data_type`'s [`PhysicalType`] is not equal to [`PhysicalType::Primitive(T::PRIMITIVE)`]
    #[inline]
    #[must_use]
    pub fn to(self, data_type: ArrowDataType) -> Self {
        check(
            &data_type,
            &self.values,
            self.validity.as_ref().map(|v| v.len()),
        )
        .unwrap();
        Self {
            data_type,
            values: self.values,
            validity: self.validity,
        }
    }

    /// Creates a (non-null) [`PrimitiveArray`] from a vector of values.
    /// This function is `O(1)`.
    /// # Examples
    /// ```
    /// use polars_arrow::array::PrimitiveArray;
    ///
    /// let array = PrimitiveArray::from_vec(vec![1, 2, 3]);
    /// assert_eq!(format!("{:?}", array), "Int32[1, 2, 3]");
    /// ```
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::new(T::PRIMITIVE.into(), values.into(), None)
    }

    /// Returns an iterator over the values and validity, `Option<&T>`.
    #[inline]
    pub fn iter(&self) -> ZipValidity<&T, std::slice::Iter<T>, BitmapIter> {
        ZipValidity::new_with_validity(self.values().iter(), self.validity())
    }

    /// Returns an iterator of the values, `&T`, ignoring the arrays' validity.
    #[inline]
    pub fn values_iter(&self) -> std::slice::Iter<T> {
        self.values().iter()
    }

    /// Returns an iterator of the non-null values `T`.
    #[inline]
    pub fn non_null_values_iter(&self) -> NonNullValuesIter<'_, [T]> {
        NonNullValuesIter::new(self.values(), self.validity())
    }

    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// The values [`Buffer`].
    /// Values on null slots are undetermined (they can be anything).
    #[inline]
    pub fn values(&self) -> &Buffer<T> {
        &self.values
    }

    /// Returns the optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Returns the arrays' [`ArrowDataType`].
    #[inline]
    pub fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    /// Returns the value at slot `i`.
    ///
    /// Equivalent to `self.values()[i]`. The value of a null slot is undetermined (it can be anything).
    /// # Panic
    /// This function panics iff `i >= self.len`.
    #[inline]
    pub fn value(&self, i: usize) -> T {
        self.values[i]
    }

    /// Returns the value at index `i`.
    /// The value on null slots is undetermined (it can be anything).
    ///
    /// # Safety
    /// Caller must be sure that `i < self.len()`
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> T {
        *self.values.get_unchecked_release(i)
    }

    // /// Returns the element at index `i` or `None` if it is null
    // /// # Panics
    // /// iff `i >= self.len()`
    // #[inline]
    // pub fn get(&self, i: usize) -> Option<T> {
    //     if !self.is_null(i) {
    //         // soundness: Array::is_null panics if i >= self.len
    //         unsafe { Some(self.value_unchecked(i)) }
    //     } else {
    //         None
    //     }
    // }

    /// Slices this [`PrimitiveArray`] by an offset and length.
    /// # Implementation
    /// This operation is `O(1)`.
    #[inline]
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "offset + length may not exceed length of array"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`PrimitiveArray`] by an offset and length.
    /// # Implementation
    /// This operation is `O(1)`.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    #[inline]
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.values.slice_unchecked(offset, length);
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();

    /// Returns this [`PrimitiveArray`] with new values.
    /// # Panics
    /// This function panics iff `values.len() != self.len()`.
    #[must_use]
    pub fn with_values(mut self, values: Buffer<T>) -> Self {
        self.set_values(values);
        self
    }

    /// Update the values of this [`PrimitiveArray`].
    /// # Panics
    /// This function panics iff `values.len() != self.len()`.
    pub fn set_values(&mut self, values: Buffer<T>) {
        assert_eq!(
            values.len(),
            self.len(),
            "values' length must be equal to this arrays' length"
        );
        self.values = values;
    }

    /// Applies a function `f` to the validity of this array.
    ///
    /// This is an API to leverage clone-on-write
    /// # Panics
    /// This function panics if the function `f` modifies the length of the [`Bitmap`].
    pub fn apply_validity<F: FnOnce(Bitmap) -> Bitmap>(&mut self, f: F) {
        if let Some(validity) = std::mem::take(&mut self.validity) {
            self.set_validity(Some(f(validity)))
        }
    }

    /// Returns an option of a mutable reference to the values of this [`PrimitiveArray`].
    pub fn get_mut_values(&mut self) -> Option<&mut [T]> {
        self.values.get_mut_slice()
    }

    /// Returns its internal representation
    #[must_use]
    pub fn into_inner(self) -> (ArrowDataType, Buffer<T>, Option<Bitmap>) {
        let Self {
            data_type,
            values,
            validity,
        } = self;
        (data_type, values, validity)
    }

    /// Creates a `[PrimitiveArray]` from its internal representation.
    /// This is the inverted from `[PrimitiveArray::into_inner]`
    pub fn from_inner(
        data_type: ArrowDataType,
        values: Buffer<T>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        check(&data_type, &values, validity.as_ref().map(|v| v.len()))?;
        Ok(unsafe { Self::from_inner_unchecked(data_type, values, validity) })
    }

    /// Creates a `[PrimitiveArray]` from its internal representation.
    /// This is the inverted from `[PrimitiveArray::into_inner]`
    ///
    /// # Safety
    /// Callers must ensure all invariants of this struct are upheld.
    pub unsafe fn from_inner_unchecked(
        data_type: ArrowDataType,
        values: Buffer<T>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self {
            data_type,
            values,
            validity,
        }
    }

    /// Try to convert this [`PrimitiveArray`] to a [`MutablePrimitiveArray`] via copy-on-write semantics.
    ///
    /// A [`PrimitiveArray`] is backed by a [`Buffer`] and [`Bitmap`] which are essentially `Arc<Vec<_>>`.
    /// This function returns a [`MutablePrimitiveArray`] (via [`std::sync::Arc::get_mut`]) iff both values
    /// and validity have not been cloned / are unique references to their underlying vectors.
    ///
    /// This function is primarily used to reuse memory regions.
    #[must_use]
    pub fn into_mut(self) -> Either<Self, MutablePrimitiveArray<T>> {
        use Either::*;

        if let Some(bitmap) = self.validity {
            match bitmap.into_mut() {
                Left(bitmap) => Left(PrimitiveArray::new(
                    self.data_type,
                    self.values,
                    Some(bitmap),
                )),
                Right(mutable_bitmap) => match self.values.into_mut() {
                    Right(values) => Right(
                        MutablePrimitiveArray::try_new(
                            self.data_type,
                            values,
                            Some(mutable_bitmap),
                        )
                        .unwrap(),
                    ),
                    Left(values) => Left(PrimitiveArray::new(
                        self.data_type,
                        values,
                        Some(mutable_bitmap.into()),
                    )),
                },
            }
        } else {
            match self.values.into_mut() {
                Right(values) => {
                    Right(MutablePrimitiveArray::try_new(self.data_type, values, None).unwrap())
                },
                Left(values) => Left(PrimitiveArray::new(self.data_type, values, None)),
            }
        }
    }

    /// Returns a new empty (zero-length) [`PrimitiveArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        Self::new(data_type, Buffer::new(), None)
    }

    /// Returns a new [`PrimitiveArray`] where all slots are null / `None`.
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        Self::new(
            data_type,
            vec![T::default(); length].into(),
            Some(Bitmap::new_zeroed(length)),
        )
    }

    /// Creates a (non-null) [`PrimitiveArray`] from an iterator of values.
    /// # Implementation
    /// This does not assume that the iterator has a known length.
    pub fn from_values<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(T::PRIMITIVE.into(), Vec::<T>::from_iter(iter).into(), None)
    }

    /// Creates a (non-null) [`PrimitiveArray`] from a slice of values.
    /// # Implementation
    /// This is essentially a memcopy and is thus `O(N)`
    pub fn from_slice<P: AsRef<[T]>>(slice: P) -> Self {
        Self::new(
            T::PRIMITIVE.into(),
            Vec::<T>::from(slice.as_ref()).into(),
            None,
        )
    }

    /// Creates a (non-null) [`PrimitiveArray`] from a [`TrustedLen`] of values.
    /// # Implementation
    /// This does not assume that the iterator has a known length.
    pub fn from_trusted_len_values_iter<I: TrustedLen<Item = T>>(iter: I) -> Self {
        MutablePrimitiveArray::<T>::from_trusted_len_values_iter(iter).into()
    }

    /// Creates a new [`PrimitiveArray`] from an iterator over values
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    pub unsafe fn from_trusted_len_values_iter_unchecked<I: Iterator<Item = T>>(iter: I) -> Self {
        MutablePrimitiveArray::<T>::from_trusted_len_values_iter_unchecked(iter).into()
    }

    /// Creates a [`PrimitiveArray`] from a [`TrustedLen`] of optional values.
    pub fn from_trusted_len_iter<I: TrustedLen<Item = Option<T>>>(iter: I) -> Self {
        MutablePrimitiveArray::<T>::from_trusted_len_iter(iter).into()
    }

    /// Creates a [`PrimitiveArray`] from an iterator of optional values.
    ///
    /// # Safety
    /// The iterator must be [`TrustedLen`](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
    /// I.e. that `size_hint().1` correctly reports its length.
    pub unsafe fn from_trusted_len_iter_unchecked<I: Iterator<Item = Option<T>>>(iter: I) -> Self {
        MutablePrimitiveArray::<T>::from_trusted_len_iter_unchecked(iter).into()
    }

    /// Alias for `Self::try_new(..).unwrap()`.
    /// # Panics
    /// This function errors iff:
    /// * The validity is not `None` and its length is different from `values`'s length
    /// * The `data_type`'s [`PhysicalType`] is not equal to [`PhysicalType::Primitive`].
    pub fn new(data_type: ArrowDataType, values: Buffer<T>, validity: Option<Bitmap>) -> Self {
        Self::try_new(data_type, values, validity).unwrap()
    }

    /// Transmute this PrimitiveArray into another PrimitiveArray.
    ///
    /// T and U must have the same size and alignment.
    pub fn transmute<U: NativeType>(self) -> PrimitiveArray<U> {
        let PrimitiveArray {
            values, validity, ..
        } = self;

        // SAFETY: this is fine, we checked size and alignment, and NativeType
        // is always Pod.
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
        assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
        let new_values = unsafe { std::mem::transmute::<Buffer<T>, Buffer<U>>(values) };
        PrimitiveArray::new(U::PRIMITIVE.into(), new_values, validity)
    }

    /// Fills this entire array with the given value, leaving the validity mask intact.
    ///
    /// Reuses the memory of the PrimitiveArray if possible.
    pub fn fill_with(mut self, value: T) -> Self {
        if let Some(values) = self.get_mut_values() {
            for x in values.iter_mut() {
                *x = value;
            }
            self
        } else {
            let values = vec![value; self.len()];
            Self::new(T::PRIMITIVE.into(), values.into(), self.validity)
        }
    }
}

impl<T: NativeType> Array for PrimitiveArray<T> {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl<T: NativeType> Splitable for PrimitiveArray<T> {
    #[inline(always)]
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_values, rhs_values) = unsafe { self.values.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        (
            Self {
                data_type: self.data_type.clone(),
                values: lhs_values,
                validity: lhs_validity,
            },
            Self {
                data_type: self.data_type.clone(),
                values: rhs_values,
                validity: rhs_validity,
            },
        )
    }
}

impl<T: NativeType> SliceAble for PrimitiveArray<T> {
    unsafe fn slice_unchecked(&self, range: Range<usize>) -> Self {
        self.clone().sliced_unchecked(range.start, range.len())
    }

    fn slice(&self, range: Range<usize>) -> Self {
        self.clone().sliced(range.start, range.len())
    }
}

impl<T: NativeType> Indexable for PrimitiveArray<T> {
    type Item = Option<T>;

    fn get(&self, i: usize) -> Self::Item {
        if !self.is_null(i) {
            // soundness: Array::is_null panics if i >= self.len
            unsafe { Some(self.value_unchecked(i)) }
        } else {
            None
        }
    }

    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        if !self.is_null_unchecked(i) {
            Some(self.value_unchecked(i))
        } else {
            None
        }
    }
}

/// A type definition [`PrimitiveArray`] for `i8`
pub type Int8Array = PrimitiveArray<i8>;
/// A type definition [`PrimitiveArray`] for `i16`
pub type Int16Array = PrimitiveArray<i16>;
/// A type definition [`PrimitiveArray`] for `i32`
pub type Int32Array = PrimitiveArray<i32>;
/// A type definition [`PrimitiveArray`] for `i64`
pub type Int64Array = PrimitiveArray<i64>;
/// A type definition [`PrimitiveArray`] for `i128`
pub type Int128Array = PrimitiveArray<i128>;
/// A type definition [`PrimitiveArray`] for `i256`
pub type Int256Array = PrimitiveArray<i256>;
/// A type definition [`PrimitiveArray`] for [`days_ms`]
pub type DaysMsArray = PrimitiveArray<days_ms>;
/// A type definition [`PrimitiveArray`] for [`months_days_ns`]
pub type MonthsDaysNsArray = PrimitiveArray<months_days_ns>;
/// A type definition [`PrimitiveArray`] for `f16`
pub type Float16Array = PrimitiveArray<f16>;
/// A type definition [`PrimitiveArray`] for `f32`
pub type Float32Array = PrimitiveArray<f32>;
/// A type definition [`PrimitiveArray`] for `f64`
pub type Float64Array = PrimitiveArray<f64>;
/// A type definition [`PrimitiveArray`] for `u8`
pub type UInt8Array = PrimitiveArray<u8>;
/// A type definition [`PrimitiveArray`] for `u16`
pub type UInt16Array = PrimitiveArray<u16>;
/// A type definition [`PrimitiveArray`] for `u32`
pub type UInt32Array = PrimitiveArray<u32>;
/// A type definition [`PrimitiveArray`] for `u64`
pub type UInt64Array = PrimitiveArray<u64>;

/// A type definition [`MutablePrimitiveArray`] for `i8`
pub type Int8Vec = MutablePrimitiveArray<i8>;
/// A type definition [`MutablePrimitiveArray`] for `i16`
pub type Int16Vec = MutablePrimitiveArray<i16>;
/// A type definition [`MutablePrimitiveArray`] for `i32`
pub type Int32Vec = MutablePrimitiveArray<i32>;
/// A type definition [`MutablePrimitiveArray`] for `i64`
pub type Int64Vec = MutablePrimitiveArray<i64>;
/// A type definition [`MutablePrimitiveArray`] for `i128`
pub type Int128Vec = MutablePrimitiveArray<i128>;
/// A type definition [`MutablePrimitiveArray`] for `i256`
pub type Int256Vec = MutablePrimitiveArray<i256>;
/// A type definition [`MutablePrimitiveArray`] for [`days_ms`]
pub type DaysMsVec = MutablePrimitiveArray<days_ms>;
/// A type definition [`MutablePrimitiveArray`] for [`months_days_ns`]
pub type MonthsDaysNsVec = MutablePrimitiveArray<months_days_ns>;
/// A type definition [`MutablePrimitiveArray`] for `f16`
pub type Float16Vec = MutablePrimitiveArray<f16>;
/// A type definition [`MutablePrimitiveArray`] for `f32`
pub type Float32Vec = MutablePrimitiveArray<f32>;
/// A type definition [`MutablePrimitiveArray`] for `f64`
pub type Float64Vec = MutablePrimitiveArray<f64>;
/// A type definition [`MutablePrimitiveArray`] for `u8`
pub type UInt8Vec = MutablePrimitiveArray<u8>;
/// A type definition [`MutablePrimitiveArray`] for `u16`
pub type UInt16Vec = MutablePrimitiveArray<u16>;
/// A type definition [`MutablePrimitiveArray`] for `u32`
pub type UInt32Vec = MutablePrimitiveArray<u32>;
/// A type definition [`MutablePrimitiveArray`] for `u64`
pub type UInt64Vec = MutablePrimitiveArray<u64>;

impl<T: NativeType> Default for PrimitiveArray<T> {
    fn default() -> Self {
        PrimitiveArray::new(T::PRIMITIVE.into(), Default::default(), None)
    }
}

impl<T: NativeType> Bounded for PrimitiveArray<T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

impl<T: NativeType> NullCount for PrimitiveArray<T> {
    fn null_count(&self) -> usize {
        <Self as Array>::null_count(self)
    }
}
