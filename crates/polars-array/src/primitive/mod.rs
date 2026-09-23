use std::borrow::Cow;

use polars_arrow::Either;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder, OptBitmapBuilder};
use polars_arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::vec::PushUnchecked;

use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    normalize_buffer, scalar_buffer_len, slice_buffer, slice_validity, try_validity_covering,
    validity_covering_unchecked,
};
use crate::builder::subslice_extend_validity;
use crate::flat::Flat;

mod builder;
pub(crate) mod bytes;
mod flat;
mod iterator;

pub use builder::PlPrimitiveArrayBuilder;
pub use iterator::{PlPrimitiveIter, PlPrimitiveValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional [`NativeType`] values.
#[derive(Clone)]
pub struct PlPrimitiveArray<T: NativeType> {
    /// Scalar: values.len() == 1
    values: Buffer<T>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl<T: NativeType> PlPrimitiveArray<T> {
    /// Creates a flat [`PlPrimitiveArray`] out of its internal components.
    ///
    /// # Errors
    /// Errors unless `values` holds `length` slots and `validity` covers `length` elements.
    pub fn try_new(
        values: Buffer<T>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is not flat for an array of length {}",
            values.len(), length,
        );

        Ok(Self {
            values,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components.
    #[inline]
    pub fn new(values: Buffer<T>, length: usize, validity: Option<PlBitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must both be flat and valid for `length` elements.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<T>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(values.len(), length));
        }

        Self {
            values,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// Errors unless `values` is scalar for `length` and `validity` covers `length` elements.
    pub fn try_new_broadcast(
        values: Buffer<T>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is not the single value the {} elements of a broadcast \
             array share",
            values.len(), length,
        );

        Ok(Self {
            values: normalize_buffer(values, length),
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components.
    #[inline]
    pub fn new_broadcast(values: Buffer<T>, length: usize, validity: Option<PlBitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must both be scalar and valid for `length` elements.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Buffer<T>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(values.len(), length));
        }

        Self {
            values: normalize_buffer(values, length),
            length,
            validity,
        }
    }

    /// Creates an empty [`PlPrimitiveArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            values: Buffer::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] from `values`.
    #[inline]
    pub fn from_values(values: Buffer<T>) -> Self {
        let length = values.len();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] from a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::from_values(Buffer::from(values))
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] by copying `values`.
    #[inline]
    pub fn from_slice(values: &[T]) -> Self {
        Self::from_vec(values.to_vec())
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: T, length: usize) -> Self {
        let values = if length == 0 {
            Buffer::new()
        } else {
            bytes::buffer_from_bytes::<T>(Buffer::from_owner([bytes::to_bytes(value)]))
        };

        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlPrimitiveArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: bytes::buffer_from_bytes::<T>(bytes::undetermined(scalar_buffer_len(length))),
            length,
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
        }
    }

    /// The values of this array, in whichever representation the backing buffer is in.
    #[inline]
    pub(crate) fn values_bytes(&self) -> bytes::ValuesBytes<'_, bytes::Bytes<T>> {
        match self.scalar_value_ignore_validity() {
            Some(value) => bytes::ValuesBytes::Scalar(bytes::to_bytes(value)),
            None => bytes::ValuesBytes::Flat(bytes::slice_to_bytes(self.values.as_slice())),
        }
    }

    /// The values slots this array holds, if no other array shares them.
    #[inline]
    pub fn flat_or_scalar_values_mut(&mut self) -> Option<&mut [T]> {
        self.values.get_mut_slice()
    }

    /// The backing values buffer, if it holds one slot per element.
    #[inline]
    pub fn flat_values(&self) -> Option<&Buffer<T>> {
        (!self.values_are_scalar()).then_some(&self.values)
    }

    /// The values buffer, if this array holds one slot per element and nothing else shares it.
    #[inline]
    pub fn flat_values_mut(&mut self) -> Option<&mut Buffer<T>> {
        (!self.values_are_scalar()).then_some(&mut self.values)
    }

    /// The backing values buffer, taken out of this array, if it holds one slot per element.
    ///
    /// Prefer this over cloning [`Self::flat_values`] where the array is not needed afterwards:
    /// a buffer nothing else shares hands its allocation over instead of copying it.
    #[inline]
    pub fn into_flat_values(self) -> Option<Buffer<T>> {
        (!self.values_are_scalar()).then_some(self.values)
    }

    /// The value every element of this array reads, if the values buffer holds a single slot.
    #[inline]
    pub fn scalar_value_ignore_validity(&self) -> Option<T> {
        self.values_are_scalar().then(|| self.values[0])
    }

    /// A builder that continues this array, reusing its values allocation rather than copying it.
    pub fn into_builder(self) -> Either<Self, PlPrimitiveArrayBuilder<T>> {
        if self.flat_values().is_none() {
            return Either::Left(self);
        }

        let mut builder_validity = OptBitmapBuilder::default();
        subslice_extend_validity(&mut builder_validity, self.validity(), 0, self.length);

        let Self {
            values,
            length,
            validity,
        } = self;

        match bytes::byte_vec_from_buffer(values) {
            Either::Right(values) => Either::Right(PlPrimitiveArrayBuilder::from_parts(
                values,
                builder_validity,
            )),
            Either::Left(values) => Either::Left(Self {
                values,
                length,
                validity,
            }),
        }
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values buffer holds a single value shared by every element.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == 1 && self.length > 0
    }

    /// Whether the values buffer holds one slot per element.
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        self.values.len() == self.length
    }

    /// Whether the validity mask holds a single value shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing buffer has one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is scalar throughout: one value repeated [`Self::len`] times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element equals, if every backing buffer holds one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<T>> {
        let is_shared = self.values.len() == 1
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns the value at `i`.
    #[inline]
    pub fn value(&self, i: usize) -> T {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> T {
        debug_assert!(i < self.length);
        unsafe {
            *self
                .values
                .get_unchecked(broadcast_index(i, self.values.len()))
        }
    }

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlPrimitiveValuesIter<'_, T> {
        PlPrimitiveValuesIter::new(&self.values, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlPrimitiveIter<'_, T> {
        PlPrimitiveIter::new(&self.values, self.validity(), self.length)
    }

    /// Iterates `length` values, repeating a scalar array's one value and ignoring validity.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlPrimitiveValuesIter<'_, T> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds a single slot, which is scalar for any length;
        // otherwise `length` is the length the values are already valid for.
        PlPrimitiveValuesIter::new(&self.values, length)
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        unsafe {
            slice_buffer(&mut self.values, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(length);
        }

        let value = unsafe { self.value_unchecked(index) };

        Self::new_scalar(value, length)
    }

    /// Returns this array with its elements in the opposite order, keeping the representation.
    #[must_use]
    pub fn reversed(&self) -> Self {
        if self.is_scalar() {
            return self.clone();
        }

        let validity = self
            .validity
            .as_ref()
            .map(|validity| PlBitmap::new_broadcast(validity.clone(), self.length).reversed());

        if self.values_are_scalar() {
            unsafe { Self::new_broadcast_unchecked(self.values.clone(), self.length, validity) }
        } else {
            let mut values = Vec::with_capacity(self.length);
            values.extend(self.values.as_slice().iter().rev().copied());

            // SAFETY: one slot was written per element, and the mask is this array's own reversed.
            unsafe { Self::new_unchecked(values.into(), self.length, validity) }
        }
    }

    /// Returns an equivalent flat array, borrowing this one if it is already flat.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let values = self.to_flat_values().into_owned();

        let validity = self
            .validity()
            .map(|validity| validity.to_flat().into_owned());

        // SAFETY: the values hold one slot per element, written out above, and the mask is the
        // flat counterpart of this array's own.
        Cow::Owned(unsafe {
            Flat::new(Self {
                values,
                length: self.length,
                validity,
            })
        })
    }

    /// The values buffer holding one slot per element, written out only if it is scalar.
    pub fn to_flat_values(&self) -> Cow<'_, Buffer<T>> {
        if self.values_are_flat() {
            return Cow::Borrowed(&self.values);
        }

        if self.length == 0 {
            return Cow::Owned(Buffer::new());
        }

        Cow::Owned(if self.scalar_value() == Some(None) {
            bytes::buffer_from_bytes::<T>(bytes::undetermined(self.length))
        } else {
            let value = bytes::to_bytes(self.values[0]);
            bytes::buffer_from_bytes::<T>(bytes::repeat(value, self.length))
        })
    }

    /// Borrows this array as a [`Flat`] one, if it is already flat.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing buffer of a flat array holds one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

crate::impl_array_methods!([T: NativeType] PlPrimitiveArray<T>, T);

impl<T: NativeType> Default for PlPrimitiveArray<T> {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<T: NativeType> From<Vec<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn from(values: Vec<T>) -> Self {
        Self::from_vec(values)
    }
}

impl<T: NativeType> From<Buffer<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn from(values: Buffer<T>) -> Self {
        Self::from_values(values)
    }
}

impl<T: NativeType> FromIterator<Option<T>> for PlPrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = Vec::with_capacity(lower);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for item in iter {
            if values.len() == values.capacity() {
                values.reserve(1);
                validity.reserve(values.capacity() - values.len());
            }

            // SAFETY: room for one more element was just made in both buffers.
            unsafe {
                values.push_unchecked(item.unwrap_or_default());
                validity.push_unchecked(item.is_some());
            }
        }

        let length = values.len();

        Self {
            values: Buffer::from(values),
            length,
            validity: validity.into_opt_validity(),
        }
    }
}

impl<T: NativeType> FromIterator<T> for PlPrimitiveArray<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

crate::impl_into_iterator!([T: NativeType] PlPrimitiveArray<T>, PlPrimitiveIter<'a, T>);

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant.
impl<T: NativeType> PartialEq for PlPrimitiveArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

crate::impl_element_debug!([T: NativeType] PlPrimitiveArray<T>, "PlPrimitiveArray");

crate::impl_pl_array!([T: NativeType] PlPrimitiveArray<T>, PlArrayType::Primitive(T::PRIMITIVE));
