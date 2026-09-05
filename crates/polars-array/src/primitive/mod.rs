use std::borrow::Cow;

use arrow::Either;
use arrow::bitmap::{Bitmap, MutableBitmap, OptBitmapBuilder};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::{
    ArrayRepr, assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    normalize_buffer, scalar_buffer_len, slice_buffer, slice_validity, try_validity_covering,
    validity_covering, validity_covering_unchecked,
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
    /// This function errors if `values` does not hold exactly `length` slots, or if `validity` does
    /// not cover exactly `length` elements.
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
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Buffer<T>, length: usize, validity: Option<PlBitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` must hold exactly `length` slots, and `validity` must cover exactly `length`
    /// elements, in either representation.
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
    /// This function errors if `values` is not scalar for `length`, per
    /// [`is_scalar_buffer_len`], or if `validity` does not cover exactly `length` elements, per
    /// [`is_scalar_buffer_len`].
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
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(values: Buffer<T>, length: usize, validity: Option<PlBitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `values` must be scalar for `length`, per [`is_scalar_buffer_len`]; `validity` must cover
    /// exactly `length` elements, in either representation.
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
        // There is no element for the value to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the value it repeats.
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
            // The value of a null element is undetermined, so the one slot every element shares
            // is never read and need not be written.
            values: bytes::buffer_from_bytes::<T>(bytes::undetermined(scalar_buffer_len(length))),
            length,
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
        }
    }

    /// The number of elements in this array.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether this array holds no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Which representation the backing values buffer is in, along with what it holds.
    #[inline]
    pub fn values_repr(&self) -> ArrayRepr<&Buffer<T>, T> {
        if self.values_are_scalar() {
            ArrayRepr::Scalar(self.values[0])
        } else {
            ArrayRepr::Flat(&self.values)
        }
    }

    /// Which representation the backing values buffer is in, along with its bytes.
    ///
    /// This is what the routines of [`bytes`] are handed: they move the values around without
    /// reading what they mean, and so are taken over the byte class of `T` rather than over `T`
    /// itself, which is nine copies of each instead of seventeen.
    #[inline]
    pub(crate) fn values_bytes(&self) -> bytes::ValuesBytes<'_, bytes::Bytes<T>> {
        match self.values_repr() {
            ArrayRepr::Flat(values) => ArrayRepr::Flat(bytes::slice_to_bytes(values.as_slice())),
            ArrayRepr::Scalar(value) => ArrayRepr::Scalar(bytes::to_bytes(value)),
        }
    }

    /// Which representation the backing values buffer is in, along with the buffer itself.
    ///
    /// Writing over the slots a buffer holds leaves it in the representation it is in, so both
    /// arms hand back the whole buffer: a caller that maps every slot maps a scalar buffer's
    /// single value once, and it still stands for every element.
    #[inline]
    pub fn values_repr_mut(&mut self) -> ArrayRepr<&mut Buffer<T>> {
        if self.values_are_scalar() {
            ArrayRepr::Scalar(&mut self.values)
        } else {
            ArrayRepr::Flat(&mut self.values)
        }
    }

    /// The backing values buffer, if it holds one slot per element.
    #[inline]
    pub fn flat_values(&self) -> Option<&Buffer<T>> {
        self.values_repr().flat()
    }

    /// The values buffer, if this array holds one slot per element and nothing else shares it.
    #[inline]
    pub fn flat_values_mut(&mut self) -> Option<&mut Buffer<T>> {
        self.values_repr_mut().flat()
    }

    /// The value every element of this array reads, if the values buffer holds a single slot.
    #[inline]
    pub fn scalar_values(&self) -> Option<T> {
        self.values_repr().scalar()
    }

    /// A builder that continues this array, reusing its values allocation rather than copying it.
    ///
    /// This is what makes appending to an array cheaper than concatenating onto it, and it is
    /// possible only when the values are flat, unsliced, and shared with nothing else — so the
    /// array is handed back untouched, on the left, whenever they are not. The validity mask is
    /// copied either way: it holds one *bit* per element, so reclaiming it would save a fraction
    /// of what reclaiming the values does, and it may be scalar, which a builder cannot hold.
    pub fn into_builder(self) -> Either<Self, PlPrimitiveArrayBuilder<T>> {
        if self.flat_values().is_none() {
            // Scalar values are a single slot standing for `length` elements; there is no
            // allocation of the right size to reclaim.
            return Either::Left(self);
        }

        // The mask is read off the array before it is taken apart, which is also what resolves a
        // scalar one into the flat bits a builder appends to.
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
            // The buffer came back untouched, so the array it came from is rebuilt as it was.
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
    ///
    /// An array of no elements holds no such value: it keeps the empty buffer in place of the one
    /// slot a scalar buffer would, and is flat.
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

    /// Whether this array is entirely stored in the scalar representation, and therefore is a
    /// single logical value repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if every backing buffer holds one
    /// slot.
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
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
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

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.length);
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<T> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<T> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// The number of null elements.
    ///
    /// Inlined so that an array with no mask to count is answered without a call at all: one left
    /// standing is an opaque write as far as the compiler is concerned, and sinks behind it every
    /// fact the caller had established about the array — the representation of its buffers
    /// included — which is exactly what a caller asking [`Self::has_nulls`] ahead of a walk is
    /// trying to hand the walk.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
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

    /// Returns an iterator over `length` values, repeating the single value of this array if that
    /// is all it holds, and ignoring validity.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlPrimitiveValuesIter<'_, T> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds a single slot, which is scalar for any length;
        // otherwise `length` is the length the values are already valid for.
        PlPrimitiveValuesIter::new(&self.values, length)
    }

    /// Returns this array with its validity mask replaced.
    ///
    /// The mask keeps whichever representation it is in: one that stands for a single bit shared
    /// by every element is not written out one bit per element to be set.
    ///
    /// # Panics
    /// Panics unless `validity` covers exactly [`len`](Self::len) elements.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<PlBitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask, which keeps the representation it is in.
    ///
    /// # Panics
    /// Panics unless `validity` covers exactly [`len`](Self::len) elements.
    pub fn set_validity(&mut self, validity: Option<PlBitmap>) {
        let length = self.len();
        self.validity = validity_covering(validity, length);
    }

    /// Drops the validity mask, making every element valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.validity = None;
        self
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.length,
            "the offset of the new slice must be smaller than the length of the array",
        );
        unsafe { self.slice_unchecked(offset, length) }
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

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn sliced(mut self, offset: usize, length: usize) -> Self {
        self.slice(offset, length);
        self
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        unsafe { self.slice_unchecked(offset, length) };
        self
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
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

        // The value of a null element is undetermined, so it is repeated as it is found: it is the
        // mask that makes every element of the result null.
        let value = unsafe { self.value_unchecked(index) };

        Self::new_scalar(value, length)
    }

    /// Returns an equivalent array whose backing buffers all hold one slot per element, borrowing
    /// this array itself if they already do.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        // Writing the repeated value out is the one costly step here, and it reads nothing of the
        // value but its bytes, so it is taken over the byte class of `T` rather than over `T`.
        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.length == 0 {
            Buffer::new()
        } else if self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated value need not be written out: a zeroed buffer stands in for it.
            bytes::buffer_from_bytes::<T>(bytes::undetermined(self.length))
        } else {
            let value = bytes::to_bytes(self.values[0]);
            bytes::buffer_from_bytes::<T>(bytes::repeat(value, self.length))
        };

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

    /// Borrows this array as a [`Flat`] one, if every backing buffer already holds one slot per
    /// element.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing buffer of a flat array holds one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

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
        let mut validity = MutableBitmap::with_capacity(lower);

        for item in iter {
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        let validity = Bitmap::from(validity);
        let validity = (validity.unset_bits() > 0).then_some(validity);

        Self {
            values: Buffer::from(values),
            length,
            validity,
        }
    }
}

impl<T: NativeType> FromIterator<T> for PlPrimitiveArray<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<'a, T: NativeType> IntoIterator for &'a PlPrimitiveArray<T> {
    type Item = Option<T>;
    type IntoIter = PlPrimitiveIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant.
impl<T: NativeType> PartialEq for PlPrimitiveArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        // Never walk two scalar arrays element by element: their length is unbounded by their
        // memory use.
        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl<T: NativeType> std::fmt::Debug for PlPrimitiveArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders nulls as `null` instead of `None`.
        struct Element<T>(Option<T>);

        impl<T: std::fmt::Debug> std::fmt::Debug for Element<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(value) => value.fmt(f),
                    None => f.write_str("null"),
                }
            }
        }

        f.write_str("PlPrimitiveArray")?;

        // Never materialize a scalar array: its length is unbounded by its memory use.
        if self.length > 1 {
            if let Some(element) = self.scalar_value() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

impl<T: NativeType> PlArray for PlPrimitiveArray<T> {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::Primitive(T::PRIMITIVE)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.is_scalar()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.slice(offset, length)
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe { self.slice_unchecked(offset, length) }
    }

    #[inline]
    fn set_validity(&mut self, validity: Option<PlBitmap>) {
        self.set_validity(validity)
    }

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        Box::new(unsafe { self.new_from_index_unchecked(index, length) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
    }

    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::StaticArrayBuilder;

    #[test]
    fn flat() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.value(1), 2);
        assert_eq!(arr.get(2), Some(3));
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(1), Some(2), Some(3)]);
        assert_eq!(arr.values_iter().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn scalar_scalars_values() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 4);

        assert_eq!(arr.len(), 4);
        assert!(arr.flat_values().is_none());
        assert_eq!(arr.scalar_values(), Some(7));
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(Some(7)));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(7));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(7); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [7; 4]);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_buffers() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3), Some(4)].into_iter().collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_values().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(3)]);
    }

    #[test]
    fn an_array_of_no_elements_keeps_no_slot() {
        // A single slot is scalar for no elements too, but there is no element left to read it, so
        // it is not kept: the array is flat, like every empty array, rather than scalar.
        let arr = PlPrimitiveArray::new_broadcast(
            Buffer::from(vec![7i32]),
            0,
            Some(PlBitmap::new_scalar(false, 0)),
        );

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.flat_values().unwrap().is_empty());
        assert!(arr.validity().unwrap().is_flat());
        assert!(arr.validity().unwrap().is_empty());

        // The same goes for a mask broadcast over an empty array after the fact.
        let arr = PlPrimitiveArray::<i32>::new_empty()
            .with_validity(Some(PlBitmap::new_scalar(false, 0)));

        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.validity().unwrap().is_empty());
    }

    #[test]
    fn a_sole_flat_array_gives_its_allocation_up_to_a_builder() {
        // Room to grow, so that appending has an allocation to append *into* — an array whose
        // buffer is exactly full would have to be moved wherever it came from.
        let mut values = Vec::with_capacity(8);
        values.extend([1i32, 2, 3]);
        let arr = PlPrimitiveArray::from_vec(values);
        let values_ptr = arr.flat_values().unwrap().as_slice().as_ptr();

        let Either::Right(mut builder) = arr.into_builder() else {
            panic!("an unshared flat array can be appended to in place");
        };
        builder.push_value(4);
        let built = builder.freeze();

        // The point of the whole exercise: the values were appended to where they already were.
        assert_eq!(built.flat_values().unwrap().as_slice().as_ptr(), values_ptr);
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [Some(1), Some(2), Some(3), Some(4)],
        );
    }

    #[test]
    fn a_shared_or_sliced_array_keeps_its_allocation() {
        let arr: PlPrimitiveArray<i32> = [Some(1), Some(2)].into_iter().collect();
        let _alive = arr.clone();
        assert!(
            arr.into_builder().is_left(),
            "a buffer another array still reads cannot be written into",
        );

        let arr: PlPrimitiveArray<i32> = [Some(1), Some(2), Some(3)].into_iter().collect();
        assert!(
            arr.sliced(1, 2).into_builder().is_left(),
            "a slice does not own the whole allocation it points into",
        );
    }

    #[test]
    fn scalar_values_have_no_allocation_to_reclaim() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        assert!(arr.into_builder().is_left());
    }

    #[test]
    fn a_reclaimed_builder_carries_the_mask_over_in_either_representation() {
        // A flat mask is copied bit for bit.
        let arr: PlPrimitiveArray<i32> = [Some(1), None].into_iter().collect();
        let Either::Right(builder) = arr.into_builder() else {
            unreachable!()
        };
        assert_eq!(builder.freeze().iter().collect::<Vec<_>>(), [Some(1), None]);

        // A scalar mask stands for one bit per element, and appending to it resolves it.
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(PlBitmap::new_scalar(false, 3)));
        let Either::Right(mut builder) = arr.into_builder() else {
            unreachable!()
        };
        builder.push_value(4);
        assert_eq!(
            builder.freeze().iter().collect::<Vec<_>>(),
            [None, None, None, Some(4)],
        );
    }
}
