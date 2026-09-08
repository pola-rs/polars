use std::borrow::Cow;
use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef, validity_eq};
use crate::broadcast::{
    assert_broadcastable, is_flat_fixed_size_values_len, is_scalar_fixed_size_values_len,
    normalize_buffer, scalar_buffer_len, slice_fixed_size_buffer, slice_validity,
    try_validity_covering, validity_covering, validity_covering_unchecked,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlFixedSizeBinaryArrayBuilder;
pub use iterator::{PlFixedSizeBinaryIter, PlFixedSizeBinaryValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional `width`-byte strings.
#[derive(Clone)]
pub struct PlFixedSizeBinaryArray {
    /// Scalar: values.len() == width
    values: Buffer<u8>,
    width: usize,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlFixedSizeBinaryArray {
    /// Creates a flat [`PlFixedSizeBinaryArray`] out of its internal components.
    ///
    /// # Errors
    /// Errors if `values` is not `length * width` bytes, or `validity` not `length` bits.
    pub fn try_new(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values buffer of length {} is not flat for a fixed size binary array of length {} \
             and width {}: it needs the width of every element laid end to end",
            values.len(), length, width,
        );

        Ok(Self {
            values,
            width,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlFixedSizeBinaryArray`] out of its internal components.
    #[inline]
    pub fn new(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new(values, width, length, validity).unwrap()
    }

    /// Creates a flat [`PlFixedSizeBinaryArray`] out of its components without validating them.
    ///
    /// # Safety
    /// `values` must hold `length * width` bytes and `validity` must cover `length` elements.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_fixed_size_values_len(values.len(), width, length));
        }

        Self {
            values,
            width,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlFixedSizeBinaryArray`] of `length` elements out of its components.
    ///
    /// # Errors
    /// Errors if `values` or `validity` is not scalar for `width` and `length`.
    pub fn try_new_broadcast(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values buffer of length {} is not the one element the {} elements of a broadcast \
             fixed size binary array of width {} cover",
            values.len(), length, width,
        );

        Ok(Self {
            values: normalize_buffer(values, length),
            width,
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlFixedSizeBinaryArray`] of `length` elements out of its components.
    #[inline]
    pub fn new_broadcast(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new_broadcast(values, width, length, validity).unwrap()
    }

    /// Creates a scalar [`PlFixedSizeBinaryArray`] of `length` elements without validating them.
    ///
    /// # Safety
    /// `values` must be scalar for `width` and `length`, and `validity` scalar for `length`.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_fixed_size_values_len(values.len(), width, length));
        }

        Self {
            values: normalize_buffer(values, length),
            width,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlFixedSizeBinaryArray`] of elements `width` bytes wide.
    #[inline]
    pub fn new_empty(width: usize) -> Self {
        Self {
            values: Buffer::new(),
            width,
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat array by cutting `values` into `width`-byte elements.
    pub fn from_values(values: Buffer<u8>, width: usize) -> Self {
        assert!(
            width > 0,
            "the length of a fixed size binary array of width zero cannot be taken from its \
             values",
        );
        assert!(
            values.len().is_multiple_of(width),
            "the values of length {} do not divide into elements of width {}",
            values.len(),
            width,
        );

        let length = values.len() / width;
        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// [`Self::from_values`] for a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<u8>, width: usize) -> Self {
        Self::from_values(Buffer::from(values), width)
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` copies of `value`, in its own memory.
    #[inline]
    pub fn new_scalar(value: &[u8], length: usize) -> Self {
        let width = value.len();

        // There is no element for the values to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the value it repeats.
        let values = if length == 0 {
            Buffer::new()
        } else {
            Buffer::from(value.to_vec())
        };

        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` nulls, `width` bytes wide each.
    #[inline]
    pub fn new_full_null(width: usize, length: usize) -> Self {
        Self {
            values: if length == 0 {
                Buffer::new()
            } else {
                Buffer::zeroed(width)
            },
            width,
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

    /// The number of bytes in every element.
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The backing values buffer, if it holds the bytes of every element, laid end to end.
    #[inline]
    pub fn flat_values(&self) -> Option<&Buffer<u8>> {
        (!self.values_are_scalar()).then_some(&self.values)
    }

    /// The bytes every element of this array reads, if the values hold a single element.
    ///
    /// The mask is not looked at: a null element still holds the bytes it covers, which is what
    /// a kernel that answers over the values alone reads too. [`Self::scalar_value`] answers
    /// over both axes, and is what a caller that has to honour nulls wants.
    #[inline]
    pub fn scalar_value_ignore_validity(&self) -> Option<&[u8]> {
        self.values_are_scalar().then(|| self.values.as_slice())
    }

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Buffer<u8>, usize, usize, Option<Bitmap>) {
        (self.values, self.width, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values hold one element that every element of this array shares.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == self.width && self.length >= 1
    }

    /// Whether the values hold the bytes of every element, laid end to end.
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        // A length times a width that overflows a `usize` is longer than any buffer can be, so
        // such an array is never flat.
        self.length.checked_mul(self.width) == Some(self.values.len())
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether the values hold the bytes of every element and the mask one bit per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is scalar throughout: one value repeated [`Self::len`] times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element equals, if both backing buffers hold one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&[u8]>> {
        let is_shared = self.values.len() == self.width
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// The [`Self::width`]-byte range of the backing values buffer that element `i` covers.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The [`Self::width`]-byte range of the backing values buffer that element `i` covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.length);

        // Scalar values hold the one element every element covers, so they are read from the
        // start; flat ones lay the elements end to end, one width apart.
        let start = if self.values_are_scalar() {
            0
        } else {
            i * self.width
        };
        start..start + self.width
    }

    /// Returns the bytes of the element at `i`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the values hold the width of every element, or the one they all cover.
        unsafe { self.values.get_unchecked(range) }
    }

    /// Returns whether the element at `i` is valid (non-null).
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

    /// Returns the bytes of the element at `i`, or `None` if it is null.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// The number of null elements.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns an iterator over the elements, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlFixedSizeBinaryValuesIter<'_> {
        PlFixedSizeBinaryValuesIter::new(self.values.as_slice(), self.width, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlFixedSizeBinaryIter<'_> {
        PlFixedSizeBinaryIter::new(
            self.values.as_slice(),
            self.width,
            self.validity(),
            self.length,
        )
    }

    /// Iterates `length` elements, repeating a scalar array's one value and ignoring validity.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlFixedSizeBinaryValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds the width of that one element, which is scalar
        // for any length; otherwise `length` is the length the values are already valid for.
        PlFixedSizeBinaryValuesIter::new(self.values.as_slice(), self.width, length)
    }

    /// Returns this array with its validity mask replaced.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<PlBitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask, which keeps the representation it is in.
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

        // There are no offsets to leave the bytes outside the slice behind, so they are sliced
        // along with it; see `slice_fixed_size_buffer`.
        unsafe {
            slice_fixed_size_buffer(&mut self.values, self.width, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    #[must_use]
    pub fn sliced(&self, offset: usize, length: usize) -> Self {
        let mut sliced = self.clone();
        sliced.slice(offset, length);
        sliced
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(&self, offset: usize, length: usize) -> Self {
        let mut sliced = self.clone();
        unsafe { sliced.slice_unchecked(offset, length) };
        sliced
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` copies of the element at `index`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        // The bytes of a null element are undetermined, so they are not carried over: it is the
        // mask that makes every element of the result null, over a zeroed element of the width.
        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(self.width, length);
        }

        // The element is sliced out of the values it is already in, which every element of the
        // result covers: nothing is copied.
        let range = unsafe { self.value_range_unchecked(index) };
        let values = if length == 0 {
            Buffer::new()
        } else {
            self.values.clone().sliced(range)
        };

        Self {
            values,
            width: self.width,
            length,
            validity: None,
        }
    }

    /// Returns an equivalent flat array, borrowing this one if it is already flat.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let validity = self
            .validity()
            .map(|validity| PlBitmap::from_bitmap(validity.to_flat().into_owned()));

        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.null_count() == self.length {
            // Every element is null, so every value is undetermined: a zeroed buffer of the right
            // length stands in for them, which is not written out one element at a time.
            Buffer::zeroed(self.flat_values_len())
        } else {
            // The one element every element covers, written out once per element.
            let flat_len = self.flat_values_len();
            let element = self.values.as_slice();

            let mut values = Vec::with_capacity(flat_len);
            for _ in 0..self.length {
                values.extend_from_slice(element);
            }
            Buffer::from(values)
        };

        // SAFETY: the values are the element every element covers, repeated once per element, and
        // the mask is the flat counterpart of one valid for this array's length, which leaves every
        Cow::Owned(unsafe {
            Flat::new(Self::new_unchecked(
                values,
                self.width,
                self.length,
                validity,
            ))
        })
    }

    /// Borrows this array as a [`Flat`] one, if it is already flat.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the values of a flat array hold the width of every element, and its mask one bit
        // per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }

    /// The number of bytes a flat counterpart of this array holds.
    #[inline]
    fn flat_values_len(&self) -> usize {
        self.length.checked_mul(self.width).expect(
            "the values of the flat counterpart of the fixed size binary array overflow a `usize`",
        )
    }
}

impl<'a> IntoIterator for &'a PlFixedSizeBinaryArray {
    type Item = Option<&'a [u8]>;
    type IntoIter = PlFixedSizeBinaryIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise, disregarding representation and the values of nulls.
impl PartialEq for PlFixedSizeBinaryArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length || self.width != other.width {
            return false;
        }

        if !validity_eq(self.validity(), other.validity(), self.length) {
            return false;
        }

        // Every element is null on both sides, so every value is undetermined and there is nothing
        // left to compare. This is also what keeps comparing two fully null scalar arrays `O(1)`.
        if self.length > 0 && self.null_count() == self.length {
            return true;
        }

        // Never walk two scalar arrays element by element: their length is unbounded by their
        // memory use. Comparing the one element they each stand for costs that element.
        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl Eq for PlFixedSizeBinaryArray {}

/// Compares an array of unknown representation against a flat one.
impl PartialEq<Flat<PlFixedSizeBinaryArray>> for PlFixedSizeBinaryArray {
    #[inline]
    fn eq(&self, other: &Flat<PlFixedSizeBinaryArray>) -> bool {
        *self == *other.as_array()
    }
}

impl std::fmt::Debug for PlFixedSizeBinaryArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The buffers are listed as they are backed, which is one element's worth for a scalar
        // array: this never materializes a length that is unbounded by the memory use.
        let mut s = f.debug_struct("PlFixedSizeBinaryArray");
        s.field("length", &self.length);
        s.field("width", &self.width);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("values", &self.values).finish()
    }
}

impl PlArray for PlFixedSizeBinaryArray {
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
        PlArrayType::FixedSizeBinary
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

    /// The bytes every test cuts into elements.
    fn values() -> Buffer<u8> {
        Buffer::from(vec![1u8, 2, 3, 4, 5, 6])
    }

    /// The three elements `[1, 2]`, `[3, 4]` and `[5, 6]` over [`values`].
    fn arr() -> PlFixedSizeBinaryArray {
        PlFixedSizeBinaryArray::from_values(values(), 2)
    }

    #[test]
    fn flat() {
        let arr = arr();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.width(), 2);
        assert!(!arr.is_empty());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.has_nulls());
        assert!(arr.is_valid(1));
        assert!(!arr.is_null(1));
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.values_are_flat());
        assert!(!arr.values_are_scalar());
        assert_eq!(arr.flat_values().unwrap().len(), 6);

        assert_eq!(arr.value_range(0), 0..2);
        assert_eq!(arr.value_range(1), 2..4);
        assert_eq!(arr.value_range(2), 4..6);

        assert_eq!(arr.value(0), [1, 2]);
        assert_eq!(arr.value(2), [5, 6]);
        assert_eq!(arr.get(1), Some([3, 4].as_slice()));
        assert_eq!(arr.iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn null_scalar() {
        // Every element is null, over the two bytes they all share: the mask is a single bit, and
        // the values are one element.
        let arr = PlFixedSizeBinaryArray::new_full_null(2, 1_000_000);

        assert!(arr.is_scalar());
        assert!(arr.validity_is_scalar());
        assert!(arr.values_are_scalar());
        assert!(!arr.values_are_flat());
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.scalar_value_ignore_validity().unwrap().len(), 2);
        assert_eq!(arr.null_count(), 1_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999));
        assert_eq!(arr.get(999_999), None);
        assert_eq!(arr.value_range(999_999), 0..2);
        assert_eq!(arr.scalar_value(), Some(None));

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
    }

    #[test]
    fn slicing_slices_the_values() {
        let arr = PlFixedSizeBinaryArray::new(
            values(),
            2,
            3,
            Some(PlBitmap::from_bitmap(Bitmap::from_iter([
                true, false, true,
            ]))),
        )
        .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_flat());

        // There is nothing to leave the bytes outside the slice behind.
        assert_eq!(arr.flat_values().unwrap().as_slice(), [3, 4, 5, 6]);
        assert_eq!(arr.value(1), [5, 6]);

        // Slicing away every element leaves no bytes behind either.
        let arr = arr.sliced(2, 0);
        assert!(arr.is_empty());
        assert_eq!(arr.flat_values().unwrap().len(), 0);
        assert_eq!(arr.width(), 2);
    }
}
