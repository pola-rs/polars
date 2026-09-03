use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{PolarsResult, polars_ensure};

use crate::broadcast::{
    is_flat_buffer_len, is_valid_buffer_len, normalize_bitmap, scalar_buffer_len,
};

mod iterator;
mod reference;
mod validity;

pub use iterator::PlBitmapIter;
pub(crate) use iterator::{ValidityFold, ValidityIter};
pub use reference::PlBitmapRef;
pub use validity::{combine_validities_and, invert};

/// An immutable, cheaply cloneable mask of `length` bits, in either the flat or the scalar
/// representation.
#[derive(Clone)]
pub struct PlBitmap {
    /// Scalar: bitmap.len() == 1
    bitmap: Bitmap,
    length: usize,
}

impl PlBitmap {
    /// Creates a flat [`PlBitmap`] of `length` bits backed by `bitmap`.
    ///
    /// # Errors
    /// This function errors unless `bitmap` holds one bit per element.
    pub fn try_new(bitmap: Bitmap, length: usize) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(bitmap.len(), length),
            ComputeError:
            "bitmap of length {} is not flat for a mask of length {}",
            bitmap.len(), length,
        );

        Ok(Self { bitmap, length })
    }

    /// Creates a flat [`PlBitmap`] of `length` bits backed by `bitmap`.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(bitmap: Bitmap, length: usize) -> Self {
        Self::try_new(bitmap, length).unwrap()
    }

    /// Creates a flat [`PlBitmap`] of `length` bits backed by `bitmap`, without validating it.
    ///
    /// # Safety
    /// `bitmap` must hold one bit per element, i.e. have `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(bitmap: Bitmap, length: usize) -> Self {
        debug_assert!(is_flat_buffer_len(bitmap.len(), length));
        Self { bitmap, length }
    }

    /// Creates a [`PlBitmap`] of `length` bits backed by a `bitmap` that broadcasts over them.
    ///
    /// # Errors
    /// This function errors if `bitmap` is neither flat (length equal to `length`) nor scalar
    /// (length one).
    pub fn try_new_broadcast(bitmap: Bitmap, length: usize) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_buffer_len(bitmap.len(), length),
            ComputeError:
            "bitmap of length {} is neither flat nor scalar for a mask of length {}",
            bitmap.len(), length,
        );

        Ok(Self {
            bitmap: normalize_bitmap(bitmap, length),
            length,
        })
    }

    /// Creates a [`PlBitmap`] of `length` bits backed by a `bitmap` that broadcasts over them.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(bitmap: Bitmap, length: usize) -> Self {
        Self::try_new_broadcast(bitmap, length).unwrap()
    }

    /// Creates a [`PlBitmap`] of `length` bits backed by a `bitmap` that broadcasts over them,
    /// without validating it.
    ///
    /// # Safety
    /// `bitmap` must be flat or scalar for `length`, per [`is_valid_buffer_len`].
    #[inline]
    pub unsafe fn new_broadcast_unchecked(bitmap: Bitmap, length: usize) -> Self {
        debug_assert!(is_valid_buffer_len(bitmap.len(), length));
        Self {
            bitmap: normalize_bitmap(bitmap, length),
            length,
        }
    }

    /// Creates an empty [`PlBitmap`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            bitmap: Bitmap::new(),
            length: 0,
        }
    }

    /// Creates a flat [`PlBitmap`] holding the bits of `bitmap`.
    #[inline]
    pub fn from_bitmap(bitmap: Bitmap) -> Self {
        let length = bitmap.len();
        Self { bitmap, length }
    }

    /// Creates a [`PlBitmap`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: bool, length: usize) -> Self {
        Self {
            bitmap: Bitmap::new_with_value(value, scalar_buffer_len(length)),
            length,
        }
    }

    /// The number of bits in this mask.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether this mask holds no bits.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The backing bitmap, if it holds one bit per element.
    #[inline]
    pub fn flat_bitmap(&self) -> Option<&Bitmap> {
        self.is_flat().then_some(&self.bitmap)
    }

    /// Borrows this mask as a [`PlBitmapRef`].
    #[inline]
    pub fn as_ref(&self) -> PlBitmapRef<'_> {
        // SAFETY: the bitmap is flat or scalar for `self.length`, upheld by every constructor.
        unsafe { PlBitmapRef::new_broadcast_unchecked(&self.bitmap, self.length) }
    }

    /// Returns the backing bitmap and the logical length of this mask.
    #[inline]
    pub fn into_inner(self) -> (Bitmap, usize) {
        (self.bitmap, self.length)
    }

    /// Whether the backing bitmap holds a single bit shared by every element.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.bitmap.len() == 1
    }

    /// Whether the backing bitmap holds one bit per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.bitmap.len() == self.length
    }

    /// The bit shared by every element, if the backing bitmap holds a single bit.
    #[inline]
    pub fn scalar_value(&self) -> Option<bool> {
        self.as_ref().scalar_value()
    }

    /// Returns the bit at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.as_ref().get(i)
    }

    /// Returns the bit at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> bool {
        unsafe { self.as_ref().get_unchecked(i) }
    }

    /// The number of unset bits.
    #[inline]
    pub fn unset_bits(&self) -> usize {
        self.as_ref().unset_bits()
    }

    /// The number of set bits.
    #[inline]
    pub fn set_bits(&self) -> usize {
        self.as_ref().set_bits()
    }

    /// Returns an iterator over the bits.
    #[inline]
    pub fn iter(&self) -> PlBitmapIter<'_> {
        self.as_ref().iter()
    }

    /// Slices this mask in place to `length` bits starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.length,
            "the offset of the new slice must be smaller than the length of the mask",
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this mask in place to `length` bits starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // A scalar bitmap is unaffected by slicing — every element reads the same bit — with the
        // one exception of an empty slice, which keeps no element to read it.
        if self.is_flat() {
            unsafe { self.bitmap.slice_unchecked(offset, length) };
        } else if length == 0 {
            unsafe { self.bitmap.slice_unchecked(0, 0) };
        }

        self.length = length;
    }

    /// Returns this mask sliced to `length` bits starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn sliced(mut self, offset: usize, length: usize) -> Self {
        self.slice(offset, length);
        self
    }

    /// Returns this mask sliced to `length` bits starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        unsafe { self.slice_unchecked(offset, length) };
        self
    }

    /// Returns an equivalent mask whose backing bitmap holds one bit per element.
    pub fn to_flat(&self) -> Self {
        Self::from_bitmap(self.as_ref().to_flat())
    }

    /// Consumes this mask into the backing bitmap, keeping the scalar representation.
    #[inline]
    pub fn into_flat_or_scalar(self) -> Bitmap {
        self.bitmap
    }

    /// Consumes this mask into an ordinary [`Bitmap`] holding one bit per element.
    pub fn into_bitmap(self) -> Bitmap {
        if self.is_flat() {
            self.bitmap
        } else {
            Bitmap::new_with_value(self.bitmap.get_bit(0), self.length)
        }
    }
}

impl Default for PlBitmap {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl From<Bitmap> for PlBitmap {
    #[inline]
    fn from(bitmap: Bitmap) -> Self {
        Self::from_bitmap(bitmap)
    }
}

impl From<MutableBitmap> for PlBitmap {
    #[inline]
    fn from(bitmap: MutableBitmap) -> Self {
        Self::from_bitmap(bitmap.into())
    }
}

/// Takes ownership of the backing bitmap, preserving the representation.
impl From<PlBitmapRef<'_>> for PlBitmap {
    #[inline]
    fn from(mask: PlBitmapRef<'_>) -> Self {
        let (bitmap, length) = mask.into_inner();
        // SAFETY: a `PlBitmapRef` upholds the same invariant.
        unsafe { Self::new_broadcast_unchecked(bitmap.clone(), length) }
    }
}

impl FromIterator<bool> for PlBitmap {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self::from_bitmap(Bitmap::from_iter(iter))
    }
}

impl<'a> IntoIterator for &'a PlBitmap {
    type Item = bool;
    type IntoIter = PlBitmapIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two masks bit-wise; the representation (flat or scalar) is irrelevant.
impl PartialEq for PlBitmap {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for PlBitmap {}

impl PartialEq<PlBitmapRef<'_>> for PlBitmap {
    #[inline]
    fn eq(&self, other: &PlBitmapRef<'_>) -> bool {
        self.as_ref() == *other
    }
}

impl PartialEq<PlBitmap> for PlBitmapRef<'_> {
    #[inline]
    fn eq(&self, other: &PlBitmap) -> bool {
        *self == other.as_ref()
    }
}

impl std::fmt::Debug for PlBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        reference::fmt_bits(self.as_ref(), "PlBitmap", f)
    }
}

/// Compares two validity masks over `length` elements, treating an absent mask as all valid.
pub(crate) fn validity_eq(
    lhs: Option<PlBitmapRef<'_>>,
    rhs: Option<PlBitmapRef<'_>>,
    length: usize,
) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        (Some(mask), None) | (None, Some(mask)) => mask.set_bits() == length,
        (None, None) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat() {
        let mask = PlBitmap::from_iter([true, false, true]);

        assert_eq!(mask.len(), 3);
        assert!(mask.is_flat());
        assert!(!mask.is_scalar());
        assert_eq!(mask.scalar_value(), None);
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert_eq!(mask.unset_bits(), 1);
        assert_eq!(mask.set_bits(), 2);
        assert_eq!(mask.iter().collect::<Vec<_>>(), [true, false, true]);
        assert_eq!(mask.to_flat(), mask);
    }

    #[test]
    fn scalar() {
        let mask = PlBitmap::new_scalar(false, 1_000);

        assert_eq!(mask.len(), 1_000);
        assert!(mask.flat_bitmap().is_none());
        assert!(mask.is_scalar());
        assert!(!mask.is_flat());
        assert_eq!(mask.scalar_value(), Some(false));
        assert!(!mask.get(999));
        assert_eq!(mask.unset_bits(), 1_000);
        assert_eq!(mask.set_bits(), 0);
        assert!(mask.iter().all(|bit| !bit));

        let flat = mask.to_flat();
        assert!(flat.is_flat());
        assert_eq!(flat.flat_bitmap().unwrap().len(), 1_000);
        assert_eq!(flat.unset_bits(), 1_000);
        assert_eq!(flat, mask);
    }

    #[test]
    fn into_bitmap_materializes_scalars() {
        let bitmap = PlBitmap::new_scalar(true, 3).into_bitmap();

        assert_eq!(bitmap.len(), 3);
        assert_eq!(bitmap.set_bits(), 3);

        assert!(PlBitmap::new_scalar(true, 0).into_bitmap().is_empty());
    }

    #[test]
    fn a_mask_over_no_elements_keeps_no_bit() {
        // A single bit is scalar for no elements too, but there is no element left to read it, so
        // it is not kept: the mask is flat, like every empty mask, rather than scalar.
        let mask = PlBitmap::new_broadcast(Bitmap::new_zeroed(1), 0);

        assert!(mask.is_empty());
        assert!(mask.is_flat());
        assert!(!mask.is_scalar());
        assert!(mask.flat_bitmap().unwrap().is_empty());
        assert_eq!(mask.scalar_value(), None);
    }
}
