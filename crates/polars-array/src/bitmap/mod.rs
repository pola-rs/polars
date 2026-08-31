use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{PolarsResult, polars_ensure};

use crate::broadcast::is_valid_buffer_len;

mod iterator;
mod reference;

pub use iterator::PlBitmapIter;
pub use reference::PlBitmapRef;

/// An immutable, cheaply cloneable mask of `length` bits, in either the flat or the scalar
/// representation.
///
/// This is the owned counterpart of [`PlBitmapRef`]. A [`Bitmap`] always stores one bit per
/// element, so a mask that is constant across a billion elements costs a billion bits to represent.
/// This type pairs a bitmap with the logical `length` it stands for, which lets that constant mask
/// be a single bit: bit `i` reads slot
/// [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index) of the backing bitmap.
/// See [`crate::broadcast`] for the full rules.
///
/// Cloning and slicing are `O(1)`, and so is constructing a mask of arbitrarily many equal bits.
///
/// # Example
/// ```
/// use polars_array::PlBitmap;
///
/// // A constant mask over a billion elements costs a single bit.
/// let mask = PlBitmap::new_scalar(false, 1_000_000_000);
/// assert_eq!(mask.len(), 1_000_000_000);
/// assert_eq!(mask.bitmap().len(), 1);
/// assert!(mask.is_scalar());
/// assert!(!mask.get(999_999_999));
///
/// // Slicing it stays free, and keeps the scalar representation.
/// let mask = mask.sliced(500, 2);
/// assert_eq!(mask.len(), 2);
/// assert_eq!(mask.bitmap().len(), 1);
///
/// // It compares equal to the flat mask it stands for.
/// assert_eq!(mask, PlBitmap::from_iter([false, false]));
/// ```
#[derive(Clone)]
pub struct PlBitmap {
    bitmap: Bitmap,
    length: usize,
}

impl PlBitmap {
    /// Creates a [`PlBitmap`] of `length` bits backed by `bitmap`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors if `bitmap` is neither flat (length equal to `length`) nor scalar
    /// (length one).
    pub fn try_new(bitmap: Bitmap, length: usize) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_buffer_len(bitmap.len(), length),
            ComputeError:
            "bitmap of length {} is neither flat nor scalar for a mask of length {}",
            bitmap.len(), length,
        );

        Ok(Self { bitmap, length })
    }

    /// Creates a [`PlBitmap`] of `length` bits backed by `bitmap`.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(bitmap: Bitmap, length: usize) -> Self {
        Self::try_new(bitmap, length).unwrap()
    }

    /// Creates a [`PlBitmap`] of `length` bits backed by `bitmap`, without validating it.
    ///
    /// # Safety
    /// `bitmap` must be either flat (length equal to `length`) or scalar (length one).
    #[inline]
    pub unsafe fn new_unchecked(bitmap: Bitmap, length: usize) -> Self {
        debug_assert!(is_valid_buffer_len(bitmap.len(), length));
        Self { bitmap, length }
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
            bitmap: Bitmap::new_with_value(value, 1),
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

    /// The backing bitmap.
    ///
    /// This is *not* guaranteed to have [`Self::len`] bits: it is either flat or scalar. Index
    /// it through [`crate::broadcast::broadcast_index`], or call [`Self::to_flat`] first.
    #[inline(always)]
    pub const fn bitmap(&self) -> &Bitmap {
        &self.bitmap
    }

    /// Borrows this mask as a [`PlBitmapRef`].
    #[inline]
    pub fn as_ref(&self) -> PlBitmapRef<'_> {
        // SAFETY: the bitmap is flat or scalar for `self.length`, upheld by every constructor.
        unsafe { PlBitmapRef::new_unchecked(&self.bitmap, self.length) }
    }

    /// Returns the backing bitmap and the logical length of this mask.
    #[inline]
    pub fn into_inner(self) -> (Bitmap, usize) {
        (self.bitmap, self.length)
    }

    /// Whether the backing bitmap holds a single bit shared by every element.
    ///
    /// This is `false` for a flat mask of length one, where the two representations coincide.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.bitmap.len() != self.length
    }

    /// Whether the backing bitmap holds one bit per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        !self.is_scalar()
    }

    /// The bit shared by every element, if the backing bitmap holds a single bit.
    ///
    /// Returns `None` for a flat mask of more than one bit, and for an empty mask. A mask of
    /// length one is both flat and scalar, so it yields its only bit.
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
    ///
    /// This is `O(1)` for a scalar mask and `O(len)` for a flat one, amortized over repeated
    /// calls on the same [`Bitmap`].
    #[inline]
    pub fn unset_bits(&self) -> usize {
        self.as_ref().unset_bits()
    }

    /// The number of set bits.
    ///
    /// This is `O(1)` for a scalar mask and `O(len)` for a flat one, amortized over repeated
    /// calls on the same [`Bitmap`].
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
    /// This function is `O(1)`.
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // A scalar bitmap is unaffected by slicing: every element reads the same bit.
        if self.is_flat() {
            unsafe { self.bitmap.slice_unchecked(offset, length) };
        }

        self.length = length;
    }

    /// Returns this mask sliced to `length` bits starting at `offset`.
    ///
    /// This function is `O(1)`.
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        unsafe { self.slice_unchecked(offset, length) };
        self
    }

    /// Returns an equivalent mask whose backing bitmap holds one bit per element.
    ///
    /// This expands a scalar mask and is therefore `O(len)`; it is a no-op clone when this mask
    /// [`is_flat`](Self::is_flat).
    pub fn to_flat(&self) -> Self {
        Self::from_bitmap(self.as_ref().to_flat())
    }

    /// Consumes this mask into an ordinary [`Bitmap`] holding one bit per element.
    ///
    /// This expands a scalar mask and is therefore `O(len)`; it is free when this mask
    /// [`is_flat`](Self::is_flat).
    pub fn into_bitmap(self) -> Bitmap {
        if self.is_scalar() {
            Bitmap::new_with_value(self.bitmap.get_bit(0), self.length)
        } else {
            self.bitmap
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

/// Takes ownership of the backing bitmap, preserving the representation. This is `O(1)`.
impl From<PlBitmapRef<'_>> for PlBitmap {
    #[inline]
    fn from(mask: PlBitmapRef<'_>) -> Self {
        // SAFETY: a `PlBitmapRef` upholds the same invariant.
        unsafe { Self::new_unchecked(mask.bitmap().clone(), mask.len()) }
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
///
/// This is what the arrays whose elements are not a single value — the nested ones — compare their
/// validity with before looking at what they hold.
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
        assert_eq!(mask.bitmap().len(), 1);
        assert!(mask.is_scalar());
        assert!(!mask.is_flat());
        assert_eq!(mask.scalar_value(), Some(false));
        assert!(!mask.get(999));
        assert_eq!(mask.unset_bits(), 1_000);
        assert_eq!(mask.set_bits(), 0);
        assert!(mask.iter().all(|bit| !bit));

        let flat = mask.to_flat();
        assert!(flat.is_flat());
        assert_eq!(flat.bitmap().len(), 1_000);
        assert_eq!(flat.unset_bits(), 1_000);
        assert_eq!(flat, mask);
    }

    #[test]
    fn empty() {
        let mask = PlBitmap::new_empty();

        assert!(mask.is_empty());
        assert!(mask.is_flat());
        assert_eq!(mask.scalar_value(), None);
        assert_eq!(mask.unset_bits(), 0);
        assert_eq!(mask.set_bits(), 0);
        assert_eq!(mask.iter().next(), None);
        assert_eq!(mask, PlBitmap::default());
        assert_eq!(mask, PlBitmap::new_scalar(true, 0));
    }

    #[test]
    fn length_one_is_both_representations() {
        let mask = PlBitmap::from_iter([true]);

        assert!(mask.is_flat());
        assert_eq!(mask.scalar_value(), Some(true));
        assert!(mask.get(0));
        assert_eq!(mask.set_bits(), 1);
        assert_eq!(mask, PlBitmap::new_scalar(true, 1));
    }

    #[test]
    fn slicing_a_scalar_mask_is_free() {
        let mask = PlBitmap::new_scalar(true, 1_000_000_000).sliced(500, 2);

        assert_eq!(mask.len(), 2);
        assert_eq!(mask.bitmap().len(), 1);
        assert!(mask.is_scalar());
        assert_eq!(mask.iter().collect::<Vec<_>>(), [true, true]);
    }

    #[test]
    fn slicing_a_flat_mask_slices_its_bitmap() {
        let mask = PlBitmap::from_iter([true, false, true, false]).sliced(1, 2);

        assert_eq!(mask.len(), 2);
        assert_eq!(mask.bitmap().len(), 2);
        assert!(mask.is_flat());
        assert_eq!(mask.iter().collect::<Vec<_>>(), [false, true]);
    }

    #[test]
    fn slicing_to_empty() {
        for mut mask in [
            PlBitmap::from_iter([true, false]),
            PlBitmap::new_scalar(true, 2),
        ] {
            mask.slice(1, 0);

            assert!(mask.is_empty());
            assert_eq!(mask.scalar_value(), None);
            assert_eq!(mask.set_bits(), 0);
            assert_eq!(mask, PlBitmap::new_empty());
        }
    }

    #[test]
    #[should_panic(expected = "the offset of the new slice must be smaller")]
    fn slicing_out_of_bounds_panics() {
        PlBitmap::new_scalar(true, 4).slice(3, 2);
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlBitmap::new_scalar(true, 3);
        let flat = PlBitmap::from_iter([true, true, true]);

        assert_eq!(scalar, flat);
        assert_eq!(scalar.as_ref(), flat);
        assert_eq!(flat, scalar.as_ref());
        assert_ne!(scalar, PlBitmap::new_scalar(true, 4));
        assert_ne!(scalar, PlBitmap::from_iter([true, true, false]));
        assert_ne!(scalar, PlBitmap::new_scalar(false, 3));
    }

    #[test]
    fn equality_of_scalars_does_not_walk_bits() {
        // Bit-by-bit comparison of a billion elements would not finish; the fast path must hit.
        let mask = PlBitmap::new_scalar(true, 1_000_000_000);

        assert_eq!(mask, mask.clone());
        assert_ne!(mask, PlBitmap::new_scalar(false, 1_000_000_000));
    }

    #[test]
    fn round_trips_through_a_reference() {
        let bitmap = Bitmap::from_iter([true, false]);
        let mask = PlBitmap::from(PlBitmapRef::new(&bitmap, 2));

        assert_eq!(mask.bitmap(), &bitmap);
        assert_eq!(mask.clone().into_bitmap(), bitmap);
        assert_eq!(mask.into_inner(), (bitmap, 2));

        // Taking ownership of a scalar mask keeps it scalar.
        let bitmap = Bitmap::new_zeroed(1);
        let mask = PlBitmap::from(PlBitmapRef::new(&bitmap, 1_000));

        assert!(mask.is_scalar());
        assert_eq!(mask.len(), 1_000);
        assert_eq!(mask.bitmap().len(), 1);
    }

    #[test]
    fn into_bitmap_materializes_scalars() {
        let bitmap = PlBitmap::new_scalar(true, 3).into_bitmap();

        assert_eq!(bitmap.len(), 3);
        assert_eq!(bitmap.set_bits(), 3);

        assert!(PlBitmap::new_scalar(true, 0).into_bitmap().is_empty());
    }

    #[test]
    fn try_new_rejects_mismatched_bitmaps() {
        assert!(PlBitmap::try_new(Bitmap::new_zeroed(2), 3).is_err());
        assert!(PlBitmap::try_new(Bitmap::new_zeroed(2), 2).is_ok());
        assert!(PlBitmap::try_new(Bitmap::new_zeroed(1), 3).is_ok());
    }

    #[test]
    fn iterator_is_exact_sized_and_double_ended() {
        let mask = PlBitmap::from_iter([true, false, true]);

        assert_eq!(mask.iter().len(), 3);
        assert_eq!(mask.iter().size_hint(), (3, Some(3)));
        assert_eq!(mask.iter().rev().collect::<Vec<_>>(), [true, false, true]);
        assert_eq!(mask.iter().nth(1), Some(false));

        let mask = PlBitmap::new_scalar(true, 5);

        assert_eq!(mask.iter().len(), 5);
        assert_eq!((&mask).into_iter().rev().collect::<Vec<_>>(), [true; 5]);
    }

    #[test]
    fn debug_does_not_materialize_scalars() {
        assert_eq!(
            format!("{:?}", PlBitmap::new_scalar(false, 1_000_000_000)),
            "PlBitmap[false; 1000000000]",
        );
        assert_eq!(
            format!("{:?}", PlBitmap::from_iter([true, false])),
            "PlBitmap[true, false]",
        );
    }
}
