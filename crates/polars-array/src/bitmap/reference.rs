use arrow::bitmap::Bitmap;
use polars_error::{PolarsResult, polars_ensure};

use crate::bitmap::PlBitmapIter;
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_valid_buffer_len,
    normalize_bitmap_ref,
};

/// A borrowed validity mask of `length` bits, in either the flat or the scalar representation.
#[derive(Clone, Copy)]
pub struct PlBitmapRef<'a> {
    /// Scalar: bitmap.len() == 1
    bitmap: &'a Bitmap,
    length: usize,
}

impl<'a> PlBitmapRef<'a> {
    /// Creates a flat [`PlBitmapRef`] of `length` bits backed by `bitmap`.
    ///
    /// # Errors
    /// This function errors unless `bitmap` holds one bit per element.
    pub fn try_new(bitmap: &'a Bitmap, length: usize) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(bitmap.len(), length),
            ComputeError:
            "bitmap of length {} is not flat for a mask of length {}",
            bitmap.len(), length,
        );

        Ok(Self { bitmap, length })
    }

    /// Creates a flat [`PlBitmapRef`] of `length` bits backed by `bitmap`.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(bitmap: &'a Bitmap, length: usize) -> Self {
        Self::try_new(bitmap, length).unwrap()
    }

    /// Creates a flat [`PlBitmapRef`] of `length` bits backed by `bitmap`, without validating it.
    ///
    /// # Safety
    /// `bitmap` must hold one bit per element, i.e. have `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(bitmap: &'a Bitmap, length: usize) -> Self {
        debug_assert!(is_flat_buffer_len(bitmap.len(), length));
        Self { bitmap, length }
    }

    /// Creates a [`PlBitmapRef`] of `length` bits backed by a `bitmap` that broadcasts over them.
    ///
    /// # Errors
    /// This function errors if `bitmap` is neither flat (length equal to `length`) nor scalar
    /// (length one).
    pub fn try_new_broadcast(bitmap: &'a Bitmap, length: usize) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_buffer_len(bitmap.len(), length),
            ComputeError:
            "bitmap of length {} is neither flat nor scalar for a mask of length {}",
            bitmap.len(), length,
        );

        Ok(Self {
            bitmap: normalize_bitmap_ref(bitmap, length),
            length,
        })
    }

    /// Creates a [`PlBitmapRef`] of `length` bits backed by a `bitmap` that broadcasts over them.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(bitmap: &'a Bitmap, length: usize) -> Self {
        Self::try_new_broadcast(bitmap, length).unwrap()
    }

    /// Creates a [`PlBitmapRef`] of `length` bits backed by a `bitmap` that broadcasts over them,
    /// without validating it.
    ///
    /// # Safety
    /// `bitmap` must be flat or scalar for `length`, per [`is_valid_buffer_len`].
    #[inline]
    pub unsafe fn new_broadcast_unchecked(bitmap: &'a Bitmap, length: usize) -> Self {
        debug_assert!(is_valid_buffer_len(bitmap.len(), length));
        Self {
            bitmap: normalize_bitmap_ref(bitmap, length),
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
    pub fn flat_bitmap(&self) -> Option<&'a Bitmap> {
        self.is_flat().then_some(self.bitmap)
    }

    /// Returns the backing bitmap and the logical length of this mask.
    #[inline(always)]
    pub const fn into_inner(self) -> (&'a Bitmap, usize) {
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
        (self.bitmap.len() == 1 && self.length > 0).then(|| self.bitmap.get_bit(0))
    }

    /// Returns the bit at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the bit at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.length);
        unsafe {
            self.bitmap
                .get_bit_unchecked(broadcast_index(i, self.bitmap.len()))
        }
    }

    /// The number of unset bits.
    pub fn unset_bits(&self) -> usize {
        if self.is_scalar() {
            // Every element shares the single bit; an empty mask never reads it.
            if self.bitmap.get_bit(0) {
                0
            } else {
                self.length
            }
        } else {
            self.bitmap.unset_bits()
        }
    }

    /// The number of set bits.
    #[inline]
    pub fn set_bits(&self) -> usize {
        self.length - self.unset_bits()
    }

    /// Materializes an ordinary [`Bitmap`] holding one bit per element.
    pub fn to_flat(&self) -> Bitmap {
        if self.is_flat() {
            self.bitmap.clone()
        } else {
            Bitmap::new_with_value(self.bitmap.get_bit(0), self.length)
        }
    }

    /// This mask as a [`Bitmap`], keeping the scalar representation where it has one.
    #[inline]
    pub fn to_flat_or_scalar(&self) -> Bitmap {
        // The backing bitmap is already flat or scalar for the mask's length, which is exactly
        // what an array accepts as its own mask: hand it over as it is.
        self.bitmap.clone()
    }

    /// Returns this mask over `length` bits, repeating its single bit if that is all it holds.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast(&self, length: usize) -> PlBitmapRef<'a> {
        assert_broadcastable(self.length, length);
        // SAFETY: a mask of one bit is backed by a single bit, which is scalar for any length;
        // otherwise `length` is the length the backing bitmap is already valid for.
        unsafe { PlBitmapRef::new_broadcast_unchecked(self.bitmap, length) }
    }

    /// Returns an iterator over the bits.
    #[inline]
    pub fn iter(&self) -> PlBitmapIter<'a> {
        PlBitmapIter::new(*self)
    }
}

impl<'a> IntoIterator for PlBitmapRef<'a> {
    type Item = bool;
    type IntoIter = PlBitmapIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two masks bit-wise; the representation (flat or scalar) is irrelevant.
impl PartialEq for PlBitmapRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        // Never walk two scalar masks bit by bit: their length is unbounded by their memory use.
        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl Eq for PlBitmapRef<'_> {}

impl std::fmt::Debug for PlBitmapRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_bits(*self, "PlBitmapRef", f)
    }
}

/// Formats `mask` as `name[..]`, without ever materializing a scalar mask.
pub(super) fn fmt_bits(
    mask: PlBitmapRef<'_>,
    name: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    f.write_str(name)?;

    // Never materialize a scalar mask: its length is unbounded by its memory use.
    if mask.is_scalar() && mask.len() > 1 {
        return write!(f, "[{}; {}]", mask.bitmap.get_bit(0), mask.len());
    }

    f.debug_list().entries(mask.iter()).finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat() {
        let bitmap = Bitmap::from_iter([true, false, true]);
        let mask = PlBitmapRef::new(&bitmap, 3);

        assert_eq!(mask.len(), 3);
        assert!(mask.is_flat());
        assert!(!mask.is_scalar());
        assert_eq!(mask.scalar_value(), None);
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert_eq!(mask.unset_bits(), 1);
        assert_eq!(mask.set_bits(), 2);
        assert_eq!(mask.to_flat(), bitmap);
    }

    #[test]
    fn scalar() {
        let bitmap = Bitmap::new_zeroed(1);
        let mask = PlBitmapRef::new_broadcast(&bitmap, 1_000);

        assert_eq!(mask.len(), 1_000);
        assert!(mask.is_scalar());
        assert_eq!(mask.scalar_value(), Some(false));
        assert!(!mask.get(999));
        assert_eq!(mask.unset_bits(), 1_000);
        assert_eq!(mask.set_bits(), 0);

        let flat = mask.to_flat();
        assert_eq!(flat.len(), 1_000);
        assert_eq!(flat.unset_bits(), 1_000);
    }

    #[test]
    fn iterates_both_representations() {
        let bitmap = Bitmap::from_iter([true, false, true]);
        let mask = PlBitmapRef::new(&bitmap, 3);

        assert_eq!(mask.iter().collect::<Vec<_>>(), [true, false, true]);
        assert_eq!(mask.iter().len(), 3);
        assert_eq!(
            mask.into_iter().rev().collect::<Vec<_>>(),
            [true, false, true]
        );

        let bitmap = Bitmap::new_zeroed(1);
        let mask = PlBitmapRef::new_broadcast(&bitmap, 4);

        assert_eq!(mask.iter().collect::<Vec<_>>(), [false; 4]);
        assert_eq!(mask.iter().len(), 4);
    }

    #[test]
    fn a_mask_over_no_elements_borrows_no_bit() {
        // A single bit is scalar for no elements too, but there is no element left to read it, so
        // it is not borrowed: the mask is flat, like every empty mask, rather than scalar.
        let bitmap = Bitmap::new_zeroed(1);
        let mask = PlBitmapRef::new_broadcast(&bitmap, 0);

        assert!(mask.is_empty());
        assert!(mask.is_flat());
        assert!(!mask.is_scalar());
        assert!(mask.flat_bitmap().unwrap().is_empty());

        // Broadcasting a single bit over no elements borrows none either.
        let mask = PlBitmapRef::new_broadcast(&bitmap, 1).broadcast(0);

        assert!(mask.is_flat());
        assert!(!mask.is_scalar());
    }
}
