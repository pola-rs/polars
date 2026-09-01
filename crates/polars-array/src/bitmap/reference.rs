use arrow::bitmap::Bitmap;
use polars_error::{PolarsResult, polars_ensure};

use crate::bitmap::PlBitmapIter;
use crate::broadcast::{assert_broadcastable, broadcast_index, is_valid_buffer_len};

/// A borrowed validity mask of `length` bits, in either the flat or the scalar representation.
///
/// A [`Bitmap`] always stores one bit per element, so a mask that is constant across a billion
/// elements costs a billion bits to represent. This type pairs a bitmap with the logical `length`
/// it stands for, which lets that constant mask be a single bit: bit `i` reads slot
/// [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index) of the backing bitmap.
/// See [`crate::broadcast`] for the full rules.
///
/// This is what [`PlPrimitiveArray::validity`](crate::PlPrimitiveArray::validity) hands out, so
/// that reading validity never has to reason about which representation the array happens to be
/// in. Use [`Self::to_flat`] to materialize an ordinary one-bit-per-element [`Bitmap`], or convert
/// it into an owned [`PlBitmap`](crate::PlBitmap) to keep the mask around.
///
/// # Example
/// ```
/// use polars_array::PlPrimitiveArray;
///
/// let arr = PlPrimitiveArray::<i32>::new_full_null(1_000_000_000);
/// let validity = arr.validity().unwrap();
///
/// assert_eq!(validity.len(), 1_000_000_000);
/// assert_eq!(validity.bitmap().len(), 1);
/// assert!(validity.is_scalar());
/// assert!(!validity.get(999_999_999));
/// ```
#[derive(Clone, Copy)]
pub struct PlBitmapRef<'a> {
    bitmap: &'a Bitmap,
    length: usize,
}

impl<'a> PlBitmapRef<'a> {
    /// Creates a [`PlBitmapRef`] of `length` bits backed by `bitmap`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors if `bitmap` is neither flat (length equal to `length`) nor scalar
    /// (length one).
    pub fn try_new(bitmap: &'a Bitmap, length: usize) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_buffer_len(bitmap.len(), length),
            ComputeError:
            "bitmap of length {} is neither flat nor scalar for a mask of length {}",
            bitmap.len(), length,
        );

        Ok(Self { bitmap, length })
    }

    /// Creates a [`PlBitmapRef`] of `length` bits backed by `bitmap`.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(bitmap: &'a Bitmap, length: usize) -> Self {
        Self::try_new(bitmap, length).unwrap()
    }

    /// Creates a [`PlBitmapRef`] of `length` bits backed by `bitmap`, without validating it.
    ///
    /// # Safety
    /// `bitmap` must be either flat (length equal to `length`) or scalar (length one).
    #[inline]
    pub unsafe fn new_unchecked(bitmap: &'a Bitmap, length: usize) -> Self {
        debug_assert!(is_valid_buffer_len(bitmap.len(), length));
        Self { bitmap, length }
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
    pub const fn bitmap(&self) -> &'a Bitmap {
        self.bitmap
    }

    /// Whether the backing bitmap holds a single bit shared by every element.
    ///
    /// A mask of one bit over one element is both scalar and [`flat`](Self::is_flat): the two
    /// representations coincide, and this reports them both.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.bitmap.len() == 1
    }

    /// Whether the backing bitmap holds one bit per element.
    ///
    /// A mask of one bit over one element is both flat and [`scalar`](Self::is_scalar).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.bitmap.len() == self.length
    }

    /// The bit shared by every element, if the backing bitmap holds a single bit.
    ///
    /// Returns `None` for a flat mask of more than one bit, and for an empty mask. A mask of
    /// length one is both flat and scalar, so it yields its only bit.
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
    ///
    /// This is `O(1)` for a scalar mask and `O(len)` for a flat one, amortized over repeated
    /// calls on the same [`Bitmap`].
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
    ///
    /// This is `O(1)` for a scalar mask and `O(len)` for a flat one, amortized over repeated
    /// calls on the same [`Bitmap`].
    #[inline]
    pub fn set_bits(&self) -> usize {
        self.length - self.unset_bits()
    }

    /// Materializes an ordinary [`Bitmap`] holding one bit per element.
    ///
    /// This expands a scalar mask and is therefore `O(len)`; it is a no-op clone when this mask
    /// [`is_flat`](Self::is_flat).
    pub fn to_flat(&self) -> Bitmap {
        if self.is_flat() {
            self.bitmap.clone()
        } else {
            Bitmap::new_with_value(self.bitmap.get_bit(0), self.length)
        }
    }

    /// Returns this mask over `length` bits, repeating its single bit if that is all it holds.
    ///
    /// This mask either has `length` bits, in which case this returns it unchanged, or a single
    /// bit, which the `length` bits of the result then all read. Either way this is `O(1)`: the
    /// backing bitmap is the one this mask already has.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast(&self, length: usize) -> PlBitmapRef<'a> {
        assert_broadcastable(self.length, length);
        // SAFETY: a mask of one bit is backed by a single bit, which is scalar for any length;
        // otherwise `length` is the length the backing bitmap is already valid for.
        unsafe { PlBitmapRef::new_unchecked(self.bitmap, length) }
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
        return write!(f, "[{}; {}]", mask.bitmap().get_bit(0), mask.len());
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
        let mask = PlBitmapRef::new(&bitmap, 1_000);

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
    fn length_one_is_both_representations() {
        let bitmap = Bitmap::new_with_value(true, 1);
        let mask = PlBitmapRef::new(&bitmap, 1);

        assert!(mask.is_flat());
        assert_eq!(mask.scalar_value(), Some(true));
        assert!(mask.get(0));
        assert_eq!(mask.set_bits(), 1);
        assert_eq!(mask.to_flat(), bitmap);
    }

    #[test]
    fn empty() {
        let bitmap = Bitmap::new();
        let mask = PlBitmapRef::new(&bitmap, 0);

        assert!(mask.is_empty());
        assert!(mask.is_flat());
        assert_eq!(mask.scalar_value(), None);
        assert_eq!(mask.unset_bits(), 0);
        assert!(mask.to_flat().is_empty());

        // A one-bit bitmap is a valid backing for an empty mask; it is simply never read.
        let bitmap = Bitmap::new_zeroed(1);
        let mask = PlBitmapRef::new(&bitmap, 0);

        assert!(mask.is_empty());
        assert_eq!(mask.scalar_value(), None);
        assert_eq!(mask.unset_bits(), 0);
        assert!(mask.to_flat().is_empty());
    }

    #[test]
    fn try_new_rejects_mismatched_bitmaps() {
        let bitmap = Bitmap::new_zeroed(2);

        assert!(PlBitmapRef::try_new(&bitmap, 3).is_err());
        assert!(PlBitmapRef::try_new(&bitmap, 2).is_ok());
        assert!(PlBitmapRef::try_new(&Bitmap::new_zeroed(1), 3).is_ok());
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
        let mask = PlBitmapRef::new(&bitmap, 4);

        assert_eq!(mask.iter().collect::<Vec<_>>(), [false; 4]);
        assert_eq!(mask.iter().len(), 4);
    }

    #[test]
    fn broadcasting_a_single_bit() {
        let bitmap = Bitmap::new_with_value(true, 1);
        let mask = PlBitmapRef::new(&bitmap, 1);

        let broadcast = mask.broadcast(1_000_000_000);
        assert_eq!(broadcast.len(), 1_000_000_000);
        assert_eq!(broadcast.bitmap().len(), 1);
        assert_eq!(broadcast.set_bits(), 1_000_000_000);
        assert_eq!(broadcast.iter().take(3).collect::<Vec<_>>(), [true; 3]);

        // A mask of the length asked for is handed back as it is, whatever it is backed by.
        let flat = Bitmap::from_iter([true, false, true]);
        let mask = PlBitmapRef::new(&flat, 3);
        assert_eq!(mask.broadcast(3), mask);
    }

    #[test]
    #[should_panic(expected = "does not broadcast")]
    fn broadcasting_more_than_one_bit_panics() {
        let bitmap = Bitmap::from_iter([true, false]);
        let _ = PlBitmapRef::new(&bitmap, 2).broadcast(4);
    }

    #[test]
    fn equality_ignores_representation() {
        let flat = Bitmap::from_iter([false, false, false]);
        let scalar = Bitmap::new_zeroed(1);

        assert_eq!(PlBitmapRef::new(&flat, 3), PlBitmapRef::new(&scalar, 3),);
        assert_ne!(PlBitmapRef::new(&flat, 3), PlBitmapRef::new(&scalar, 4),);
        assert_ne!(
            PlBitmapRef::new(&flat, 3),
            PlBitmapRef::new(&Bitmap::new_with_value(true, 1), 3),
        );
    }

    #[test]
    fn debug_does_not_materialize_scalars() {
        let bitmap = Bitmap::new_zeroed(1);
        assert_eq!(
            format!("{:?}", PlBitmapRef::new(&bitmap, 1_000_000_000)),
            "PlBitmapRef[false; 1000000000]",
        );

        let bitmap = Bitmap::from_iter([true, false]);
        assert_eq!(
            format!("{:?}", PlBitmapRef::new(&bitmap, 2)),
            "PlBitmapRef[true, false]",
        );
    }
}
