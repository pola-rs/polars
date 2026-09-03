//! What a [`PlBooleanArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::{BitmapIter, ZipValidity};

use super::PlBooleanArray;
use crate::flat::Flat;

/// The methods a [`PlBooleanArray`] gains from having one bit per element in every backing bitmap.
///
/// These are the counterparts of the methods on [`BooleanArray`](arrow::array::BooleanArray), whose
/// values bitmap *is* its elements: they hand out the backing bitmaps as they are and read them
/// without a [`broadcast_index`](crate::broadcast::broadcast_index). Each shadows the
/// broadcast-aware method of the same name on [`PlBooleanArray`], which remains reachable through
/// the deref.
impl Flat<PlBooleanArray> {
    /// The values, as an ordinary [`Bitmap`] of exactly [`len`](PlBooleanArray::len) bits.
    ///
    /// Unlike [`PlBooleanArray::values`], this needs no [`PlBitmapRef`](crate::PlBitmapRef) to hide
    /// a scalar bit: bit `i` is element `i`. The values of null elements are undetermined (they can
    /// be anything).
    #[inline(always)]
    pub const fn values(&self) -> &Bitmap {
        &self.0.values
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlBooleanArray::len) bits.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> bool {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.0.length);
        unsafe { self.0.values.get_bit_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.0.length);
        // SAFETY: the mask has one bit per element, so `i` is in bounds of it too.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_bit_unchecked(i) })
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
    pub fn get(&self, i: usize) -> Option<bool> {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<bool> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// This walks the values bitmap directly, so — unlike [`PlBooleanArray::values_iter`] — it is
    /// an ordinary [`BitmapIter`]. The values of null elements are undetermined (they can be
    /// anything).
    #[inline]
    pub fn values_iter(&self) -> BitmapIter<'_> {
        self.0.values.iter()
    }

    /// Returns an iterator over the optional elements.
    ///
    /// This zips the two backing bitmaps directly, so — unlike [`PlBooleanArray::iter`] — it
    /// mirrors [`BooleanArray::iter`](arrow::array::BooleanArray::iter).
    #[inline]
    pub fn iter(&self) -> ZipValidity<bool, BitmapIter<'_>, BitmapIter<'_>> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity())
    }

    /// Consumes this array into its backing bitmaps, which both hold one bit per element.
    ///
    /// The length is not part of the result: it is the length of the values bitmap.
    #[inline]
    pub fn into_inner(self) -> (Bitmap, Option<Bitmap>) {
        let PlBooleanArray {
            values,
            length: _,
            validity,
        } = self.0;

        (values, validity)
    }
}

impl<'a> IntoIterator for &'a Flat<PlBooleanArray> {
    type Item = Option<bool>;
    type IntoIter = ZipValidity<bool, BitmapIter<'a>, BitmapIter<'a>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlBooleanArray> for Flat<PlBooleanArray>`](Flat).
impl PartialEq<Flat<PlBooleanArray>> for PlBooleanArray {
    #[inline]
    fn eq(&self, other: &Flat<PlBooleanArray>) -> bool {
        *self == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_flat_materializes_scalars() {
        let scalar = PlBooleanArray::new_scalar(true, 3);
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().len(), 3);
        assert_eq!(flat.values_iter().collect::<Vec<_>>(), [true; 3]);
        assert_eq!(flat, scalar);

        let null_scalar = PlBooleanArray::new_full_null(3);
        let flat = null_scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
        assert_eq!(flat, null_scalar);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr: PlBooleanArray = [Some(true), None, Some(false)].into_iter().collect();
        let flat = arr.as_flat().expect("the array is flat");

        assert_eq!(flat.values(), arr.flat_values().unwrap());
        assert_eq!(*flat, arr);
        assert_eq!(
            flat.iter().collect::<Vec<_>>(),
            [Some(true), None, Some(false)]
        );

        // Neither a scalar bitmap nor a scalar validity mask can be borrowed as flat.
        assert!(PlBooleanArray::new_scalar(true, 3).as_flat().is_none());
        assert!(
            PlBooleanArray::from_vec(vec![true, false, true])
                .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                .as_flat()
                .is_none()
        );

        // A scalar array of unbounded length is still `O(1)` to reject.
        assert!(
            PlBooleanArray::new_full_null(1_000_000_000)
                .as_flat()
                .is_none()
        );
    }

    #[test]
    fn elements_are_read_without_a_broadcast() {
        let flat = [Some(true), None, Some(false)]
            .into_iter()
            .collect::<PlBooleanArray>()
            .to_flat();

        assert!(flat.value(0));
        assert_eq!(flat.get(0), Some(true));
        assert!(flat.is_valid(0));
        assert!(flat.is_null(1));
        assert_eq!(flat.get(1), None);
        assert_eq!(flat.get(2), Some(false));

        assert!(!unsafe { flat.value_unchecked(2) });
        assert_eq!(unsafe { flat.get_unchecked(1) }, None);
        assert!(unsafe { flat.is_null_unchecked(1) });
    }
}
