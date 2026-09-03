//! What a [`PlBinaryViewArray`] gains from being known to be [`Flat`].

use arrow::array::View;
use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::{PlBinaryViewArray, PlBinaryViewIter};
use crate::flat::Flat;

/// The methods a [`PlBinaryViewArray`] gains from having one slot per element in its views and its
/// validity mask.
///
/// These are the counterparts of the methods on
/// [`BinaryViewArray`](arrow::array::BinaryViewArray), whose views *are* its elements: they hand
/// out the views and the mask as they are and read them without a
/// [`broadcast_index`](crate::broadcast::broadcast_index). Each shadows the broadcast-aware method
/// of the same name on [`PlBinaryViewArray`], which remains reachable through the deref.
///
/// The data buffers are not among them: they are indexed by the views rather than by an element
/// index, so being flat says nothing about them and [`PlBinaryViewArray::data_buffers`] is the
/// only way to reach them, flat or not.
impl Flat<PlBinaryViewArray> {
    /// The backing views buffer, holding exactly [`len`](PlBinaryViewArray::len) slots.
    ///
    /// Unlike [`PlBinaryViewArray::flat_views`], this needs no [`Option`] to admit a scalar views
    /// buffer: it is guaranteed to hold one view per element, so slot
    /// `i` is element `i`. The views of null elements are undetermined (they can be anything that
    /// reads bytes the array holds).
    #[inline(always)]
    pub const fn views(&self) -> &Buffer<View> {
        &self.0.views
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlBinaryViewArray::len) bits.
    ///
    /// Unlike [`PlBinaryViewArray::validity`], this needs no [`PlBitmapRef`](crate::PlBitmapRef)
    /// to hide a scalar bit: bit `i` is element `i`.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Returns the view of the element at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn view(&self, i: usize) -> View {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.view_unchecked(i) }
    }

    /// Returns the view of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn view_unchecked(&self, i: usize) -> View {
        debug_assert!(i < self.0.length);
        unsafe { *self.0.views.get_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
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
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.0.length);
        // SAFETY: every view reads bytes the data buffers hold, upheld by every constructor.
        unsafe {
            self.0
                .views
                .get_unchecked(i)
                .get_slice_unchecked(self.0.buffers.as_slice())
        }
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
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// Consumes this array into its views, the data buffers they read, and its validity mask.
    ///
    /// The length is not part of the result: it is the length of the views. The data buffers are
    /// handed out as they are, since they never held one slot per element to begin with.
    #[inline]
    pub fn into_inner(self) -> (Buffer<View>, Buffer<Buffer<u8>>, Option<Bitmap>) {
        let PlBinaryViewArray {
            views,
            buffers,
            length: _,
            validity,
        } = self.0;

        (views, buffers, validity)
    }
}

impl<'a> IntoIterator for &'a Flat<PlBinaryViewArray> {
    type Item = Option<&'a [u8]>;
    type IntoIter = PlBinaryViewIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlBinaryViewArray> for Flat<PlBinaryViewArray>`](Flat).
impl PartialEq<Flat<PlBinaryViewArray>> for PlBinaryViewArray {
    #[inline]
    fn eq(&self, other: &Flat<PlBinaryViewArray>) -> bool {
        *self == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value of more than [`View::MAX_INLINE_SIZE`] bytes, which no view inlines.
    const LONG: &[u8] = b"a value that is too long to inline";

    #[test]
    fn to_flat_materializes_scalars() {
        let scalar = PlBinaryViewArray::new_scalar(LONG, 3);
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.views().len(), 3);
        assert_eq!(flat, scalar);

        // Only the views are written out: the bytes stay in the buffer they are already in.
        assert!(flat.data_buffers().is_same_buffer(scalar.data_buffers()));
        assert_eq!(flat.total_buffer_len(), LONG.len());

        let null_scalar = PlBinaryViewArray::new_full_null(3);
        let flat = null_scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
        assert_eq!(flat, null_scalar);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr: PlBinaryViewArray = [Some(b"foo".as_slice()), None, Some(LONG)]
            .into_iter()
            .collect();
        let flat = arr.as_flat().expect("the array is flat");

        assert_eq!(flat.value(0), b"foo");
        assert_eq!(*flat, arr);
        assert!(
            flat.views().is_same_buffer(arr.flat_views().unwrap()),
            "the views buffer must be borrowed, not materialized again",
        );

        // Neither a scalar views buffer nor a scalar validity mask can be borrowed as flat.
        assert!(PlBinaryViewArray::new_scalar(b"foo", 3).as_flat().is_none());
        assert!(
            PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar", b"baz"])
                .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                .as_flat()
                .is_none()
        );

        // A scalar array of unbounded length is still `O(1)` to reject.
        assert!(
            PlBinaryViewArray::new_full_null(1_000_000_000)
                .as_flat()
                .is_none()
        );
    }

    #[test]
    fn elements_are_read_without_a_broadcast() {
        let flat = [Some(b"foo".as_slice()), None, Some(LONG)]
            .into_iter()
            .collect::<PlBinaryViewArray>()
            .to_flat();

        assert_eq!(flat.value(0), b"foo");
        assert_eq!(flat.get(0), Some(b"foo".as_slice()));
        assert!(flat.is_valid(0));
        assert!(flat.is_null(1));
        assert_eq!(flat.get(1), None);
        assert_eq!(flat.get(2), Some(LONG));

        assert!(flat.view(0).is_inline());
        assert!(!flat.view(2).is_inline());

        assert_eq!(unsafe { flat.value_unchecked(2) }, LONG);
        assert_eq!(unsafe { flat.get_unchecked(1) }, None);
        assert!(unsafe { flat.is_null_unchecked(1) });
        assert_eq!(unsafe { flat.view_unchecked(0) }, flat.view(0));

        // The mask is an ordinary bitmap of one bit per element.
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert!(
            PlBinaryViewArray::from_values_iter([b"foo".as_slice()])
                .to_flat()
                .validity()
                .is_none()
        );
    }
}
