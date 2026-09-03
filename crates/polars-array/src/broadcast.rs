//! The rules governing the flat and scalar representations.
//!
//! Every backing buffer of an array is in one of two representations for the array's length:
//!
//! - *flat*: the buffer holds one slot per element;
//! - *scalar*: the buffer holds a single slot that every element reads.
//!
//! An empty array is flat, and never scalar: there is no element for a single slot to be shared
//! by, so every buffer of an empty array is empty as well. The broadcasting constructors and
//! setters still accept a single slot for an array of no elements — a scalar over `n` elements is
//! a scalar over `0` of them too — but store the empty buffer in its place, so that no array is
//! ever backed by a slot none of its elements reads. That is what the `normalize_*` functions of
//! this module do, and what the `slice_*` functions keep holding as an array is sliced down to
//! nothing.

use std::sync::LazyLock;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

use crate::array::PlArray;

/// Iterates the slots a backing buffer holds for an array of `length` elements, in order.
///
/// # Panics
/// Panics unless `buffer` is [flat](is_flat_buffer_len) or [scalar](is_scalar_buffer_len) for
/// `length`, which every array of this crate upholds of every buffer it is backed by.
#[inline]
pub(crate) fn broadcast_slice<T>(buffer: &[T], length: usize) -> SliceBroadcastIter<'_, T> {
    SliceBroadcastIter::new_broadcast(buffer, length).unwrap_or_else(|| {
        panic!(
            "a buffer of length {} is neither flat nor scalar for an array of length {length}",
            buffer.len(),
        )
    })
}

/// Maps a logical element index onto a slot in a backing buffer of length `buffer_len`.
#[inline(always)]
pub const fn broadcast_index(i: usize, buffer_len: usize) -> usize {
    if i < buffer_len { i } else { 0 }
}

/// Whether a backing buffer of length `buffer_len` is *flat* for an array of length `length`.
#[inline]
pub const fn is_flat_buffer_len(buffer_len: usize, length: usize) -> bool {
    buffer_len == length
}

/// Whether a backing buffer of length `buffer_len` is *scalar* for an array of length `length`.
///
/// A single slot is scalar for any length, and no slot at all is scalar for an array of no
/// elements, which is what such an array stores: see the [module docs](self).
#[inline]
pub const fn is_scalar_buffer_len(buffer_len: usize, length: usize) -> bool {
    buffer_len == 1 || (length == 0 && buffer_len == 0)
}

/// The number of slots a scalar backing buffer holds for an array of `length` elements: one, or
/// none at all for an array of no elements.
#[inline]
pub const fn scalar_buffer_len(length: usize) -> usize {
    (length > 0) as usize
}

/// The number of offsets a scalar list or binary array of `length` elements holds.
#[inline]
pub const fn scalar_offsets_len(length: usize) -> usize {
    scalar_buffer_len(length) + 1
}

/// Whether a backing buffer of length `buffer_len` is valid for an array of length `length`.
#[inline]
pub const fn is_valid_buffer_len(buffer_len: usize, length: usize) -> bool {
    is_flat_buffer_len(buffer_len, length) || is_scalar_buffer_len(buffer_len, length)
}

/// Whether an array of `length` elements broadcasts to an array of `to_length` elements.
#[inline]
pub const fn is_broadcastable(length: usize, to_length: usize) -> bool {
    is_valid_buffer_len(length, to_length)
}

/// Panics unless an array of `length` elements broadcasts to `to_length` elements.
#[inline]
pub(crate) fn assert_broadcastable(length: usize, to_length: usize) {
    assert!(
        is_broadcastable(length, to_length),
        "an array of length {length} does not broadcast to length {to_length}",
    );
}

/// Whether an offsets buffer of length `offsets_len` is *flat* for a list or binary array of length
/// `length`.
#[inline]
pub const fn is_flat_offsets_len(offsets_len: usize, length: usize) -> bool {
    match offsets_len.checked_sub(1) {
        Some(starts_len) => is_flat_buffer_len(starts_len, length),
        None => false,
    }
}

/// Whether an offsets buffer of length `offsets_len` is *scalar* for a list or binary array of
/// length `length`.
#[inline]
pub const fn is_scalar_offsets_len(offsets_len: usize, length: usize) -> bool {
    match offsets_len.checked_sub(1) {
        Some(starts_len) => is_scalar_buffer_len(starts_len, length),
        None => false,
    }
}

/// Whether an offsets buffer of length `offsets_len` is valid for a list or binary array of length
/// `length`.
#[inline]
pub const fn is_valid_offsets_len(offsets_len: usize, length: usize) -> bool {
    is_flat_offsets_len(offsets_len, length) || is_scalar_offsets_len(offsets_len, length)
}

/// Whether a values array of `values_len` values is *flat* for a fixed size list array of `length`
/// elements that are `width` values wide.
#[inline]
pub const fn is_flat_fixed_size_values_len(values_len: usize, width: usize, length: usize) -> bool {
    match length.checked_mul(width) {
        Some(flat_len) => values_len == flat_len,
        // A flat values array that overflows a `usize` is longer than any buffer can be.
        None => false,
    }
}

/// Whether a values array of `values_len` values is *scalar* for a fixed size list array of
/// `length` elements that are `width` values wide.
#[inline]
pub const fn is_scalar_fixed_size_values_len(
    values_len: usize,
    width: usize,
    length: usize,
) -> bool {
    // One element is scalar for any length, and no element at all is scalar for an array of no
    // elements, which is what such an array stores: see the module docs.
    values_len == width || (length == 0 && values_len == 0)
}

/// Whether a values array of `values_len` values is valid for a fixed size list array of `length`
/// elements that are `width` values wide.
#[inline]
pub const fn is_valid_fixed_size_values_len(
    values_len: usize,
    width: usize,
    length: usize,
) -> bool {
    is_flat_fixed_size_values_len(values_len, width, length)
        || is_scalar_fixed_size_values_len(values_len, width, length)
}

/// An empty [`Bitmap`] that lives for the whole program, to borrow where a mask over no elements
/// is called for.
pub(crate) fn empty_bitmap() -> &'static Bitmap {
    static EMPTY: LazyLock<Bitmap> = LazyLock::new(Bitmap::new);
    &EMPTY
}

/// The buffer an array of `length` elements stores for `buffer`, which is flat or scalar for it.
///
/// An array of no elements keeps no slot: the single one a scalar buffer holds is dropped.
#[inline]
pub(crate) fn normalize_buffer<T>(buffer: Buffer<T>, length: usize) -> Buffer<T> {
    if length == 0 && !buffer.is_empty() {
        Buffer::new()
    } else {
        buffer
    }
}

/// The bitmap an array of `length` elements stores for `bitmap`, which is flat or scalar for it.
///
/// An array of no elements keeps no bit: the single one a scalar bitmap holds is dropped.
#[inline]
pub(crate) fn normalize_bitmap(bitmap: Bitmap, length: usize) -> Bitmap {
    if length == 0 && !bitmap.is_empty() {
        Bitmap::new()
    } else {
        bitmap
    }
}

/// The bitmap a mask of `length` bits borrows for `bitmap`, which is flat or scalar for it.
///
/// A mask over no elements borrows no bit: the single one a scalar bitmap holds is passed over for
/// an empty bitmap.
#[inline]
pub(crate) fn normalize_bitmap_ref(bitmap: &Bitmap, length: usize) -> &Bitmap {
    if length == 0 && !bitmap.is_empty() {
        empty_bitmap()
    } else {
        bitmap
    }
}

/// The validity mask an array of `length` elements stores for `validity`, which is flat or scalar
/// for it, as [`normalize_bitmap`].
#[inline]
pub(crate) fn normalize_validity(validity: Option<Bitmap>, length: usize) -> Option<Bitmap> {
    validity.map(|validity| normalize_bitmap(validity, length))
}

/// The offsets a list or binary array of `length` elements stores for `offsets`, which are flat or
/// scalar for it.
///
/// An array of no elements keeps the one offset that holds no starts, exactly as slicing such an
/// array down to nothing does: the range every element of a scalar array shares has no element
/// left to share it.
#[inline]
pub(crate) fn normalize_offsets(offsets: Buffer<u64>, length: usize) -> Buffer<u64> {
    if length == 0 && offsets.len() != 1 {
        offsets.sliced(0..1)
    } else {
        offsets
    }
}

/// The values a fixed size list array of `length` elements stores for `values`, which are flat or
/// scalar for it.
///
/// An array of no elements covers no values: the one element a scalar values array holds is
/// sliced away.
#[inline]
pub(crate) fn normalize_values(mut values: Box<dyn PlArray>, length: usize) -> Box<dyn PlArray> {
    if length == 0 && !values.is_empty() {
        values.slice(0, 0);
    }
    values
}

/// Slices a backing buffer that is flat or scalar for an array of `array_len` elements down to the
/// `length` slots at `offset`.
///
/// A scalar buffer is left as it is — every element of the slice reads the same slot as every
/// element of the array does — unless the slice is empty, which leaves no element to read it.
///
/// # Safety
/// `offset + length` must not exceed `array_len`.
#[inline]
pub(crate) unsafe fn slice_buffer<T>(
    buffer: &mut Buffer<T>,
    array_len: usize,
    offset: usize,
    length: usize,
) {
    if is_flat_buffer_len(buffer.len(), array_len) {
        unsafe { buffer.slice_in_place_unchecked(offset..offset + length) };
    } else if length == 0 {
        unsafe { buffer.slice_in_place_unchecked(0..0) };
    }
}

/// Slices a backing bitmap that is flat or scalar for an array of `array_len` elements down to the
/// `length` bits at `offset`, as [`slice_buffer`].
///
/// # Safety
/// `offset + length` must not exceed `array_len`.
#[inline]
pub(crate) unsafe fn slice_bitmap(
    bitmap: &mut Bitmap,
    array_len: usize,
    offset: usize,
    length: usize,
) {
    if is_flat_buffer_len(bitmap.len(), array_len) {
        unsafe { bitmap.slice_unchecked(offset, length) };
    } else if length == 0 {
        unsafe { bitmap.slice_unchecked(0, 0) };
    }
}

/// Slices the validity mask of an array of `array_len` elements down to the `length` bits at
/// `offset`, as [`slice_bitmap`].
///
/// # Safety
/// `offset + length` must not exceed `array_len`.
#[inline]
pub(crate) unsafe fn slice_validity(
    validity: &mut Option<Bitmap>,
    array_len: usize,
    offset: usize,
    length: usize,
) {
    if let Some(validity) = validity.as_mut() {
        unsafe { slice_bitmap(validity, array_len, offset, length) };
    }
}

/// Slices the offsets of a list or binary array of `array_len` elements down to the `length`
/// elements at `offset`.
///
/// The values the offsets point into are left as they are: nothing normalizes them, so what falls
/// outside the slice simply stops being reachable. Scalar offsets are left alone as well, like a
/// scalar mask, unless the slice is empty: the one offset that holds no starts is then all it
/// keeps of the range every element of the array shares.
///
/// # Safety
/// `offset + length` must not exceed `array_len`.
#[inline]
pub(crate) unsafe fn slice_offsets(
    offsets: &mut Buffer<u64>,
    array_len: usize,
    offset: usize,
    length: usize,
) {
    if is_flat_offsets_len(offsets.len(), array_len) {
        unsafe { offsets.slice_in_place_unchecked(offset..offset + length + 1) };
    } else if length == 0 {
        unsafe { offsets.slice_in_place_unchecked(0..1) };
    }
}

/// Slices a backing buffer of a fixed size array of `array_len` elements that are `width` slots
/// wide down to the `length` elements at `offset`, a width at a time.
///
/// A scalar buffer is left as it is — every element of the slice covers the same slots as every
/// element of the array does — unless the slice is empty, which leaves no element to cover them.
///
/// # Safety
/// `offset + length` must not exceed `array_len`.
#[inline]
pub(crate) unsafe fn slice_fixed_size_buffer<T>(
    buffer: &mut Buffer<T>,
    width: usize,
    array_len: usize,
    offset: usize,
    length: usize,
) {
    if is_flat_fixed_size_values_len(buffer.len(), width, array_len) {
        unsafe { buffer.slice_in_place_unchecked(offset * width..(offset + length) * width) };
    } else if length == 0 {
        unsafe { buffer.slice_in_place_unchecked(0..0) };
    }
}

/// Slices the values array of a fixed size list array of `array_len` elements that are `width`
/// values wide down to the `length` elements at `offset`, as [`slice_fixed_size_buffer`].
///
/// # Safety
/// `offset + length` must not exceed `array_len`.
#[inline]
pub(crate) unsafe fn slice_fixed_size_values(
    values: &mut Box<dyn PlArray>,
    width: usize,
    array_len: usize,
    offset: usize,
    length: usize,
) {
    if is_flat_fixed_size_values_len(values.len(), width, array_len) {
        unsafe { values.slice_unchecked(offset * width, length * width) };
    } else if length == 0 {
        unsafe { values.slice_unchecked(0, 0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slicing leaves a scalar buffer alone — every element of the slice reads the same slot — but
    /// an empty slice has no element left to read it, so the slot goes.
    #[test]
    fn slicing_to_nothing_drops_the_scalar_slot() {
        unsafe {
            let mut values = Buffer::from(vec![7i32]);
            slice_buffer(&mut values, 3, 1, 2);
            assert_eq!(values.as_slice(), [7]);
            slice_buffer(&mut values, 3, 1, 0);
            assert!(values.is_empty());

            let mut values = Buffer::from(vec![1i32, 2, 3]);
            slice_buffer(&mut values, 3, 1, 2);
            assert_eq!(values.as_slice(), [2, 3]);

            let mut bitmap = Bitmap::new_zeroed(1);
            slice_bitmap(&mut bitmap, 3, 1, 2);
            assert_eq!(bitmap.len(), 1);
            slice_bitmap(&mut bitmap, 3, 1, 0);
            assert!(bitmap.is_empty());

            let mut validity = Some(Bitmap::new_zeroed(1));
            slice_validity(&mut validity, 3, 0, 0);
            assert!(validity.unwrap().is_empty());
            let mut validity = None;
            slice_validity(&mut validity, 3, 0, 0);
            assert!(validity.is_none());

            let mut offsets = Buffer::from(vec![2u64, 5]);
            slice_offsets(&mut offsets, 3, 1, 2);
            assert_eq!(offsets.as_slice(), [2, 5]);
            slice_offsets(&mut offsets, 3, 1, 0);
            assert_eq!(offsets.as_slice(), [2]);

            let mut offsets = Buffer::from(vec![0u64, 1, 2, 3]);
            slice_offsets(&mut offsets, 3, 1, 2);
            assert_eq!(offsets.as_slice(), [1, 2, 3]);

            let mut values = Buffer::from(vec![1u8, 2]);
            slice_fixed_size_buffer(&mut values, 2, 3, 1, 2);
            assert_eq!(values.as_slice(), [1, 2]);
            slice_fixed_size_buffer(&mut values, 2, 3, 1, 0);
            assert!(values.is_empty());

            let mut values = Buffer::from(vec![1u8, 2, 3, 4, 5, 6]);
            slice_fixed_size_buffer(&mut values, 2, 3, 1, 2);
            assert_eq!(values.as_slice(), [3, 4, 5, 6]);

            let element = || -> Box<dyn PlArray> {
                Box::new(crate::PlPrimitiveArray::from_vec(vec![1i32, 2]))
            };
            let mut values = element();
            slice_fixed_size_values(&mut values, 2, 3, 1, 2);
            assert_eq!(values.len(), 2);
            slice_fixed_size_values(&mut values, 2, 3, 1, 0);
            assert!(values.is_empty());

            let mut values: Box<dyn PlArray> =
                Box::new(crate::PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6]));
            slice_fixed_size_values(&mut values, 2, 3, 1, 2);
            assert_eq!(values.len(), 4);
        }
    }

    #[test]
    fn a_single_slot_is_scalar_for_no_elements_but_is_not_kept() {
        assert!(is_scalar_buffer_len(1, 0));
        assert!(is_scalar_buffer_len(0, 0));
        assert!(!is_scalar_buffer_len(2, 0));
        assert_eq!(scalar_buffer_len(0), 0);

        assert!(is_scalar_offsets_len(2, 0));
        assert!(is_scalar_offsets_len(1, 0));
        assert!(!is_scalar_offsets_len(0, 0));

        assert!(is_scalar_fixed_size_values_len(2, 2, 0));
        assert!(is_scalar_fixed_size_values_len(0, 2, 0));
        assert!(!is_scalar_fixed_size_values_len(4, 2, 0));
        assert!(!is_scalar_fixed_size_values_len(0, 2, 3));

        assert!(normalize_buffer(Buffer::from(vec![7i32]), 0).is_empty());
        assert_eq!(normalize_buffer(Buffer::from(vec![7i32]), 3).len(), 1);
        assert!(normalize_bitmap(Bitmap::new_zeroed(1), 0).is_empty());
        assert!(normalize_bitmap_ref(&Bitmap::new_zeroed(1), 0).is_empty());
        assert_eq!(
            normalize_validity(Some(Bitmap::new_zeroed(1)), 5)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            normalize_offsets(Buffer::from(vec![2u64, 5]), 0).as_slice(),
            [2]
        );
        assert_eq!(
            normalize_offsets(Buffer::from(vec![2u64, 5]), 3).as_slice(),
            [2, 5]
        );
    }
}
