//! The rules governing the scalar (scalar) representation.

use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

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
#[inline]
pub const fn is_scalar_buffer_len(buffer_len: usize, length: usize) -> bool {
    buffer_len == 1 || (length == 0 && buffer_len == 0)
}

/// The number of slots a scalar backing buffer holds for an array of `length` elements.
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
    if length == 0 {
        values_len == 0
    } else {
        values_len == width
    }
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
