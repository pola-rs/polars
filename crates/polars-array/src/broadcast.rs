//! The rules governing the scalar (scalar) representation.

use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

/// Iterates the slots a backing buffer holds for an array of `length` elements, in order.
///
/// This is [`broadcast_index`] hoisted out of the loop: which of the two representations the
/// buffer is in is settled once, here, rather than at every element, so that a flat buffer is
/// walked as the slice it already is — which vectorizes — and a scalar one yields the single
/// value it holds `length` times over, without materializing it.
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
///
/// Returns `i` when the buffer holds a slot for every element, and `0` when the buffer is a
/// scalar buffer holding one shared value.
///
/// The result is only in bounds when `i` is a valid element index of an array whose length and
/// buffers satisfy the invariants described in the [module docs](self).
#[inline(always)]
pub const fn broadcast_index(i: usize, buffer_len: usize) -> usize {
    if i < buffer_len { i } else { 0 }
}

/// Whether a backing buffer of length `buffer_len` is *flat* for an array of length `length`.
///
/// A flat buffer holds one slot per element, so [`broadcast_index`] is the identity on it. This is
/// what the non-broadcasting constructors — `try_new` and its companions — require of every
/// backing buffer. See the [module docs](self).
#[inline]
pub const fn is_flat_buffer_len(buffer_len: usize, length: usize) -> bool {
    buffer_len == length
}

/// Whether a backing buffer of length `buffer_len` is *scalar* for an array of length `length`.
///
/// A scalar buffer holds the single value every element shares, so [`broadcast_index`] maps every
/// element onto slot `0`. This is what the broadcasting constructors — `try_new_broadcast` and its
/// companions — require of every backing buffer. An array of no elements reads no slot at all, so
/// it additionally admits an empty buffer. See the [module docs](self).
#[inline]
pub const fn is_scalar_buffer_len(buffer_len: usize, length: usize) -> bool {
    buffer_len == 1 || (length == 0 && buffer_len == 0)
}

/// Whether a backing buffer of length `buffer_len` is valid for an array of length `length`.
///
/// A buffer is valid exactly when it is [flat](is_flat_buffer_len) or
/// [scalar](is_scalar_buffer_len); the two coincide for an array of one element, and both admit
/// the empty buffer of an array of none.
#[inline]
pub const fn is_valid_buffer_len(buffer_len: usize, length: usize) -> bool {
    is_flat_buffer_len(buffer_len, length) || is_scalar_buffer_len(buffer_len, length)
}

/// Whether an array of `length` elements broadcasts to an array of `to_length` elements.
///
/// Broadcasting repeats the single element of an array of length one to any length; an array of
/// any other length only broadcasts to the length it already has. This is
/// [`is_valid_buffer_len`] of the logical lengths — an array broadcasts to `to_length` exactly
/// when its elements would be a valid backing buffer of `to_length` elements. See the [module
/// docs](self).
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

/// Whether an offsets buffer of length `offsets_len` is *flat* for a list or binary array of
/// length `length`.
///
/// The offsets hold one more slot than the starts they begin with — the end of the last element —
/// so this is [`is_flat_buffer_len`] of one slot fewer. An empty buffer is never flat: the end of
/// the last element is needed even when there are no elements. See the [module docs](self).
#[inline]
pub const fn is_flat_offsets_len(offsets_len: usize, length: usize) -> bool {
    match offsets_len.checked_sub(1) {
        Some(starts_len) => is_flat_buffer_len(starts_len, length),
        None => false,
    }
}

/// Whether an offsets buffer of length `offsets_len` is *scalar* for a list or binary array of
/// length `length`.
///
/// This is [`is_scalar_buffer_len`] of the starts, which are one slot fewer than the offsets: two
/// offsets stand for the one range every element shares. An array of no elements covers no range,
/// so it additionally admits the single offset that holds no starts at all. See the [module
/// docs](self).
#[inline]
pub const fn is_scalar_offsets_len(offsets_len: usize, length: usize) -> bool {
    match offsets_len.checked_sub(1) {
        Some(starts_len) => is_scalar_buffer_len(starts_len, length),
        None => false,
    }
}

/// Whether an offsets buffer of length `offsets_len` is valid for a list or binary array of length
/// `length`.
///
/// The offsets are valid exactly when they are [flat](is_flat_offsets_len) or
/// [scalar](is_scalar_offsets_len). An empty buffer is never valid: the end of the last element is
/// needed even when there are no elements. See the [module docs](self).
#[inline]
pub const fn is_valid_offsets_len(offsets_len: usize, length: usize) -> bool {
    is_flat_offsets_len(offsets_len, length) || is_scalar_offsets_len(offsets_len, length)
}

/// Whether a values array of `values_len` values is *flat* for a fixed size list array of `length`
/// elements that are `width` values wide.
///
/// The values hold `width` slots per element rather than one, so this is
/// [`is_flat_buffer_len`] scaled by the width: the values of every element, laid end to end. See
/// the [module docs](self).
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
///
/// A scalar values array holds the one element every element covers, which is `width` values wide.
/// Being empty leaves no element for it to stand for, so a `length` of zero admits only empty
/// values — unlike the buffers above, which an empty array lets stand either way. See the [module
/// docs](self).
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
///
/// The values are valid exactly when they are [flat](is_flat_fixed_size_values_len) or
/// [scalar](is_scalar_fixed_size_values_len). See the [module docs](self).
#[inline]
pub const fn is_valid_fixed_size_values_len(
    values_len: usize,
    width: usize,
    length: usize,
) -> bool {
    is_flat_fixed_size_values_len(values_len, width, length)
        || is_scalar_fixed_size_values_len(values_len, width, length)
}
