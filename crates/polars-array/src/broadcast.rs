//! The rules governing the scalar (scalar) representation.
//!
//! Every array in this crate stores its logical length in a dedicated `length` field, decoupled
//! from the lengths of its backing buffers. A backing buffer is read through
//! [`broadcast_index`]: element `i` of the array reads slot `i` of the buffer if the buffer is long
//! enough, and slot `0` otherwise.
//!
//! That leaves exactly two admissible states per buffer, which every constructor validates:
//!
//! * *flat*: `buffer.len() == length`, the usual one-slot-per-element layout.
//! * *scalar*: `buffer.len() == 1`, a single value shared by all `length` elements.
//!
//! A buffer of length one is simultaneously flat and scalar when `length == 1`; the two
//! interpretations agree, so this is not ambiguous. The predicates of the arrays report that
//! faithfully: an array of one element is both `is_flat` and `is_scalar`, since a single element
//! is a single shared value. A `length` of zero admits an empty buffer (flat) as well as a
//! one-element buffer (scalar), and neither is ever read.
//!
//! Intermediate buffer lengths (`1 < buffer.len() < length`) are *not* valid, even though
//! [`broadcast_index`] would happily map them: an array is either flat or scalar.
//!
//! # The offsets of a list array
//!
//! The offsets of a [`PlListArray`](crate::PlListArray) are the one backing buffer that does not
//! hold one slot per element when it is flat: element `i` covers the range
//! `offsets[i]..offsets[i + 1]`, so the buffer holds the start of every element plus the end of the
//! last. It is the *starts* that are flat or scalar, and they are one slot shorter than the buffer:
//!
//! * *flat*: `offsets.len() == length + 1`, one range per element, laid end to end.
//! * *scalar*: `offsets.len() == 2`, a single range shared by all `length` elements.
//!
//! Such a buffer is read through [`broadcast_index(i, offsets.len() - 1)`](broadcast_index), and
//! validated with [`is_valid_offsets_len`].
//!
//! # The values of a fixed size list array
//!
//! The values of a [`PlFixedSizeListArray`](crate::PlFixedSizeListArray) are the other backing
//! buffer that does not hold one slot per element: element `i` covers `width` of them at a time.
//! It is therefore the *elements* the buffer holds that are flat or scalar, and each of them is
//! `width` slots wide:
//!
//! * *flat*: `values.len() == length * width`, the values of every element laid end to end.
//! * *scalar*: `values.len() == width`, the one element all `length` elements share.
//!
//! Such a buffer is read at `broadcast_index(i, length) * width`, and validated with
//! [`is_valid_fixed_size_values_len`]. Being empty leaves no element for a scalar values array to
//! stand for, so a `length` of zero admits only empty values, unlike the buffers above.

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

/// Whether a backing buffer of length `buffer_len` is valid for an array of length `length`.
#[inline]
pub const fn is_valid_buffer_len(buffer_len: usize, length: usize) -> bool {
    buffer_len == length || buffer_len == 1
}

/// Whether an offsets buffer of length `offsets_len` is valid for a list array of length `length`.
///
/// The offsets hold one more slot than the starts they begin with — the end of the last element —
/// so this is [`is_valid_buffer_len`] of one slot fewer. An empty buffer is never valid: the end of
/// the last element is needed even when there are no elements. See the [module docs](self).
#[inline]
pub const fn is_valid_offsets_len(offsets_len: usize, length: usize) -> bool {
    match offsets_len.checked_sub(1) {
        Some(starts_len) => is_valid_buffer_len(starts_len, length),
        None => false,
    }
}

/// Whether a values array of `values_len` values is valid for a fixed size list array of `length`
/// elements that are `width` values wide.
///
/// The values hold `width` slots per element rather than one, so this is [`is_valid_buffer_len`]
/// scaled by the width: `length * width` values when the array is flat, and the `width` of the one
/// element every element shares when it is scalar. An empty array has no element for a scalar
/// values array to stand for, so it admits only empty values. See the [module docs](self).
#[inline]
pub const fn is_valid_fixed_size_values_len(
    values_len: usize,
    width: usize,
    length: usize,
) -> bool {
    match length.checked_mul(width) {
        Some(flat_len) if values_len == flat_len => true,
        // A flat values array that overflows a `usize` is longer than any buffer can be, so the
        // scalar representation is all that is left.
        _ => length >= 1 && values_len == width,
    }
}
