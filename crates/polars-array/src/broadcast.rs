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
//! interpretations agree, so this is not ambiguous. A `length` of zero admits an empty buffer
//! (flat) as well as a one-element buffer (scalar), and neither is ever read.
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
