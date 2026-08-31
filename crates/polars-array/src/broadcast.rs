//! The rules governing the scalar (broadcast) representation.
//!
//! Every array in this crate stores its logical length in a dedicated `length` field, decoupled
//! from the lengths of its backing buffers. A backing buffer is read through
//! [`broadcast_index`]: element `i` of the array reads slot `i` of the buffer if the buffer is long
//! enough, and slot `0` otherwise.
//!
//! That leaves exactly two admissible states per buffer, which every constructor validates:
//!
//! * *flat*: `buffer.len() == length`, the usual one-slot-per-element layout.
//! * *broadcast*: `buffer.len() == 1`, a single value shared by all `length` elements.
//!
//! A buffer of length one is simultaneously flat and broadcast when `length == 1`; the two
//! interpretations agree, so this is not ambiguous. A `length` of zero admits an empty buffer
//! (flat) as well as a one-element buffer (broadcast), and neither is ever read.
//!
//! Intermediate buffer lengths (`1 < buffer.len() < length`) are *not* valid, even though
//! [`broadcast_index`] would happily map them: an array is either flat or broadcast.

/// Maps a logical element index onto a slot in a backing buffer of length `buffer_len`.
///
/// Returns `i` when the buffer holds a slot for every element, and `0` when the buffer is a
/// broadcast buffer holding one shared value.
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
