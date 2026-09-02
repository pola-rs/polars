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
//! # Constructing one representation or the other
//!
//! Which of the two an array is in is never inferred from the buffers it is handed: every array
//! has two families of constructors, and the caller picks the representation by name.
//!
//! * `try_new`, `new` and `new_unchecked` build a *flat* array. Every backing buffer has to hold
//!   one slot per element — [`is_flat_buffer_len`], or [`is_flat_offsets_len`] and
//!   [`is_flat_fixed_size_values_len`] for the buffers that hold more than one slot per element.
//! * `try_new_broadcast`, `new_broadcast` and `new_broadcast_unchecked` build a *scalar* array of
//!   any `length`. Every backing buffer has to hold the single value every element shares —
//!   [`is_scalar_buffer_len`], [`is_scalar_offsets_len`] and
//!   [`is_scalar_fixed_size_values_len`].
//!
//! Neither family accepts a buffer of a length that merely happens to be admissible: a values
//! buffer of one element passed to `new` is an error rather than a silently broadcast array, and
//! `new_broadcast` refuses a buffer holding one slot per element of an array longer than one. The
//! two therefore accept the same buffers exactly where the representations themselves coincide: an
//! array of one element, and the empty buffer of an array of none. An array of no elements reads
//! no slot at all, so the scalar family additionally lets it keep the one slot nothing reads.
//!
//! An array that mixes the two — a single shared value under a flat validity mask, say — is a
//! valid array that neither family builds in one step. Build it in the representation its values
//! are in and replace the mask with one of the validity setters below.
//!
//! # Replacing the validity mask
//!
//! The setters name the representation the same way, so that neither of them broadcasts a mask
//! the caller did not mean to share:
//!
//! * `set_validity` and `with_validity` install a *flat* mask, one bit per element —
//!   [`is_flat_buffer_len`].
//! * `set_validity_broadcast` and `with_validity_broadcast` install a mask that broadcasts over
//!   the array, so they additionally admit the single bit every element shares —
//!   [`is_valid_buffer_len`].
//!
//! The broadcasting ones are the wider of the two rather than the mirror image, unlike the
//! constructors: a caller that has a mask which is already flat *or* scalar for the array —
//! whichever the mask it was handed happened to be — installs it in one call, exactly as it
//! would iterate one with [`broadcast_iter`](crate::PlPrimitiveArray::broadcast_iter).
//!
//! # Broadcasting an array
//!
//! The same rule applies one level up, to the arrays themselves: an array of a single element
//! stands for that element repeated any number of times, exactly as a scalar buffer stands for
//! the value in its one slot. That is what
//! [`broadcast_iter`](crate::PlPrimitiveArray::broadcast_iter) exploits: an array of one element
//! iterates as `length` copies of that element, which is `O(1)` because the copies are never
//! materialized. Whether the array is flat or scalar does not come into it — an array of one
//! element holds a single slot in every backing buffer either way, so it is the *logical* lengths
//! that [`is_broadcastable`] relates.
//!
//! # The offsets of a list or binary array
//!
//! The offsets of a [`PlListArray`](crate::PlListArray) — and of a
//! [`PlBinaryArray`](crate::PlBinaryArray), which are governed by the same rule — are the one
//! backing buffer that does not hold one slot per element when it is flat: element `i` covers the
//! range `offsets[i]..offsets[i + 1]`, so the buffer holds the start of every element plus the end
//! of the last. It is the *starts* that are flat or scalar, and they are one slot shorter than the
//! buffer:
//!
//! * *flat*: `offsets.len() == length + 1`, one range per element, laid end to end.
//! * *scalar*: `offsets.len() == 2`, a single range shared by all `length` elements.
//!
//! Such a buffer is read through [`broadcast_index(i, offsets.len() - 1)`](broadcast_index), and
//! validated with [`is_valid_offsets_len`].
//!
//! # The values of a fixed size array
//!
//! The values of a [`PlFixedSizeListArray`](crate::PlFixedSizeListArray) — and the bytes of a
//! [`PlFixedSizeBinaryArray`](crate::PlFixedSizeBinaryArray), which are governed by the same rule —
//! are the other backing buffer that does not hold one slot per element: element `i` covers `width`
//! of them at a time. It is therefore the *elements* the buffer holds that are flat or scalar, and
//! each of them is `width` slots wide:
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
