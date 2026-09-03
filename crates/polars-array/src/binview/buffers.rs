//! The data buffers a [`PlBinaryViewArray`](super::PlBinaryViewArray) copies its bytes into.

use arrow::array::{BINVIEW_ARROW_BUFFER_LEN_LIMIT, BINVIEW_MAX_ROW_BYTE_LEN, View};

/// The capacity the first data buffer is allocated with, and the smallest any of them gets.
const DEFAULT_BLOCK_SIZE: usize = 8 * 1024;
/// The largest capacity the doubling of the data buffers reaches, which bounds what a buffer
/// over-allocates for the values still to come.
const MAX_EXP_BLOCK_SIZE: usize = 16 * 1024 * 1024;

// Growing a buffer never carries it past the limit by itself: only a single value longer than the
// limit does, and that value is all the buffer holding it ever holds.
const _: () = assert!(MAX_EXP_BLOCK_SIZE < BINVIEW_ARROW_BUFFER_LEN_LIMIT);

/// Copies `bytes` into `buffers`, and returns the [`View`] holding them.
///
/// # Panics
/// Panics if `bytes` is longer than [`BINVIEW_MAX_ROW_BYTE_LEN`], the longest a view can point at,
/// or if the buffers of the array being built outgrow the buffer index of a view.
#[inline]
pub(super) fn copy_value(buffers: &mut Vec<Vec<u8>>, buffer_idx_offset: u32, bytes: &[u8]) -> View {
    copy_value_limited::<BINVIEW_ARROW_BUFFER_LEN_LIMIT, BINVIEW_MAX_ROW_BYTE_LEN>(
        buffers,
        buffer_idx_offset,
        bytes,
    )
}

/// Copies `bytes` into a data buffer of its own, and returns the [`View`] holding them.
///
/// # Panics
/// Panics if `bytes` is longer than [`BINVIEW_MAX_ROW_BYTE_LEN`], the longest a view can point at.
pub(super) fn copy_only_value(bytes: &[u8]) -> (View, Vec<Vec<u8>>) {
    copy_only_value_limited::<BINVIEW_MAX_ROW_BYTE_LEN>(bytes)
}

/// [`copy_only_value`], against a limit the tests lower.
fn copy_only_value_limited<const MAX_ROW_BYTE_LEN: usize>(bytes: &[u8]) -> (View, Vec<Vec<u8>>) {
    if bytes.len() <= View::MAX_INLINE_SIZE as usize {
        return (View::new_inline(bytes), Vec::new());
    }

    assert_row_fits::<MAX_ROW_BYTE_LEN>(bytes.len());

    // SAFETY: the bytes are longer than `View::MAX_INLINE_SIZE`, and they are the whole of the
    // buffer the view is over.
    let view = unsafe { View::new_noninline_unchecked(bytes, 0, 0) };
    (view, vec![bytes.to_vec()])
}

/// [`copy_value`], against limits the tests lower to what they can reach without allocating
/// gigabytes.
fn copy_value_limited<const BUFFER_LEN_LIMIT: usize, const MAX_ROW_BYTE_LEN: usize>(
    buffers: &mut Vec<Vec<u8>>,
    buffer_idx_offset: u32,
    bytes: &[u8],
) -> View {
    if bytes.len() <= View::MAX_INLINE_SIZE as usize {
        return View::new_inline(bytes);
    }

    reserve::<BUFFER_LEN_LIMIT, MAX_ROW_BYTE_LEN>(buffers, bytes.len());

    let buffer_idx = u32::try_from(buffer_idx_offset as usize + buffers.len() - 1)
        .expect("the built array holds more data buffers than a view can index");

    let buffer = buffers.last_mut().unwrap();
    // The reservation left the buffer short enough for the bytes to be reached by a view.
    let offset = buffer.len() as u32;
    buffer.extend_from_slice(bytes);

    // SAFETY: the bytes are longer than `View::MAX_INLINE_SIZE`, and they were just written to
    // `offset` of the buffer this index reaches.
    unsafe { View::new_noninline_unchecked(bytes, buffer_idx, offset) }
}

/// Makes room for `additional` bytes at the end of the last of `buffers`, pushing a new buffer
/// where they do not fit.
#[inline]
fn reserve<const BUFFER_LEN_LIMIT: usize, const MAX_ROW_BYTE_LEN: usize>(
    buffers: &mut Vec<Vec<u8>>,
    additional: usize,
) {
    let (len, capacity) = buffers
        .last()
        .map_or((0, 0), |buffer| (buffer.len(), buffer.capacity()));

    // A buffer is never grown once it is written into, so that the bytes already in it are never
    // copied again: what does not fit in the capacity it was allocated with starts a new buffer,
    // as does what would carry it past the limit.
    if len.saturating_add(additional) > usize::min(BUFFER_LEN_LIMIT, capacity) {
        push_buffer::<MAX_ROW_BYTE_LEN>(buffers, additional);
    }
}

/// Pushes a buffer with room for `additional` bytes onto `buffers`.
#[cold]
fn push_buffer<const MAX_ROW_BYTE_LEN: usize>(buffers: &mut Vec<Vec<u8>>, additional: usize) {
    assert_row_fits::<MAX_ROW_BYTE_LEN>(additional);

    // The buffers double in size to amortize the cost of filling them, up to a block size that
    // bounds what the last of them over-allocates. A value longer than a whole block is the one
    // thing a buffer is allowed to be larger than that for: the single view over it reads the
    // value out of one buffer, so the buffer has to hold all of it.
    let previous_capacity = buffers.last().map_or(0, Vec::capacity);
    let capacity = usize::max(
        additional,
        (previous_capacity * 2).clamp(DEFAULT_BLOCK_SIZE, MAX_EXP_BLOCK_SIZE),
    );
    buffers.push(Vec::with_capacity(capacity));
}

/// Asserts that a value of `len` bytes is one a view can point at.
fn assert_row_fits<const MAX_ROW_BYTE_LEN: usize>(len: usize) {
    assert!(
        len <= MAX_ROW_BYTE_LEN,
        "value of {len} bytes is longer than the {MAX_ROW_BYTE_LEN} bytes a view can hold",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Limits in the same order as the real ones — a value may be longer than a buffer normally
    /// grows — small enough to reach without allocating gigabytes.
    const BUFFER_LEN_LIMIT: usize = 32;
    const MAX_ROW_BYTE_LEN: usize = 64;

    /// A value of `len` bytes, told apart from the other values by `tag`.
    fn value(len: usize, tag: u8) -> Vec<u8> {
        std::iter::repeat_n(tag, len).collect()
    }

    fn copy(buffers: &mut Vec<Vec<u8>>, bytes: &[u8]) -> View {
        copy_value_limited::<BUFFER_LEN_LIMIT, MAX_ROW_BYTE_LEN>(buffers, 0, bytes)
    }

    /// The bytes the view stands for, read back out of the buffers they were copied into.
    fn read<'a>(view: &'a View, buffers: &'a [Vec<u8>]) -> &'a [u8] {
        // SAFETY: the view came out of a copy into these buffers.
        unsafe { view.get_slice_unchecked(buffers) }
    }

    #[test]
    fn an_inlined_value_reaches_no_buffer() {
        let mut buffers = Vec::new();
        let view = copy(&mut buffers, b"foo");

        assert!(view.is_inline());
        assert!(
            buffers.is_empty(),
            "the bytes a view carries are copied nowhere",
        );
        assert_eq!(read(&view, &buffers), b"foo");
    }

    #[test]
    fn values_share_the_buffer_they_fit_in() {
        let mut buffers = Vec::new();
        let first = copy(&mut buffers, &value(13, b'a'));
        let second = copy(&mut buffers, &value(13, b'b'));

        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].len(), 26);
        assert_eq!(first.buffer_idx, 0);
        assert_eq!(second.buffer_idx, 0);
        assert_eq!(second.offset, 13);
        assert_eq!(read(&first, &buffers), value(13, b'a'));
        assert_eq!(read(&second, &buffers), value(13, b'b'));
    }

    #[test]
    fn a_value_that_would_carry_a_buffer_past_the_limit_starts_a_new_one() {
        let mut buffers = Vec::new();
        let first = copy(&mut buffers, &value(20, b'a'));
        let second = copy(&mut buffers, &value(20, b'b'));

        assert_eq!(
            buffers.len(),
            2,
            "40 bytes do not fit in a buffer limited to {BUFFER_LEN_LIMIT}",
        );
        assert_eq!(buffers[0].len(), 20);
        assert_eq!(buffers[1].len(), 20);
        assert_eq!(first.buffer_idx, 0);
        assert_eq!(second.buffer_idx, 1);
        assert_eq!(second.offset, 0);
        assert_eq!(read(&first, &buffers), value(20, b'a'));
        assert_eq!(read(&second, &buffers), value(20, b'b'));
    }
}
