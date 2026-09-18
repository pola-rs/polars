//! The data buffers a [`PlBinaryViewArray`](super::PlBinaryViewArray) copies its bytes into.

use polars_arrow::array::{BINVIEW_ARROW_BUFFER_LEN_LIMIT, BINVIEW_MAX_ROW_BYTE_LEN, View};

/// The capacity the first data buffer is allocated with, and the smallest any of them gets.
const DEFAULT_BLOCK_SIZE: usize = 8 * 1024;
/// The largest capacity the doubling of the data buffers reaches, which bounds over-allocation.
const MAX_EXP_BLOCK_SIZE: usize = 16 * 1024 * 1024;

const _: () = assert!(MAX_EXP_BLOCK_SIZE < BINVIEW_ARROW_BUFFER_LEN_LIMIT);

/// Copies `bytes` into `buffers`, and returns the [`View`] holding them.
#[inline]
pub(super) fn copy_value(buffers: &mut Vec<Vec<u8>>, buffer_idx_offset: u32, bytes: &[u8]) -> View {
    copy_value_limited::<BINVIEW_ARROW_BUFFER_LEN_LIMIT, BINVIEW_MAX_ROW_BYTE_LEN>(
        buffers,
        buffer_idx_offset,
        bytes,
    )
}

/// Copies `bytes` into a data buffer of its own, and returns the [`View`] holding them.
pub(super) fn copy_only_value(bytes: &[u8]) -> (View, Vec<Vec<u8>>) {
    copy_only_value_limited::<BINVIEW_MAX_ROW_BYTE_LEN>(bytes)
}

/// Takes `bytes` over as the only data buffer, and returns the [`View`] holding them.
pub(super) fn own_only_value(bytes: Vec<u8>) -> (View, Vec<Vec<u8>>) {
    own_only_value_limited::<BINVIEW_MAX_ROW_BYTE_LEN>(bytes)
}

/// [`own_only_value`], against a limit the tests lower.
fn own_only_value_limited<const MAX_ROW_BYTE_LEN: usize>(bytes: Vec<u8>) -> (View, Vec<Vec<u8>>) {
    if bytes.len() <= View::MAX_INLINE_SIZE as usize {
        return (View::new_inline(&bytes), Vec::new());
    }

    assert_row_fits::<MAX_ROW_BYTE_LEN>(bytes.len());

    // SAFETY: the bytes are longer than `View::MAX_INLINE_SIZE`, and they are the whole of the
    // buffer the view is over.
    let view = unsafe { View::new_noninline_unchecked(&bytes, 0, 0) };
    (view, vec![bytes])
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

/// [`copy_value`], against limits the tests lower to what they can reach cheaply.
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
    let offset = buffer.len() as u32;
    buffer.extend_from_slice(bytes);

    // SAFETY: the bytes are longer than `View::MAX_INLINE_SIZE`, and they were just written to
    // `offset` of the buffer this index reaches.
    unsafe { View::new_noninline_unchecked(bytes, buffer_idx, offset) }
}

/// Makes room for `additional` bytes in the last of `buffers`, pushing a new one where they fit.
#[inline]
fn reserve<const BUFFER_LEN_LIMIT: usize, const MAX_ROW_BYTE_LEN: usize>(
    buffers: &mut Vec<Vec<u8>>,
    additional: usize,
) {
    let (len, capacity) = buffers
        .last()
        .map_or((0, 0), |buffer| (buffer.len(), buffer.capacity()));

    if len.saturating_add(additional) > usize::min(BUFFER_LEN_LIMIT, capacity) {
        push_buffer::<MAX_ROW_BYTE_LEN>(buffers, additional);
    }
}

/// Pushes a buffer with room for `additional` bytes onto `buffers`.
#[cold]
fn push_buffer<const MAX_ROW_BYTE_LEN: usize>(buffers: &mut Vec<Vec<u8>>, additional: usize) {
    assert_row_fits::<MAX_ROW_BYTE_LEN>(additional);

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
