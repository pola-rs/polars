#![allow(unsafe_op_in_unsafe_fn)]
//! Multi-column keys stored as rows of `u64` words, with a stride fixed by the key
//! schema. Fixed-width values are stored by value and strings as their views; only
//! strings too long to inline compare their bytes.
mod hot;
mod keys;
mod layout;
mod map;
#[cfg(test)]
mod tests;

pub(crate) use hot::{HotKeyRows, KeyRowCollector};
pub use keys::KeyRowKeys;
pub(crate) use layout::KeyRowLayout;
pub(crate) use map::KeyRowIndexMap;

const BASE_KEY_BUFFER_CAPACITY: usize = 1024;
const MAX_KEY_BUFFER_CAPACITY: usize = 1 << 30;
pub(crate) const VERIFY_BATCH_SIZE: usize = 256;

/// Stores `bytes` in the last of `buffers`, starting a new one when it is full, and
/// returns where they were stored.
fn push_long_bytes(buffers: &mut Vec<Vec<u8>>, bytes: &[u8]) -> (u32, u32) {
    if buffers
        .last()
        .is_none_or(|buf| buf.len() + bytes.len() > buf.capacity())
    {
        let next_cap = buffers.last().map_or(BASE_KEY_BUFFER_CAPACITY, |buf| {
            (2 * buf.capacity()).min(MAX_KEY_BUFFER_CAPACITY)
        });
        buffers.push(Vec::with_capacity(next_cap.max(bytes.len())));
    }
    let buffer_idx = (buffers.len() - 1) as u32;
    let buffer = buffers.last_mut().unwrap();
    let offset = buffer.len() as u32;
    buffer.extend_from_slice(bytes);
    (buffer_idx, offset)
}
