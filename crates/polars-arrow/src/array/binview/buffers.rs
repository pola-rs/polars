use std::hash::{Hash, Hasher};
use std::sync::Arc;
use polars_utils::aliases::{InitHashMaps, PlHashSet, PlIndexSet};
use crate::buffer::Buffer;

pub struct BufferKey<'a> {
    pub inner: &'a Buffer<u8>,
}

impl Hash for BufferKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.inner.as_ptr() as u64)
    }
}

impl PartialEq for BufferKey<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.as_ptr() == other.inner.as_ptr()
    }
}

impl Eq for BufferKey<'_> {}

pub fn dedupe_view_buffers<'a, I: Iterator<Item=&'a Arc<[Buffer<u8>]>>>(iter: I) -> PlIndexSet<BufferKey<'a>> {
    // Deduplicate a whole Arc<[buffer]> group
    let mut processed_buffer_groups = PlHashSet::new();
    // Deduplicate the separate Buffers.
    let mut buffers = PlIndexSet::new();
    for data_buffers in iter {
        if processed_buffer_groups.insert(data_buffers.as_ptr() as usize) {
            buffers.extend(data_buffers.iter().map(|buf| BufferKey { inner: buf }))
        }
    }
    buffers
}
