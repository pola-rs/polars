use std::sync::Arc;

/// Unique ref-counted identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct UniqueId(
    // We use an Arc to reserve a memory address rather than a static atomic counter, as the latter
    // may cause duplicate IDs on wrap-around.
    //
    // Note, this inner repr is a private implementation detail.
    Arc<()>,
);

impl UniqueId {
    pub fn to_usize(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

impl Default for UniqueId {
    fn default() -> Self {
        Self(Arc::new(()))
    }
}
