use std::sync::Arc;

/// Unique ref-counted identifier.
#[derive(Debug, Clone)]
pub struct UniqueId(
    // We use an Arc to reserve a memory address rather than a static atomic counter, as the latter
    // may cause duplicate IDs on wrap-around.
    //
    // Note, this inner repr is a private implementation detail.
    Arc<()>,
);

impl UniqueId {
    #[inline]
    pub fn to_usize(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

impl Default for UniqueId {
    fn default() -> Self {
        Self(Arc::new(()))
    }
}

impl PartialEq for UniqueId {
    fn eq(&self, other: &Self) -> bool {
        self.to_usize() == other.to_usize()
    }
}

impl std::hash::Hash for UniqueId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_usize().hash(state)
    }
}

#[cfg(test)]
mod tests {
    use super::UniqueId;

    #[test]
    fn test_unique_id() {
        let a = UniqueId::default();
        let b = UniqueId::default();

        assert_ne!(a, b);
    }
}
