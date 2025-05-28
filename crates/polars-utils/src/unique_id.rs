use std::sync::Arc;

/// Intentionally a custom inner type to prevent deriving eq / hash, as that would not be correct.
struct Inner;

/// Unique ref-counted identifier.
#[derive(Clone)]
pub struct UniqueId(
    // We use an Arc to reserve a memory address rather than a static atomic counter, as the latter
    // may cause duplicate IDs on wrap-around.
    //
    // Note, this inner repr is a private implementation detail.
    Arc<Inner>,
);

impl std::fmt::Debug for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UniqueId({})", self.to_usize())
    }
}

impl UniqueId {
    #[inline]
    pub fn to_usize(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

impl Default for UniqueId {
    fn default() -> Self {
        Self(Arc::new(Inner {}))
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
