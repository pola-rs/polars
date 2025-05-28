use std::any::Any;
use std::sync::Arc;

/// Unique ref-counted identifier.
#[derive(Clone)]
pub enum MemoryId {
    /// We use an Arc to reserve a memory address rather than a static atomic counter, as the latter
    /// may cause duplicate IDs on wrap-around.
    ///
    /// Notes:
    /// * By having a dyn Any, we can be constructed from any Arc'ed type while still avoiding extra
    ///   allocations for ZST's.
    Arc(Arc<dyn Any + Send + Sync>),

    /// Stores a plain `usize`. This repr is used as the result of a serialization round-trip.
    ///
    /// Due to not being bound to an `Arc`, it may collide with `Arc`-backed `MemoryId`s.
    Unbinded(usize),
}

impl std::fmt::Debug for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryId({})", self.to_usize())
    }
}

impl MemoryId {
    #[inline]
    pub fn to_usize(&self) -> usize {
        match self {
            Self::Arc(v) => Arc::as_ptr(v) as *const () as usize,
            Self::Unbinded(v) => *v,
        }
    }

    pub fn from_arc<T: Any + Send + Sync>(arc: Arc<T>) -> Self {
        Self::Arc(arc.clone())
    }

    /// Downcasts to a concrete Arc type. Returns None `Self` is `Unbinded`.
    ///
    /// # Panics
    /// On a debug build, panics if `Self` is an `Arc` but does not contain `T`.
    pub fn downcast_arc<T: Any>(self) -> Option<Arc<T>> {
        match self {
            Self::Arc(inner) => {
                // Note, ref type here must match exactly with T.
                let v: &dyn Any = inner.as_ref();

                if v.type_id() != std::any::TypeId::of::<T>() {
                    if cfg!(debug_assertions) {
                        panic!("invalid downcast of MemoryId")
                    } else {
                        // Just return None on release.
                        return None;
                    }
                }

                // Safety: Type IDs checked above.
                let ptr: *const dyn Any = Arc::into_raw(inner);
                let ptr: *const T = ptr as _;
                Some(unsafe { Arc::from_raw(ptr) })
            },

            Self::Unbinded(_) => None,
        }
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::Arc(Arc::new(()))
    }
}

impl PartialEq for MemoryId {
    fn eq(&self, other: &Self) -> bool {
        self.to_usize() == other.to_usize()
    }
}

impl Eq for MemoryId {}

impl PartialOrd for MemoryId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl Ord for MemoryId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&self.to_usize(), &other.to_usize())
    }
}

impl std::hash::Hash for MemoryId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_usize().hash(state)
    }
}

#[cfg(feature = "serde")]
mod _serde_impl {
    use super::MemoryId;

    impl serde::ser::Serialize for MemoryId {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            self.to_usize().serialize(serializer)
        }
    }

    impl<'de> serde::de::Deserialize<'de> for MemoryId {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            usize::deserialize(deserializer).map(Self::Unbinded)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use super::MemoryId;

    #[test]
    fn test_unique_id() {
        let id = MemoryId::default();

        assert_eq!(id, id);
        assert_ne!(id, MemoryId::default());

        // Following code explains the memory layout
        let MemoryId::Arc(arc_ref) = &id else {
            unreachable!()
        };
        let inner_ref: &dyn Any = arc_ref.as_ref();

        assert_eq!(std::mem::size_of_val(inner_ref), 0);
        assert_eq!(std::mem::size_of::<Arc<dyn Any>>(), 16);

        assert_eq!(
            Arc::as_ptr(arc_ref) as *const () as usize,
            inner_ref as *const _ as *const () as usize,
        );
    }

    #[test]
    fn test_unique_id_downcast() {
        let id = MemoryId::default();
        let _: Arc<()> = id.downcast_arc().unwrap();

        let inner: Arc<usize> = Arc::new(37);
        let id = MemoryId::from_arc(inner);

        let out: Arc<usize> = id.downcast_arc().unwrap();
        assert_eq!(*out.as_ref(), 37);
    }
}
