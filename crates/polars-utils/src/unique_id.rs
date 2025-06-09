use std::any::Any;
use std::fmt::LowerHex;
use std::sync::Arc;

/// Unique identifier potentially backed by an `Arc` address to protect against collisions.
///
/// Note that a serialization roundtrip will force this to become a [`UniqueId::Plain`] variant.
#[derive(Clone)]
pub enum UniqueId {
    /// ID that derives from the memory address of an `Arc` to protect against collisions.
    /// Note that it internally stores as a `dyn Any` to allow it to re-use an existing `Arc` holding
    /// any type.
    MemoryRef(Arc<dyn Any + Send + Sync>),

    /// Stores a plain `usize`. Unlike the `MemoryRef` variant, there is no internal protection against
    /// collisions - this must handled separately.
    ///
    /// Note: This repr may also be constructed as the result of a serialization round-trip.
    Plain(usize),
}

impl UniqueId {
    #[inline]
    pub fn to_usize(&self) -> usize {
        match self {
            Self::MemoryRef(v) => Arc::as_ptr(v) as *const () as usize,
            Self::Plain(v) => *v,
        }
    }

    /// Use an existing `Arc<T>` as backing for an ID.
    pub fn from_arc<T: Any + Send + Sync>(arc: Arc<T>) -> Self {
        Self::MemoryRef(arc.clone())
    }

    /// Downcasts to a concrete Arc type. Returns None `Self` is `Plain`.
    ///
    /// # Panics
    /// On a debug build, panics if `Self` is an `Arc` but does not contain `T`. On a release build
    /// this will instead `None`.
    pub fn downcast_arc<T: Any>(self) -> Option<Arc<T>> {
        match self {
            Self::MemoryRef(inner) => {
                // Note, ref type here must match exactly with T.
                let v: &dyn Any = inner.as_ref();

                if v.type_id() != std::any::TypeId::of::<T>() {
                    if cfg!(debug_assertions) {
                        panic!("invalid downcast of UniqueId")
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

            Self::Plain(_) => None,
        }
    }
}

impl Default for UniqueId {
    fn default() -> Self {
        Self::MemoryRef(Arc::new(()))
    }
}

impl PartialEq for UniqueId {
    fn eq(&self, other: &Self) -> bool {
        self.to_usize() == other.to_usize()
    }
}

impl Eq for UniqueId {}

impl PartialOrd for UniqueId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl Ord for UniqueId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&self.to_usize(), &other.to_usize())
    }
}

impl std::hash::Hash for UniqueId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_usize().hash(state)
    }
}

impl std::fmt::Display for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.to_usize(), f)
    }
}

impl std::fmt::Debug for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use UniqueId::*;

        write!(
            f,
            "UniqueId::{}({})",
            match self {
                MemoryRef(_) => "MemoryRef",
                Plain(_) => "Plain",
            },
            self.to_usize()
        )
    }
}

impl LowerHex for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LowerHex::fmt(&self.to_usize(), f)
    }
}

#[cfg(feature = "serde")]
mod _serde_impl {
    use super::UniqueId;

    impl serde::ser::Serialize for UniqueId {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            usize::serialize(&self.to_usize(), serializer)
        }
    }

    impl<'de> serde::de::Deserialize<'de> for UniqueId {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            usize::deserialize(deserializer).map(Self::Plain)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use super::UniqueId;

    #[test]
    fn test_unique_id() {
        let id = UniqueId::default();

        assert!(matches!(id, UniqueId::MemoryRef(_)));

        assert_eq!(id, id);
        assert_ne!(id, UniqueId::default());

        assert_eq!(id, UniqueId::Plain(id.to_usize()));

        // Following code shows the memory layout
        let UniqueId::MemoryRef(arc_ref) = &id else {
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
        let id = UniqueId::default();
        let _: Arc<()> = id.downcast_arc().unwrap();

        let inner: Arc<usize> = Arc::new(37);
        let id = UniqueId::from_arc(inner);

        let out: Arc<usize> = id.downcast_arc().unwrap();
        assert_eq!(*out.as_ref(), 37);
    }
}
