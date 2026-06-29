use std::any::Any;
use std::borrow::Cow;
use std::hash::{BuildHasher, Hash, Hasher};

use polars_utils::aliases::PlFixedStateQuality;

use super::ExtensionTypeImpl;

/// A generic extension type used when the specific extension type is not registered.
pub struct GenericExtensionType {
    name: String,
    metadata: Option<String>,
}

impl GenericExtensionType {
    /// Create a new `GenericExtensionType` with the given name and optional metadata.
    pub fn new(name: String, metadata: Option<String>) -> Self {
        Self { name, metadata }
    }
}

impl ExtensionTypeImpl for GenericExtensionType {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.as_deref().map(Cow::Borrowed)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(Self {
            name: self.name.clone(),
            metadata: self.metadata.clone(),
        })
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        let Some(other) = (other as &dyn Any).downcast_ref::<GenericExtensionType>() else {
            return false;
        };

        self.name == other.name && self.metadata == other.metadata
    }

    fn dyn_hash(&self) -> u64 {
        let mut hasher = PlFixedStateQuality::default().build_hasher();
        self.name.hash(&mut hasher);
        self.metadata.hash(&mut hasher);
        hasher.finish()
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        if let Some(md) = &self.metadata {
            Cow::Owned(format!(
                "ExtensionType(name='{}', metadata='{}')",
                self.name, md
            ))
        } else {
            Cow::Owned(format!("ExtensionType(name='{}')", self.name))
        }
    }
}
