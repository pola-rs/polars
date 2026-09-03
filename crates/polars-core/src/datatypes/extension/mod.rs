use std::any::Any;
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

use crate::datatypes::DataType;

mod generic;
mod registry;

use generic::GenericExtensionType;
pub use registry::{
    UnknownExtensionTypeBehavior, get_extension_type_or_generic, get_extension_type_or_storage,
    register_extension_type, set_unknown_extension_type_behavior, unregister_extension_type,
};

pub trait ExtensionTypeFactory: 'static + Send + Sync {
    fn create_type_instance(
        &self,
        name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl>;
}

pub trait ExtensionTypeImpl: 'static + Send + Sync + Any {
    /// Name of the extension type.
    fn name(&self) -> Cow<'_, str>;

    /// Serialize the metadata of the extension type.
    fn serialize_metadata(&self) -> Option<Cow<'_, str>>;

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl>;
    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool;
    fn dyn_hash(&self) -> u64;

    /// Display representation of the extension type.
    ///
    /// Should be a short string representation, lowercase. For example: str, datetime[ms].
    fn dyn_display(&self) -> Cow<'_, str>;

    /// Debug representation of the extension type.
    ///
    /// Should be a more verbose string representation, useful for debugging, in TitleCase,
    /// for example: String, Decimal(10, 2).
    fn dyn_debug(&self) -> Cow<'_, str>;
}

#[repr(transparent)]
pub struct ExtensionTypeInstance(pub Box<dyn ExtensionTypeImpl>);

impl Clone for ExtensionTypeInstance {
    fn clone(&self) -> Self {
        Self(self.0.dyn_clone())
    }
}

impl PartialEq for ExtensionTypeInstance {
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(&*other.0)
    }
}

impl Eq for ExtensionTypeInstance {}

impl Hash for ExtensionTypeInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let h = self.0.dyn_hash();
        h.hash(state);
    }
}

impl Display for ExtensionTypeInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.dyn_display())
    }
}

impl Debug for ExtensionTypeInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.dyn_debug())
    }
}

impl ExtensionTypeInstance {
    pub fn name(&self) -> Cow<'_, str> {
        self.0.name()
    }

    pub fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.0.serialize_metadata()
    }
}
