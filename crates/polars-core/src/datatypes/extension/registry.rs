use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, LazyLock, RwLock};

use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_utils::pl_str::PlSmallStr;

use super::{ExtensionTypeFactory, ExtensionTypeInstance};
use crate::prelude::DataType;
use crate::prelude::extension::ExtensionTypeImpl;

pub enum UnknownExtensionTypeBehavior {
    LoadAsGeneric,
    LoadAsStorage,
    RaiseError,
}

pub fn get_extension_type_or_storage(
    name: &str,
    storage: &DataType,
    metadata: Option<&str>,
) -> Option<ExtensionTypeInstance> {
    match REGISTRY.read().unwrap().get(name) {
        Some(Some(factory)) => Some(ExtensionTypeInstance(
            factory.create_type_instance(name, storage, metadata),
        )),
        Some(None) => None,
        None => {
            // FIXME @ extension-type: give warning, fallback behavior.
            let typ = super::GenericExtensionType::new(name.to_string(), metadata.map(|s| s.to_string()));
            Some(ExtensionTypeInstance(Box::new(typ)))
        }
    }
}

pub fn get_extension_type_or_generic(
    name: &str,
    storage: &DataType,
    metadata: Option<&str>,
) -> ExtensionTypeInstance {
    if let Some(Some(factory)) = REGISTRY.read().unwrap().get(name) {
        return ExtensionTypeInstance(factory.create_type_instance(name, storage, metadata));
    }

    let typ = super::GenericExtensionType::new(name.to_string(), metadata.map(|s| s.to_string()));
    ExtensionTypeInstance(Box::new(typ))
}

pub fn load_extension_type_as_storage(name: &str) -> bool {
    matches!(REGISTRY.read().unwrap().get(name), Some(None))
}

static REGISTRY: LazyLock<RwLock<HashMap<PlSmallStr, Option<Arc<dyn ExtensionTypeFactory>>>>> =
    LazyLock::new(|| RwLock::default());

pub fn register_extension_type(
    name: &str,
    t: Option<Arc<dyn ExtensionTypeFactory>>,
) -> PolarsResult<()> {
    match REGISTRY.write().unwrap().entry(name.into()) {
        Entry::Occupied(_) => {
            polars_bail!(ComputeError: "attempted to register duplicate extension type with name '{name}'")
        },
        Entry::Vacant(v) => {
            v.insert(t);
            Ok(())
        },
    }
}

pub fn unregister_extension_type(
    name: &str,
) -> PolarsResult<Option<Arc<dyn ExtensionTypeFactory>>> {
    REGISTRY.write().unwrap().remove(name).ok_or_else(||
        polars_err!(ComputeError: "attempted to unregister unknown extension type with name '{name}'")
    )
}
