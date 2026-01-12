use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use hashbrown::hash_map::Entry;
use polars_error::{PolarsResult, polars_bail, polars_err, polars_warn};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::pl_str::PlSmallStr;

use super::{ExtensionTypeFactory, ExtensionTypeInstance};
use crate::prelude::{DataType, POLARS_OBJECT_EXTENSION_NAME};

#[repr(u8)]
pub enum UnknownExtensionTypeBehavior {
    LoadAsGeneric = 0,
    LoadAsStorage,
    WarnAndLoadAsStorage,
}

static UNKNOWN_EXTENSION_TYPE_BEHAVIOR: AtomicU8 =
    AtomicU8::new(UnknownExtensionTypeBehavior::LoadAsGeneric as u8);

pub fn set_unknown_extension_type_behavior(behavior: UnknownExtensionTypeBehavior) {
    UNKNOWN_EXTENSION_TYPE_BEHAVIOR.store(behavior as u8, Ordering::Relaxed);
}

pub fn get_unknown_extension_type_behavior() -> UnknownExtensionTypeBehavior {
    match UNKNOWN_EXTENSION_TYPE_BEHAVIOR.load(Ordering::Relaxed) {
        0 => UnknownExtensionTypeBehavior::LoadAsGeneric,
        1 => UnknownExtensionTypeBehavior::LoadAsStorage,
        2 => UnknownExtensionTypeBehavior::WarnAndLoadAsStorage,
        _ => unreachable!(),
    }
}

/// Returns the extension type or `None` if the extension type should be loaded as its storage type.
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
        None => match get_unknown_extension_type_behavior() {
            UnknownExtensionTypeBehavior::LoadAsStorage => None,
            UnknownExtensionTypeBehavior::LoadAsGeneric => {
                let typ = super::GenericExtensionType::new(
                    name.to_string(),
                    metadata.map(|s| s.to_string()),
                );
                Some(ExtensionTypeInstance(Box::new(typ)))
            },
            UnknownExtensionTypeBehavior::WarnAndLoadAsStorage => {
                if UNKNOWN_EXTENSION_TYPE_BEHAVIOR.swap(
                    UnknownExtensionTypeBehavior::LoadAsStorage as u8,
                    Ordering::Relaxed,
                ) == UnknownExtensionTypeBehavior::WarnAndLoadAsStorage as u8
                {
                    polars_warn!("Extension type '{name}' is not registered; loading as its storage type.

To avoid this warning, register the extension type or set environment variable 'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR' to 'load_as_storage' or 'load_as_extension'.

In Polars 2.0, the default behavior will change to 'load_as_extension'.");
                }
                None
            },
        },
    }
}

/// Returns the extension type; if unknown, returns a generic extension type.
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

#[allow(clippy::type_complexity)]
static REGISTRY: LazyLock<RwLock<PlHashMap<PlSmallStr, Option<Arc<dyn ExtensionTypeFactory>>>>> =
    LazyLock::new(|| {
        let mut m = PlHashMap::new();
        m.insert(PlSmallStr::from_static(POLARS_OBJECT_EXTENSION_NAME), None);
        RwLock::new(m)
    });

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
