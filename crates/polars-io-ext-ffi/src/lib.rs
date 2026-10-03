//! FFI contracts between Polars and its separately downloaded I/O plugins (e.g. `polars_iceberg`).
//!
//! Nothing in this crate depends on Polars. There is no generic plugin interface: each supported
//! plugin type has its own minimal contract module, identified by a **plugin ID** such as
//! [`iceberg_v1::ID`]. A plugin ID fixes everything that crosses the boundary: struct layouts,
//! function signatures, the request and output documents and the meaning of their fields,
//! lifetime and threading rules, and the types of [`common`].
//!
//! Compatibility rules:
//! * A released contract module is frozen. Any change, however small (including a change of
//!   meaning), is a new module with a new ID, e.g. `iceberg_v2`.
//! * A plugin exposes each contract as a PyCapsule whose name is the plugin ID and whose pointer
//!   is the contract's `Plugin` struct. The Python module of the plugin lists its IDs in
//!   `_polars_io_plugin_ids: tuple[str, ...]` and returns the capsule for an ID from
//!   `_capsule(id: str)`. This handshake never changes.
//! * Polars uses the newest ID that both sides support, dispatches by exact capsule name, and
//!   refuses unknown IDs. Polars drops an ID by removing it from its supported list (calls then
//!   fail with a compatibility error), a plugin by no longer exporting it.
//! * IDs end in `.v<N>`; `N` is used to order IDs in error messages only.
//! * An ID is dropped on one side only after the other side has released its successor, so that
//!   a compatible pair of released versions always exists.
//!
//! Each plugin type's contract is behind a Cargo feature of the same name (e.g. `iceberg`).
//! Features only control what is compiled; they never change an ID or a layout.
#![allow(clippy::missing_safety_doc)]

pub mod common;
#[cfg(feature = "iceberg")]
pub mod iceberg_v1;

use std::ffi::CStr;

/// Prefix of all plugin IDs (capsule names).
pub const ID_PREFIX: &str = "polars.io_plugin.";

/// All plugin IDs defined by this crate, for exhaustive matching.
///
/// Only the ID strings ([`PluginId::name`]) cross the FFI boundary; the enum and its
/// discriminants are not part of any contract and may be reordered freely. Adding a variant
/// forces every `match` on it to handle the new contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PluginId {
    #[cfg(feature = "iceberg")]
    IcebergV1,
}

impl PluginId {
    /// Every ID, oldest first within each plugin type.
    pub const ALL: &[Self] = &[
        #[cfg(feature = "iceberg")]
        Self::IcebergV1,
    ];

    /// The ID string, used as capsule name.
    pub const fn name(self) -> &'static CStr {
        match self {
            #[cfg(feature = "iceberg")]
            Self::IcebergV1 => iceberg_v1::ID,
        }
    }

    pub fn as_str(self) -> &'static str {
        self.name().to_str().unwrap()
    }

    /// The ID with this name; `None` for names unknown to this build.
    pub fn from_name(name: &CStr) -> Option<Self> {
        Self::ALL.iter().copied().find(|id| id.name() == name)
    }

    pub fn from_id_str(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|id| id.as_str() == name)
    }
}

impl std::fmt::Display for PluginId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_roundtrip() {
        for &id in PluginId::ALL {
            assert!(id.as_str().starts_with(ID_PREFIX));
            assert_eq!(PluginId::from_name(id.name()), Some(id));
            assert_eq!(PluginId::from_id_str(id.as_str()), Some(id));
        }
        assert_eq!(PluginId::from_id_str("polars.io_plugin.iceberg.v0"), None);
    }
}
