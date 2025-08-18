use std::sync::Arc;

use polars_core::prelude::PlIndexMap;

// Note, there are a lot of single variant enums here, but the intention is that we'll support
// Delta deletion vectors as well at some point in the future.

#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DeletionFilesList {
    // Chose to use a hashmap keyed by the scan source index.
    // * There may be data files without deletion files.
    // * A single data file may have multiple associated deletion files.
    //
    // Note that this uses `PlIndexMap` instead of `PlHashMap` for schemars compatibility.
    //
    // Other possible options:
    // * ListArray(inner: Utf8Array)
    //
    /// Iceberg positional deletes
    IcebergPositionDelete(Arc<PlIndexMap<usize, Arc<[String]>>>),
}

impl DeletionFilesList {
    /// Converts `Some(v)` to `None` if `v` is empty.
    pub fn filter_empty(this: Option<Self>) -> Option<Self> {
        use DeletionFilesList::*;

        match this {
            Some(IcebergPositionDelete(paths)) => {
                (!paths.is_empty()).then_some(IcebergPositionDelete(paths))
            },
            None => None,
        }
    }

    pub fn num_files_with_deletions(&self) -> usize {
        use DeletionFilesList::*;

        match self {
            IcebergPositionDelete(paths) => paths.len(),
        }
    }
}

impl std::hash::Hash for DeletionFilesList {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use DeletionFilesList::*;

        std::mem::discriminant(self).hash(state);

        match self {
            IcebergPositionDelete(paths) => {
                let addr = paths
                    .first()
                    .map_or(0, |(_, paths)| Arc::as_ptr(paths) as *const () as usize);

                addr.hash(state)
            },
        }
    }
}

impl std::fmt::Display for DeletionFilesList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DeletionFilesList::*;

        match self {
            IcebergPositionDelete(paths) => {
                let s = if paths.len() == 1 { "" } else { "s" };
                write!(f, "iceberg-position-delete: {} source{s}", paths.len())?;
            },
        }

        Ok(())
    }
}
