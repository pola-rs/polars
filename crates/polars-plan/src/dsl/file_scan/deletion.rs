use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::prelude::PlIndexMap;
use polars_utils::pl_path::PlRefPath;

#[cfg(feature = "python")]
pub use super::python_delta_dv_provider::{
    DELTA_DV_PROVIDER_VTABLE, DeltaDeletionVectorProvider, DeltaDeletionVectorProviderVTable,
};

// Note, there are a lot of single variant enums here, but the intention is that we'll support
// Delta deletion vectors as well at some point in the future.

#[derive(Debug, Clone, Eq, PartialEq, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DeletionFilesList {
    // Chose to use a hashmap keyed by the scan source index.
    // * There may be data files without deletion files.
    // * A single data file may have multiple associated deletion files.
    //
    // Note that this uses `PlIndexMap` instead of `PlIndexMap` for schemars compatibility.
    //
    // Other possible options:
    // * ListArray(inner: Utf8Array)
    //
    /// Iceberg deletes
    Iceberg(Arc<PlIndexMap<usize, IcebergDeletes>>),
    /// Delta deletion vector
    #[cfg(feature = "python")]
    Delta(DeltaDeletionVectorProvider),
}

#[derive(Debug, Clone, Eq, PartialEq, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergDeletes {
    PositionDeletes(Buffer<PlRefPath>),
    DeletionVector(PlRefPath),
}

impl IcebergDeletes {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match self {
            IcebergDeletes::PositionDeletes(x) => x.len(),
            IcebergDeletes::DeletionVector(_) => 1,
        }
    }
}

impl DeletionFilesList {
    /// Converts `Some(v)` to `None` if `v` is empty.
    pub fn filter_empty(this: Option<Self>) -> Option<Self> {
        use DeletionFilesList::*;

        match this {
            Some(Iceberg(paths)) => (!paths.is_empty()).then_some(Iceberg(paths)),
            #[cfg(feature = "python")]
            Some(Delta(provider)) => Some(Delta(provider)),
            None => None,
        }
    }

    /// Returns the number of files with deletions, but only if known at plan time.
    pub fn num_files_with_deletions(&self) -> Option<usize> {
        use DeletionFilesList::*;

        match self {
            Iceberg(paths) => Some(paths.len()),
            #[cfg(feature = "python")]
            Delta(_) => None,
        }
    }
}

impl std::hash::Hash for DeletionFilesList {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use DeletionFilesList::*;

        std::mem::discriminant(self).hash(state);

        match self {
            Iceberg(paths) => {
                for i in 0..8 {
                    usize::hash(&paths.get_index(i).map_or(0, |x| *x.0), state)
                }
            },
            #[cfg(feature = "python")]
            Delta(provider) => provider.hash(state),
        }
    }
}

impl std::fmt::Display for DeletionFilesList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DeletionFilesList::*;

        match self {
            Iceberg(paths) => {
                let s = if paths.len() == 1 { "" } else { "s" };
                write!(f, "iceberg-position-delete: {} source{s}", paths.len())?;
            },
            #[cfg(feature = "python")]
            Delta(_) => {
                write!(f, "delta-deletion-vector-python-callback")?;
            },
        }

        Ok(())
    }
}
