use std::sync::{Arc, OnceLock};

use polars_core::frame::DataFrame;
use polars_core::prelude::{
    BooleanChunked, DataType, IntoColumn, PlHashMap, PlIndexMap, UInt64Chunked,
};
use polars_error::PolarsResult;
use polars_utils::python_function::PythonObject;

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
    // Note that this uses `PlIndexMap` instead of `PlHashMap` for schemars compatibility.
    //
    // Other possible options:
    // * ListArray(inner: Utf8Array)
    //
    /// Iceberg positional deletes
    IcebergPositionDelete(Arc<PlIndexMap<usize, Arc<[String]>>>),
}

#[derive(Clone)]
pub struct DeltaDeletionVectorCallback(
    pub Arc<dyn Fn() -> PolarsResult<Option<DataFrame>> + Send + Sync>,
);

impl PartialEq for DeltaDeletionVectorCallback {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for DeltaDeletionVectorCallback {}

impl std::hash::Hash for DeltaDeletionVectorCallback {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl std::fmt::Debug for DeltaDeletionVectorCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeltaDeletionVectorCallback")
    }
}

/// This is for `polars-python` to inject so that the implementation can be done there:
/// * The impls for converting from Python objects are there.
pub static DELTA_DV_PROVIDER_VTABLE: OnceLock<DeltaDeletionVectorProviderVTable> = OnceLock::new();

pub struct DeltaDeletionVectorProviderVTable {
    pub call: fn(callback: &PythonObject) -> PolarsResult<Option<DataFrame>>,
}

pub fn delta_dv_provider_vtable() -> Result<&'static DeltaDeletionVectorProviderVTable, &'static str>
{
    DELTA_DV_PROVIDER_VTABLE
        .get()
        .ok_or("DELTA_DV_PROVIDER_VTABLE not initialized")
}

/// For Delta Deletion Vector provider
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct DeltaDeletionVectorProvider {
    callback: PythonObject,
    selected_indices: Option<Arc<[usize]>>,
}

impl DeltaDeletionVectorProvider {
    pub fn new(callback: PythonObject) -> Self {
        Self {
            callback,
            selected_indices: None,
        }
    }

    pub fn with_selected_indices(mut self, indices: impl Iterator<Item = usize> + Clone) -> Self {
        self.selected_indices = Some(indices.collect::<Vec<_>>().into());
        self
    }

    pub fn call(&self) -> PolarsResult<Option<DataFrame>> {
        let dv = (delta_dv_provider_vtable().unwrap().call)(&self.callback)?;

        let Some(mut dv) = dv else {
            return Ok(None);
        };

        match &self.selected_indices {
            Some(selected_indices) => {
                // Filter the Deletion Vector (DV) table and map the old "idx" column to the
                // new "idx" column.
                //
                // Example, given:
                //   (all) paths = [0, 1, 2, 3]
                //   incoming DV table:
                //     (old) idx |   mask
                //         1     |  mask_1
                //         2     |  mask_2
                //         0     |  mask_0
                //   selected_indices = [0, 2, 3]
                //
                // Gets processed as follows:
                //   selected_indices gets mapped from old idx: [0, 2, 3] to new idx: [0, 1, 2]
                // and therefore:
                //   DV: (1, mask_1) => mapped to None => filtered out
                //   DV: (2, mask_2) => mapped to new idx 1 => retained
                //   DV: (0, mask_0) => mapped to new_idx 0 => retained
                //
                // Finally returns as:
                //   (new) idx |   mask
                //       1     |  mask_2
                //       0     |  mask_0

                let idx_map: PlHashMap<u64, u64> = selected_indices
                    .as_ref()
                    .iter()
                    .enumerate()
                    .map(|(out_idx, &source_idx)| Ok((source_idx as u64, out_idx as u64)))
                    .collect::<PolarsResult<_>>()?;

                let idx_col = dv.column("idx")?.cast(&DataType::UInt64)?;
                let idx_col = idx_col.u64()?;
                let remapped_idx: UInt64Chunked = idx_col
                    .iter()
                    .map(|opt_v| opt_v.and_then(|v| idx_map.get(&v).copied()))
                    .collect();
                let mask: BooleanChunked = remapped_idx.iter().map(|v| v.is_some()).collect();
                let dv = dv.with_column(remapped_idx.into_column().with_name("idx".into()))?;

                let filtered = dv.filter(&mask)?;
                Ok(Some(filtered))
            },
            None => Ok(Some(dv)),
        }
    }
}

impl std::hash::Hash for DeltaDeletionVectorProvider {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.callback.0.as_ptr() as usize).hash(state);
    }
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
