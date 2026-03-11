use std::sync::{Arc, OnceLock};

use arrow::array::BooleanArray;
use polars_core::frame::DataFrame;
use polars_core::prelude::{BooleanChunked, DataType, InitHashMaps, PlHashMap};
use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::python_function::PythonObject;

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

    /// Narrow the selected_indices, or initialize on first invocation. This supports
    /// incremental filtering. The calling site is responsible for the order of invocation
    /// and must ensure that indices are in range.
    pub fn narrow_selected_indices(mut self, indices: impl Iterator<Item = usize> + Clone) -> Self {
        let new_indices: Arc<[usize]> = match &self.selected_indices {
            Some(existing) => indices.map(|i| existing[i]).collect(),
            None => indices.collect::<Vec<_>>().into(),
        };
        self.selected_indices = Some(new_indices);
        self
    }

    pub fn call(&self) -> PolarsResult<PlHashMap<usize, BooleanChunked>> {
        let Some(dv) = (delta_dv_provider_vtable().unwrap().call)(&self.callback)? else {
            return Ok(PlHashMap::new());
        };

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

        // Build old_idx -> new_idx map from selected_indices.
        let idx_remap: Option<PlHashMap<usize, usize>> = self.selected_indices.as_ref().map(|v| {
            v.iter()
                .enumerate()
                .map(|(new_idx, &old_idx)| (old_idx, new_idx))
                .collect()
        });

        let idx_col = dv.column("idx")?.cast(&DataType::UInt64)?;
        let idx_col = idx_col.u64()?;
        let mask_col = dv.column("mask")?.list()?;

        let mut out = PlHashMap::new();

        for (idx, mask) in idx_col.iter().zip(mask_col.iter()) {
            let idx = idx.unwrap() as usize;

            let out_idx = match &idx_remap {
                Some(remap) => match remap.get(&idx) {
                    Some(&new_idx) => new_idx,
                    None => continue,
                },
                None => idx,
            };

            let mask = mask.unwrap();
            let mask_bool = mask.as_any().downcast_ref::<BooleanArray>().ok_or_else(
                || polars_err!(ComputeError: "expected boolean in Delta deletion vector mask"),
            )?;
            let chunked = unsafe {
                BooleanChunked::from_chunks(PlSmallStr::EMPTY, vec![Box::new(mask_bool.clone())])
            };
            out.insert(out_idx, chunked);
        }

        Ok(out)
    }

    pub fn callback(&self) -> &PythonObject {
        &self.callback
    }
}

impl std::hash::Hash for DeltaDeletionVectorProvider {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.callback.0.as_ptr() as usize).hash(state);
    }
}

impl std::fmt::Display for DeltaDeletionVectorProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeltaDeletionVectorCallback")
    }
}
