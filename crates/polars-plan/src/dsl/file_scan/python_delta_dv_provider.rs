use std::sync::OnceLock;

use arrow::array::ListArray;
use polars_buffer::Buffer;
use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::pl_path::PlRefPath;
use polars_utils::python_function::PythonObject;

/// This is for `polars-python` to inject so that the implementation can be done there:
/// * The impls for converting from Python objects are there.
pub static DELTA_DV_PROVIDER_VTABLE: OnceLock<DeltaDeletionVectorProviderVTable> = OnceLock::new();

pub struct DeltaDeletionVectorProviderVTable {
    pub call:
        fn(callback: &PythonObject, paths: Buffer<PlRefPath>) -> PolarsResult<Option<DataFrame>>,
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
}

impl DeltaDeletionVectorProvider {
    pub fn new(callback: PythonObject) -> Self {
        Self { callback }
    }

    /// Return the deletion vector as Boolean list the selected_paths, maintaining the path order.
    pub fn call(&self, selected_paths: Buffer<PlRefPath>) -> PolarsResult<Option<ListArray<i64>>> {
        let Some(dv) =
            (delta_dv_provider_vtable().unwrap().call)(&self.callback, selected_paths.clone())?
        else {
            return Ok(None);
        };

        if selected_paths.len() != dv.height() {
            polars_bail!(ComputeError:
                "delta deletion vector file count must match: expected {}, got {}", 
                selected_paths.len(), dv.height());
        };

        let mask_col = dv.column("selection_vector")?.list()?;

        if mask_col.null_count() == selected_paths.len() {
            return Ok(None);
        };

        let arr = mask_col.rechunk();
        let out = arr.downcast_as_array().clone();
        Ok(Some(out))
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
