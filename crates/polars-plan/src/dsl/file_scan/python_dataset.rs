use std::fmt::Debug;
use std::sync::OnceLock;

use polars_core::error::PolarsResult;
use polars_core::schema::SchemaRef;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::python_function::PythonObject;

use crate::dsl::DslPlan;

/// This is for `polars-python` to inject so that the implementation can be done there:
/// * The impls for converting from Python objects are there.
pub static DATASET_PROVIDER_VTABLE: OnceLock<PythonDatasetProviderVTable> = OnceLock::new();

pub struct PythonDatasetProviderVTable {
    pub reader_name: fn(dataset_object: &PythonObject) -> PlSmallStr,

    pub schema: fn(dataset_object: &PythonObject) -> PolarsResult<SchemaRef>,

    #[expect(clippy::type_complexity)]
    pub to_dataset_scan: fn(
        dataset_object: &PythonObject,
        limit: Option<usize>,
        projection: Option<&[PlSmallStr]>,
    ) -> PolarsResult<DslPlan>,
}

pub fn dataset_provider_vtable() -> Result<&'static PythonDatasetProviderVTable, &'static str> {
    DATASET_PROVIDER_VTABLE
        .get()
        .ok_or("DATASET_PROVIDER_VTABLE not initialized")
}

/// Currently intended only for Iceberg support
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PythonDatasetProvider {
    dataset_object: PythonObject,
}

impl PythonDatasetProvider {
    pub fn new(dataset_object: PythonObject) -> Self {
        Self { dataset_object }
    }

    pub fn reader_name(&self) -> PlSmallStr {
        (dataset_provider_vtable().unwrap().reader_name)(&self.dataset_object)
    }

    pub fn schema(&self) -> PolarsResult<SchemaRef> {
        (dataset_provider_vtable().unwrap().schema)(&self.dataset_object)
    }

    pub fn to_dataset_scan(
        &self,
        limit: Option<usize>,
        projection: Option<&[PlSmallStr]>,
    ) -> PolarsResult<DslPlan> {
        (dataset_provider_vtable().unwrap().to_dataset_scan)(
            &self.dataset_object,
            limit,
            projection,
        )
    }
}
