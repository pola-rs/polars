use polars_core::prelude::AnyValue;
use polars_error::{PolarsResult, polars_err};
use polars_plan::dsl::default_values::IcebergDefaultFieldValues;

#[derive(Debug, Clone, Copy)]
pub struct IcebergDefaultValueProviderRef<'a> {
    scan_source_idx: usize,
    default_values: &'a IcebergDefaultFieldValues,
}

impl<'a> IcebergDefaultValueProviderRef<'a> {
    pub fn new(default_values: &'a IcebergDefaultFieldValues, scan_source_idx: usize) -> Self {
        Self {
            scan_source_idx,
            default_values,
        }
    }
}

impl IcebergDefaultValueProviderRef<'_> {
    /// Note: `physical_id` should be a primitive typed field.
    pub fn get_default_value(&self, physical_id: u32) -> PolarsResult<Option<AnyValue<'_>>> {
        let IcebergDefaultFieldValues {
            identity_transformed_partition_fields,
            initial_defaults,
        } = self.default_values;

        if let Some(v) = identity_transformed_partition_fields.get(&physical_id) {
            let c = v.as_ref().map_err(|e| {
                polars_err!(
                    ComputeError:
                    "error loading identity transform value from metadata for missing field: \
                    {e}"
                )
            })?;

            // Note: `c` can be shorter than `scan_source_idx` if the iceberg partition field is deleted.
            return Ok(c.get(self.scan_source_idx).ok().filter(|av| !av.is_null()));
        } else if let Some(scalar) = initial_defaults.get(&physical_id)
            && !scalar.is_null()
        {
            return Ok(Some(scalar.as_any_value()));
        };

        Ok(None)
    }
}
