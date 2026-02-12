use polars_core::prelude::AnyValue;
use polars_error::{PolarsResult, polars_err};
use polars_plan::dsl::default_values::IcebergIdentityTransformedPartitionFields;

#[derive(Debug, Clone, Copy)]
pub struct IcebergDefaultValueProviderRef<'a> {
    scan_source_idx: usize,
    identity_transformed_values: &'a IcebergIdentityTransformedPartitionFields,
}

impl<'a> IcebergDefaultValueProviderRef<'a> {
    pub fn new(
        identity_transformed_values: &'a IcebergIdentityTransformedPartitionFields,
        scan_source_idx: usize,
    ) -> Self {
        Self {
            scan_source_idx,
            identity_transformed_values,
        }
    }
}

impl IcebergDefaultValueProviderRef<'_> {
    /// Note: `physical_id` should be a primitive typed field.
    pub fn get_default_value(&self, physical_id: u32) -> PolarsResult<Option<AnyValue<'_>>> {
        let Some(v) = self.identity_transformed_values.get(&physical_id) else {
            return Ok(None);
        };

        let c = v.as_ref().map_err(|e| {
            polars_err!(
                ComputeError:
                "error loading identity transform value from metadata for missing field: \
                {e}"
            )
        })?;

        // Note: `c` can be shorter than `scan_source_idx` if the iceberg partition field is deleted.
        Ok(c.get(self.scan_source_idx).ok().filter(|av| !av.is_null()))
    }
}
