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
    pub fn has_non_null_initial_default(&self, physical_id: u32) -> bool {
        self.default_values
            .initial_defaults
            .get(&physical_id)
            .is_some_and(|v| !v.is_null())
    }

    /// Only a value from the file's partition metadata can establish that a missing
    /// ancestor struct was non-null. An initial default for a child cannot do so.
    pub fn has_non_null_identity_partition_value(&self, physical_id: u32) -> PolarsResult<bool> {
        if !self
            .default_values
            .identity_transformed_partition_fields
            .contains_key(&physical_id)
        {
            return Ok(false);
        }

        if let Some(present) = self
            .default_values
            .identity_partition_fields_present
            .get(&physical_id)
        {
            if present.bool()?.get(self.scan_source_idx) != Some(true) {
                return Ok(false);
            }
        } else if self.has_non_null_initial_default(physical_id) {
            // Older plugins coalesce partition values and initial defaults without reporting
            // which was used. Do not guess the validity of the parent in that case.
            return Err(polars_err!(ComputeError:
                "cannot reconstruct a missing Iceberg struct: identity partition presence \
                is unavailable for field {physical_id}; upgrade the Iceberg planner"
            ));
        }

        Ok(self.get_default_value(physical_id)?.is_some())
    }

    /// Note: `physical_id` should be a primitive typed field.
    pub fn get_default_value(&self, physical_id: u32) -> PolarsResult<Option<AnyValue<'_>>> {
        let IcebergDefaultFieldValues {
            identity_transformed_partition_fields,
            initial_defaults,
            ..
        } = self.default_values;

        if let Some(v) = identity_transformed_partition_fields.get(&physical_id) {
            let c = v.as_ref().map_err(|e| {
                polars_err!(
                    ComputeError:
                    "error loading identity transform value from metadata for missing field: \
                    {e}"
                )
            })?;

            // Note: `c` can be shorter than `scan_source_idx` if the iceberg partition field is
            // deleted; those sources take the `initial-default`. Within `c`, files of partition
            // specs without the identity field already hold it, so a null is a null.
            if let Ok(av) = c.get(self.scan_source_idx) {
                return Ok(Some(av).filter(|av| !av.is_null()));
            }
        }

        if let Some(scalar) = initial_defaults.get(&physical_id)
            && !scalar.is_null()
        {
            return Ok(Some(scalar.as_any_value()));
        };

        Ok(None)
    }
}
