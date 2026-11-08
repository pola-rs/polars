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
    /// Note: `physical_id` should be a primitive typed field, or a struct (whose default can only
    /// say whether it is null).
    pub fn get_default_value(&self, physical_id: u32) -> PolarsResult<Option<AnyValue<'_>>> {
        let IcebergDefaultFieldValues {
            identity_transformed_partition_fields,
            identity_partition_present: _,
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

    /// Whether the file has a (non-null) identity-partition value for `physical_id`, which a
    /// writer only gets from a non-null value of the field.
    pub fn has_identity_partition_value(&self, physical_id: u32) -> bool {
        // The values of files whose partition spec lacks the identity field are the
        // `initial-default`, not partition data.
        if let Some(present) = self
            .default_values
            .identity_partition_present
            .get(&physical_id)
            && !present
                .get(self.scan_source_idx)
                .is_ok_and(|av| matches!(av, AnyValue::Boolean(true)))
        {
            return false;
        }

        match self
            .default_values
            .identity_transformed_partition_fields
            .get(&physical_id)
        {
            // Loading errors surface in `get_default_value()`.
            Some(Err(_)) => true,
            Some(Ok(c)) => c.get(self.scan_source_idx).is_ok_and(|av| !av.is_null()),
            None => false,
        }
    }
}
