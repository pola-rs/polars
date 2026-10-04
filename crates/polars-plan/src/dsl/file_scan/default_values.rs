use std::hash::Hash;
use std::sync::Arc;

use polars_core::prelude::Column;
use polars_core::scalar::Scalar;
use polars_utils::aliases::PlIndexMapHashable;

/// Default field values when they are missing from the data file.
#[derive(Debug, Clone, Eq, Hash, PartialEq, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DefaultFieldValues {
    /// This is to follow the spec for missing columns:
    /// * Return the value from partition metadata if an Identity Transform exists for the field
    ///
    /// Note: This is not the Iceberg V3 `initial-default`.
    Iceberg(Arc<IcebergDefaultFieldValues>),
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct IcebergDefaultFieldValues {
    pub identity_transformed_partition_fields: PlIndexMapHashable<u32, Result<Column, String>>,
    pub initial_defaults: PlIndexMapHashable<u32, Scalar>,
}

impl IcebergDefaultFieldValues {
    pub fn is_empty(&self) -> bool {
        let IcebergDefaultFieldValues {
            identity_transformed_partition_fields,
            initial_defaults,
        } = self;
        identity_transformed_partition_fields.is_empty() && initial_defaults.is_empty()
    }
}
