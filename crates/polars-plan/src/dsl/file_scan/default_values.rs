use std::sync::Arc;

use polars_core::prelude::{Column, PlIndexMap};

/// Default field values when they are missing from the data file.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DefaultFieldValues {
    /// This is to follow the spec for missing columns:
    /// * Return the value from partition metadata if an Identity Transform exists for the field
    ///
    /// Note: This is not the Iceberg V3 `initial-default`.
    Iceberg(Arc<IcebergIdentityTransformedPartitionFields>),
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct IcebergIdentityTransformedPartitionFields(pub PlIndexMap<u32, Result<Column, String>>);

impl Eq for IcebergIdentityTransformedPartitionFields {}

impl std::hash::Hash for IcebergIdentityTransformedPartitionFields {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for key in self.keys() {
            key.hash(state);
        }
    }
}

impl std::ops::Deref for IcebergIdentityTransformedPartitionFields {
    type Target = PlIndexMap<u32, Result<Column, String>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for IcebergIdentityTransformedPartitionFields {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
