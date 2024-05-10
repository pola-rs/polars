use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RowIndex {
    pub name: Arc<str>,
    pub offset: IdxSize,
}

/// Options for Hive partitioning.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HiveOptions {
    pub enabled: bool,
    pub schema: Option<SchemaRef>,
}

impl Default for HiveOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            schema: None,
        }
    }
}
