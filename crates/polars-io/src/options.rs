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
    /// This can be `None` to automatically enable for single directory scans
    /// and disable otherwise. However it should be initialized if it is inside
    /// a DSL / IR plan.
    pub enabled: Option<bool>,
    pub hive_start_idx: usize,
    pub schema: Option<SchemaRef>,
    pub try_parse_dates: bool,
}

impl Default for HiveOptions {
    fn default() -> Self {
        Self {
            enabled: Some(true),
            hive_start_idx: 0,
            schema: None,
            try_parse_dates: true,
        }
    }
}
