use polars_core::schema::SchemaRef;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RowIndex {
    pub name: PlSmallStr,
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

impl HiveOptions {
    pub fn new_enabled() -> Self {
        Self {
            enabled: Some(true),
            hive_start_idx: 0,
            schema: None,
            try_parse_dates: true,
        }
    }

    pub fn new_disabled() -> Self {
        Self {
            enabled: Some(false),
            hive_start_idx: 0,
            schema: None,
            try_parse_dates: false,
        }
    }
}

impl Default for HiveOptions {
    fn default() -> Self {
        Self::new_enabled()
    }
}
