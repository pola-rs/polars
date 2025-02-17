#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use super::sort::SortOrder;

/// Column order that specifies what method was used to aggregate min/max values for
/// statistics.
///
/// If column order is undefined, then it is the legacy behaviour and all values should
/// be compared as signed values/bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum ColumnOrder {
    /// Column uses the order defined by its logical or physical type
    /// (if there is no logical type), parquet-format 2.4.0+.
    TypeDefinedOrder(SortOrder),
    /// Undefined column order, means legacy behaviour before parquet-format 2.4.0.
    /// Sort order is always SIGNED.
    Undefined,
}

impl ColumnOrder {
    /// Returns sort order associated with this column order.
    pub fn sort_order(&self) -> SortOrder {
        match *self {
            ColumnOrder::TypeDefinedOrder(order) => order,
            ColumnOrder::Undefined => SortOrder::Signed,
        }
    }
}
