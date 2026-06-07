#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::sort::SortOrder;

/// Column order that specifies what method was used to aggregate min/max values for
/// statistics.
///
/// If column order is undefined, then it is the legacy behaviour and all values should
/// be compared as signed values/bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum ColumnOrder {
    /// Column uses the order defined by its logical or physical type
    /// (if there is no logical type), parquet-format 2.4.0+.
    TypeDefinedOrder(SortOrder),
    /// IEEE 754 total order for float columns (PARQUET-2249). Min/max include
    /// NaN; NaN presence is reported by the statistics `nan_count` field.
    IEEE754TotalOrder,
    /// Undefined column order, means legacy behaviour before parquet-format 2.4.0.
    /// Sort order is always SIGNED.
    Undefined,
}

impl ColumnOrder {
    /// Returns sort order associated with this column order.
    pub fn sort_order(&self) -> SortOrder {
        match *self {
            ColumnOrder::TypeDefinedOrder(order) => order,
            ColumnOrder::IEEE754TotalOrder => SortOrder::Signed,
            ColumnOrder::Undefined => SortOrder::Signed,
        }
    }
}

/// Decoded `ColumnOrder` union tag, resolved to a public [`ColumnOrder`] (with
/// the type-dependent [`SortOrder`]) in `parse_column_orders`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnOrderTag {
    TypeDefined,
    IEEE754TotalOrder,
}
