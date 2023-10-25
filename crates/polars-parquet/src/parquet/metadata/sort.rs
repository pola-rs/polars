#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use crate::parquet::schema::types::{
    IntegerType, PhysicalType, PrimitiveConvertedType, PrimitiveLogicalType,
};

/// Sort order for page and column statistics.
///
/// Types are associated with sort orders and column stats are aggregated using a sort
/// order, and a sort order should be considered when comparing values with statistics
/// min/max.
///
/// See reference in
/// <https://github.com/apache/parquet-cpp/blob/master/src/parquet/types.h>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum SortOrder {
    /// Signed (either value or legacy byte-wise) comparison.
    Signed,
    /// Unsigned (depending on physical type either value or byte-wise) comparison.
    Unsigned,
    /// Comparison is undefined.
    Undefined,
}

/// Returns sort order for a physical/logical type.
pub fn get_sort_order(
    logical_type: &Option<PrimitiveLogicalType>,
    converted_type: &Option<PrimitiveConvertedType>,
    physical_type: &PhysicalType,
) -> SortOrder {
    if let Some(logical_type) = logical_type {
        return get_logical_sort_order(logical_type);
    };
    if let Some(converted_type) = converted_type {
        return get_converted_sort_order(converted_type);
    };
    get_physical_sort_order(physical_type)
}

fn get_logical_sort_order(logical_type: &PrimitiveLogicalType) -> SortOrder {
    // TODO: Should this take converted and logical type, for compatibility?
    use PrimitiveLogicalType::*;
    match logical_type {
        String | Enum | Json | Bson => SortOrder::Unsigned,
        Integer(t) => match t {
            IntegerType::Int8 | IntegerType::Int16 | IntegerType::Int32 | IntegerType::Int64 => {
                SortOrder::Signed
            },
            _ => SortOrder::Unsigned,
        },
        Decimal(_, _) => SortOrder::Signed,
        Date => SortOrder::Signed,
        Time { .. } => SortOrder::Signed,
        Timestamp { .. } => SortOrder::Signed,
        Unknown => SortOrder::Undefined,
        Uuid => SortOrder::Unsigned,
    }
}

fn get_converted_sort_order(converted_type: &PrimitiveConvertedType) -> SortOrder {
    use PrimitiveConvertedType::*;
    match converted_type {
        // Unsigned byte-wise comparison.
        Utf8 | Json | Bson | Enum => SortOrder::Unsigned,
        Int8 | Int16 | Int32 | Int64 => SortOrder::Signed,
        Uint8 | Uint16 | Uint32 | Uint64 => SortOrder::Unsigned,
        // Signed comparison of the represented value.
        Decimal(_, _) => SortOrder::Signed,
        Date => SortOrder::Signed,
        TimeMillis | TimeMicros | TimestampMillis | TimestampMicros => SortOrder::Signed,
        Interval => SortOrder::Undefined,
    }
}

fn get_physical_sort_order(physical_type: &PhysicalType) -> SortOrder {
    use PhysicalType::*;
    match physical_type {
        // Order: false, true
        Boolean => SortOrder::Unsigned,
        Int32 | Int64 => SortOrder::Signed,
        Int96 => SortOrder::Undefined,
        // Notes to remember when comparing float/double values:
        // If the min is a NaN, it should be ignored.
        // If the max is a NaN, it should be ignored.
        // If the min is +0, the row group may contain -0 values as well.
        // If the max is -0, the row group may contain +0 values as well.
        // When looking for NaN values, min and max should be ignored.
        Float | Double => SortOrder::Signed,
        // Unsigned byte-wise comparison
        ByteArray | FixedLenByteArray(_) => SortOrder::Unsigned,
    }
}
