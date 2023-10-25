use std::sync::Arc;

use parquet_format_safe::Statistics as ParquetStatistics;

use super::Statistics;
use crate::{
    error::Result,
    schema::types::{PhysicalType, PrimitiveType},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryStatistics {
    pub primitive_type: PrimitiveType,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<Vec<u8>>,
    pub min_value: Option<Vec<u8>>,
}

impl Statistics for BinaryStatistics {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &PhysicalType::ByteArray
    }

    fn null_count(&self) -> Option<i64> {
        self.null_count
    }
}

pub fn read(v: &ParquetStatistics, primitive_type: PrimitiveType) -> Result<Arc<dyn Statistics>> {
    Ok(Arc::new(BinaryStatistics {
        primitive_type,
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.clone(),
        min_value: v.min_value.clone(),
    }))
}

pub fn write(v: &BinaryStatistics) -> ParquetStatistics {
    ParquetStatistics {
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.clone(),
        min_value: v.min_value.clone(),
        min: None,
        max: None,
    }
}
