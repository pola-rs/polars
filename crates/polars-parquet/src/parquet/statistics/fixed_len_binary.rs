use std::sync::Arc;

use parquet_format_safe::Statistics as ParquetStatistics;

use super::Statistics;
use crate::{
    error::{Error, Result},
    schema::types::{PhysicalType, PrimitiveType},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedLenStatistics {
    pub primitive_type: PrimitiveType,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<Vec<u8>>,
    pub min_value: Option<Vec<u8>>,
}

impl Statistics for FixedLenStatistics {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &self.primitive_type.physical_type
    }

    fn null_count(&self) -> Option<i64> {
        self.null_count
    }
}

pub fn read(
    v: &ParquetStatistics,
    size: usize,
    primitive_type: PrimitiveType,
) -> Result<Arc<dyn Statistics>> {
    if let Some(ref v) = v.max_value {
        if v.len() != size {
            return Err(Error::oos(
                "The max_value of statistics MUST be plain encoded",
            ));
        }
    };
    if let Some(ref v) = v.min_value {
        if v.len() != size {
            return Err(Error::oos(
                "The min_value of statistics MUST be plain encoded",
            ));
        }
    };

    Ok(Arc::new(FixedLenStatistics {
        primitive_type,
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.clone().map(|mut x| {
            x.truncate(size);
            x
        }),
        min_value: v.min_value.clone().map(|mut x| {
            x.truncate(size);
            x
        }),
    }))
}

pub fn write(v: &FixedLenStatistics) -> ParquetStatistics {
    ParquetStatistics {
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.clone(),
        min_value: v.min_value.clone(),
        min: None,
        max: None,
    }
}
