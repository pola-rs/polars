use std::sync::Arc;

use parquet_format_safe::Statistics as ParquetStatistics;

use super::Statistics;
use crate::error::{Error, Result};
use crate::schema::types::{PhysicalType, PrimitiveType};
use crate::types;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveStatistics<T: types::NativeType> {
    pub primitive_type: PrimitiveType,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub min_value: Option<T>,
    pub max_value: Option<T>,
}

impl<T: types::NativeType> Statistics for PrimitiveStatistics<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &T::TYPE
    }

    fn null_count(&self) -> Option<i64> {
        self.null_count
    }
}

pub fn read<T: types::NativeType>(
    v: &ParquetStatistics,
    primitive_type: PrimitiveType,
) -> Result<Arc<dyn Statistics>> {
    if let Some(ref v) = v.max_value {
        if v.len() != std::mem::size_of::<T>() {
            return Err(Error::oos(
                "The max_value of statistics MUST be plain encoded",
            ));
        }
    };
    if let Some(ref v) = v.min_value {
        if v.len() != std::mem::size_of::<T>() {
            return Err(Error::oos(
                "The min_value of statistics MUST be plain encoded",
            ));
        }
    };

    Ok(Arc::new(PrimitiveStatistics::<T> {
        primitive_type,
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.as_ref().map(|x| types::decode(x)),
        min_value: v.min_value.as_ref().map(|x| types::decode(x)),
    }))
}

pub fn write<T: types::NativeType>(v: &PrimitiveStatistics<T>) -> ParquetStatistics {
    ParquetStatistics {
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.map(|x| x.to_le_bytes().as_ref().to_vec()),
        min_value: v.min_value.map(|x| x.to_le_bytes().as_ref().to_vec()),
        min: None,
        max: None,
    }
}
