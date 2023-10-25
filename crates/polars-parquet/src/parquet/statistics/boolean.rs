use std::sync::Arc;

use parquet_format_safe::Statistics as ParquetStatistics;

use super::Statistics;
use crate::{
    error::{Error, Result},
    schema::types::PhysicalType,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanStatistics {
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<bool>,
    pub min_value: Option<bool>,
}

impl Statistics for BooleanStatistics {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &PhysicalType::Boolean
    }

    fn null_count(&self) -> Option<i64> {
        self.null_count
    }
}

pub fn read(v: &ParquetStatistics) -> Result<Arc<dyn Statistics>> {
    if let Some(ref v) = v.max_value {
        if v.len() != std::mem::size_of::<bool>() {
            return Err(Error::oos(
                "The max_value of statistics MUST be plain encoded",
            ));
        }
    };
    if let Some(ref v) = v.min_value {
        if v.len() != std::mem::size_of::<bool>() {
            return Err(Error::oos(
                "The min_value of statistics MUST be plain encoded",
            ));
        }
    };

    Ok(Arc::new(BooleanStatistics {
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v
            .max_value
            .as_ref()
            .and_then(|x| x.first())
            .map(|x| *x != 0),
        min_value: v
            .min_value
            .as_ref()
            .and_then(|x| x.first())
            .map(|x| *x != 0),
    }))
}

pub fn write(v: &BooleanStatistics) -> ParquetStatistics {
    ParquetStatistics {
        null_count: v.null_count,
        distinct_count: v.distinct_count,
        max_value: v.max_value.map(|x| vec![x as u8]),
        min_value: v.min_value.map(|x| vec![x as u8]),
        min: None,
        max: None,
    }
}
