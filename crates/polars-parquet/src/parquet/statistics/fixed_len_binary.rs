use parquet_format_safe::Statistics as ParquetStatistics;

use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::types::PrimitiveType;

#[derive(Debug, Clone, PartialEq)]
pub struct FixedLenStatistics {
    pub primitive_type: PrimitiveType,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<Vec<u8>>,
    pub min_value: Option<Vec<u8>>,
}

impl FixedLenStatistics {
    pub fn deserialize(
        v: &ParquetStatistics,
        size: usize,
        primitive_type: PrimitiveType,
    ) -> ParquetResult<Self> {
        if let Some(ref v) = v.max_value {
            if v.len() != size {
                return Err(ParquetError::oos(
                    "The max_value of statistics MUST be plain encoded",
                ));
            }
        };
        if let Some(ref v) = v.min_value {
            if v.len() != size {
                return Err(ParquetError::oos(
                    "The min_value of statistics MUST be plain encoded",
                ));
            }
        };

        Ok(Self {
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
        })
    }

    pub fn serialize(&self) -> ParquetStatistics {
        ParquetStatistics {
            null_count: self.null_count,
            distinct_count: self.distinct_count,
            max_value: self.max_value.clone(),
            min_value: self.min_value.clone(),
            min: None,
            max: None,
        }
    }
}
