use parquet_format_safe::Statistics as ParquetStatistics;

use crate::parquet::error::ParquetResult;
use crate::parquet::schema::types::PrimitiveType;

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryStatistics {
    pub primitive_type: PrimitiveType,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<Vec<u8>>,
    pub min_value: Option<Vec<u8>>,
}

impl BinaryStatistics {
    pub fn deserialize(
        v: &ParquetStatistics,
        primitive_type: PrimitiveType,
    ) -> ParquetResult<Self> {
        Ok(BinaryStatistics {
            primitive_type,
            null_count: v.null_count,
            distinct_count: v.distinct_count,
            max_value: v.max_value.clone(),
            min_value: v.min_value.clone(),
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
