use parquet_format_safe::Statistics as ParquetStatistics;

use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::types;

#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveStatistics<T: types::NativeType> {
    pub primitive_type: PrimitiveType,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub min_value: Option<T>,
    pub max_value: Option<T>,
}

impl<T: types::NativeType> PrimitiveStatistics<T> {
    pub fn deserialize(
        v: &ParquetStatistics,
        primitive_type: PrimitiveType,
    ) -> ParquetResult<Self> {
        if v.max_value
            .as_ref()
            .is_some_and(|v| v.len() != std::mem::size_of::<T>())
        {
            return Err(ParquetError::oos(
                "The max_value of statistics MUST be plain encoded",
            ));
        };
        if v.min_value
            .as_ref()
            .is_some_and(|v| v.len() != std::mem::size_of::<T>())
        {
            return Err(ParquetError::oos(
                "The min_value of statistics MUST be plain encoded",
            ));
        };

        Ok(Self {
            primitive_type,
            null_count: v.null_count,
            distinct_count: v.distinct_count,
            max_value: v.max_value.as_ref().map(|x| types::decode(x)),
            min_value: v.min_value.as_ref().map(|x| types::decode(x)),
        })
    }

    pub fn serialize(&self) -> ParquetStatistics {
        ParquetStatistics {
            null_count: self.null_count,
            distinct_count: self.distinct_count,
            max_value: self.max_value.map(|x| x.to_le_bytes().as_ref().to_vec()),
            min_value: self.min_value.map(|x| x.to_le_bytes().as_ref().to_vec()),
            min: None,
            max: None,
        }
    }
}
