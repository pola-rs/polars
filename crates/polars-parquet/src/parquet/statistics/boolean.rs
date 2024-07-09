use parquet_format_safe::Statistics as ParquetStatistics;

use crate::parquet::error::{ParquetError, ParquetResult};

#[derive(Debug, Clone, PartialEq)]
pub struct BooleanStatistics {
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<bool>,
    pub min_value: Option<bool>,
}

impl BooleanStatistics {
    pub fn deserialize(v: &ParquetStatistics) -> ParquetResult<Self> {
        if let Some(ref v) = v.max_value {
            if v.len() != std::mem::size_of::<bool>() {
                return Err(ParquetError::oos(
                    "The max_value of statistics MUST be plain encoded",
                ));
            }
        };
        if let Some(ref v) = v.min_value {
            if v.len() != std::mem::size_of::<bool>() {
                return Err(ParquetError::oos(
                    "The min_value of statistics MUST be plain encoded",
                ));
            }
        };

        Ok(Self {
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
        })
    }

    pub fn serialize(&self) -> ParquetStatistics {
        ParquetStatistics {
            null_count: self.null_count,
            distinct_count: self.distinct_count,
            max_value: self.max_value.map(|x| vec![x as u8]),
            min_value: self.min_value.map(|x| vec![x as u8]),
            min: None,
            max: None,
        }
    }
}
