use arrow::array::{BooleanArray, PrimitiveArray};

use super::ColumnPageStatistics;
use crate::parquet::indexes::PageIndex;

pub fn deserialize(indexes: &[PageIndex<bool>]) -> ColumnPageStatistics {
    ColumnPageStatistics {
        min: Box::new(BooleanArray::from_trusted_len_iter(
            indexes.iter().map(|index| index.min),
        )),
        max: Box::new(BooleanArray::from_trusted_len_iter(
            indexes.iter().map(|index| index.max),
        )),
        null_count: PrimitiveArray::from_trusted_len_iter(
            indexes
                .iter()
                .map(|index| index.null_count.map(|x| x as u64)),
        ),
    }
}
