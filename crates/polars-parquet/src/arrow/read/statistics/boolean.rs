use arrow::array::{MutableArray, MutableBooleanArray};
use polars_error::PolarsResult;

use crate::parquet::statistics::{BooleanStatistics, Statistics as ParquetStatistics};

pub(super) fn push(
    from: Option<&dyn ParquetStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutableBooleanArray>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutableBooleanArray>()
        .unwrap();
    let from = from.map(|s| s.as_any().downcast_ref::<BooleanStatistics>().unwrap());
    min.push(from.and_then(|s| s.min_value));
    max.push(from.and_then(|s| s.max_value));
    Ok(())
}
