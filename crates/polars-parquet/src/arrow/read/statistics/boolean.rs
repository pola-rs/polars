use arrow::array::{MutableArray, MutableBooleanArray};
use polars_error::PolarsResult;

use crate::parquet::statistics::BooleanStatistics;

pub(super) fn push(
    from: Option<&BooleanStatistics>,
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

    min.push(from.and_then(|s| s.min_value));
    max.push(from.and_then(|s| s.max_value));

    Ok(())
}
