use crate::array::{MutableArray, MutableBooleanArray};
use parquet2::statistics::{BooleanStatistics, Statistics as ParquetStatistics};

use crate::error::Result;

pub(super) fn push(
    from: Option<&dyn ParquetStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> Result<()> {
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
