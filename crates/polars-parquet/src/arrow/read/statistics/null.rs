use arrow::array::*;
use polars_error::PolarsResult;

pub(super) fn push(min: &mut dyn MutableArray, max: &mut dyn MutableArray) -> PolarsResult<()> {
    let min = min.as_mut_any().downcast_mut::<MutableNullArray>().unwrap();
    let max = max.as_mut_any().downcast_mut::<MutableNullArray>().unwrap();
    min.push_null();
    max.push_null();

    Ok(())
}
