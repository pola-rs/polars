use arrow::array::{MutableArray, MutableBinaryViewArray, ViewType};
use polars_error::PolarsResult;

use crate::parquet::statistics::{BinaryStatistics, Statistics as ParquetStatistics};

pub(super) fn push<T: ViewType + ?Sized>(
    from: Option<&dyn ParquetStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutableBinaryViewArray<T>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutableBinaryViewArray<T>>()
        .unwrap();
    let from = from.map(|s| s.as_any().downcast_ref::<BinaryStatistics>().unwrap());
    min.push(from.and_then(|s| {
        let opt_b = s.min_value.as_deref();
        unsafe { opt_b.map(|b| T::from_bytes_unchecked(b)) }
    }));
    max.push(from.and_then(|s| {
        let opt_b = s.max_value.as_deref();
        unsafe { opt_b.map(|b| T::from_bytes_unchecked(b)) }
    }));
    Ok(())
}
