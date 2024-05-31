use arrow::array::{MutableArray, MutableUtf8Array};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use crate::parquet::statistics::BinaryStatistics;

pub(super) fn push<O: Offset>(
    from: Option<&BinaryStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutableUtf8Array<O>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutableUtf8Array<O>>()
        .unwrap();

    min.push(
        from.and_then(|s| s.min_value.as_deref().map(simdutf8::basic::from_utf8))
            .transpose()?,
    );
    max.push(
        from.and_then(|s| s.max_value.as_deref().map(simdutf8::basic::from_utf8))
            .transpose()?,
    );

    Ok(())
}
