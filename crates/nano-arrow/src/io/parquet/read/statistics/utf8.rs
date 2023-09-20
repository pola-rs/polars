use parquet2::statistics::{BinaryStatistics, Statistics as ParquetStatistics};

use crate::array::{MutableArray, MutableUtf8Array};
use crate::error::Result;
use crate::offset::Offset;

pub(super) fn push<O: Offset>(
    from: Option<&dyn ParquetStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> Result<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutableUtf8Array<O>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutableUtf8Array<O>>()
        .unwrap();
    let from = from.map(|s| s.as_any().downcast_ref::<BinaryStatistics>().unwrap());

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
