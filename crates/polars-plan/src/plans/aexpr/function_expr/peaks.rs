use polars_ops::chunked_array::peaks::peak_min_max;

use super::*;

pub(super) fn peak_min(s: &Column) -> PolarsResult<Column> {
    peak_min_max(s, &AnyValue::Int8(0), &AnyValue::Int8(0), false).map(IntoColumn::into_column)
}

pub(super) fn peak_max(s: &Column) -> PolarsResult<Column> {
    peak_min_max(s, &AnyValue::Int8(0), &AnyValue::Int8(0), true).map(IntoColumn::into_column)
}
