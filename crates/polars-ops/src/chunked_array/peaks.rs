use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

pub fn peak_min_max(
    column: &Column,
    start: &AnyValue<'_>,
    end: &AnyValue<'_>,
    is_peak_max: bool,
) -> PolarsResult<BooleanChunked> {
    let name = column.name().clone();
    let column = column.to_physical_repr();
    let column = column.as_materialized_series();
    match column.dtype() {
        dt if dt.is_bool() => {
            let series = column.cast(&DataType::Int8)?;
            let column = series.into_column();
            peak_min_max(&column, start, end, is_peak_max)
        },
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = column.as_ref().as_ref().as_ref();
                let start = start.extract();
                let end = end.extract();
                Ok(if is_peak_max {
                    peak_max_with_start_end(ca, start, end)
                } else {
                    peak_min_with_start_end(ca, start, end)
                }.with_name(name))
            })
        },
        dt => polars_bail!(opq = peak_max, dt),
    }
}

/// Get a boolean mask of the local maximum peaks.
pub fn peak_max_with_start_end<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    start: Option<T::Native>,
    end: Option<T::Native>,
) -> BooleanChunked
where
    ChunkedArray<T>: for<'a> ChunkCompareIneq<&'a ChunkedArray<T>, Item = BooleanChunked>,
{
    let shift_left = ca.shift_and_fill(1, start);
    let shift_right = ca.shift_and_fill(-1, end);
    ChunkedArray::lt(&shift_left, ca) & ChunkedArray::lt(&shift_right, ca)
}

/// Get a boolean mask of the local minimum peaks.
pub fn peak_min_with_start_end<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    start: Option<T::Native>,
    end: Option<T::Native>,
) -> BooleanChunked
where
    ChunkedArray<T>: for<'a> ChunkCompareIneq<&'a ChunkedArray<T>, Item = BooleanChunked>,
{
    let shift_left = ca.shift_and_fill(1, start);
    let shift_right = ca.shift_and_fill(-1, end);
    ChunkedArray::gt(&shift_left, ca) & ChunkedArray::gt(&shift_right, ca)
}
