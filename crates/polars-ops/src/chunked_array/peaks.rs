use polars_arrow::array::BooleanArray;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;

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

/// `is_peak(neighbor, value)` must hold for both neighbors. A missing neighbor counts as
/// unknown, so the result is null unless the other neighbor already fails.
fn peak_mask<N: Copy>(
    name: PlSmallStr,
    values: &[N],
    start: Option<N>,
    end: Option<N>,
    is_peak: impl Fn(&N, &N) -> bool,
) -> BooleanChunked {
    let last = values.len().wrapping_sub(1);
    let arr: BooleanArray = (0..values.len())
        .map(|i| {
            // SAFETY: `i` and its neighbors, when in range, are in bounds.
            unsafe {
                let value = values.get_unchecked(i);
                let left = if i == 0 {
                    start
                } else {
                    Some(*values.get_unchecked(i - 1))
                };
                let right = if i == last {
                    end
                } else {
                    Some(*values.get_unchecked(i + 1))
                };
                let left = left.map(|l| is_peak(&l, value));
                let right = right.map(|r| is_peak(&r, value));
                match (left, right) {
                    (Some(l), Some(r)) => Some(l && r),
                    (Some(false), None) | (None, Some(false)) => Some(false),
                    _ => None,
                }
            }
        })
        .collect_arr();
    BooleanChunked::with_chunk(name, arr)
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
    if let Ok(values) = ca.cont_slice() {
        return peak_mask(ca.name().clone(), values, start, end, |a, b| a.tot_lt(b));
    }
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
    if let Ok(values) = ca.cont_slice() {
        return peak_mask(ca.name().clone(), values, start, end, |a, b| a.tot_gt(b));
    }
    let shift_left = ca.shift_and_fill(1, start);
    let shift_right = ca.shift_and_fill(-1, end);
    ChunkedArray::gt(&shift_left, ca) & ChunkedArray::gt(&shift_right, ca)
}
