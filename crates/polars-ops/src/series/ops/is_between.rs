use std::ops::BitAnd;

use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_defs::expr::ClosedInterval;
use polars_utils::total_ord::TotalOrd;

fn is_between_scalar_bounds<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    lower: T::Native,
    upper: T::Native,
    closed: ClosedInterval,
) -> BooleanChunked {
    let (left_open, right_open) = match closed {
        ClosedInterval::None => (true, true),
        ClosedInterval::Left => (false, true),
        ClosedInterval::Right => (true, false),
        ClosedInterval::Both => (false, false),
    };
    let chunks = ca.downcast_iter().map(|arr| {
        let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|x| {
            let above = if left_open {
                x.tot_gt(&lower)
            } else {
                x.tot_ge(&lower)
            };
            let below = if right_open {
                x.tot_lt(&upper)
            } else {
                x.tot_le(&upper)
            };
            above & below
        }));
        BooleanArray::from_data_default(values, arr.validity().cloned())
    });
    BooleanChunked::from_chunk_iter(ca.name().clone(), chunks)
}

pub fn is_between(
    s: &Series,
    lower: &Series,
    upper: &Series,
    closed: ClosedInterval,
) -> PolarsResult<BooleanChunked> {
    if s.dtype().is_primitive_numeric()
        && lower.dtype() == s.dtype()
        && upper.dtype() == s.dtype()
        && lower.len() == 1
        && upper.len() == 1
        && lower.null_count() == 0
        && upper.null_count() == 0
    {
        return Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
            let lower: &ChunkedArray<$T> = lower.as_ref().as_ref();
            let upper: &ChunkedArray<$T> = upper.as_ref().as_ref();
            is_between_scalar_bounds(
                ca,
                lower.get(0).unwrap(),
                upper.get(0).unwrap(),
                closed,
            )
        }));
    }

    let left_cmp_op = match closed {
        ClosedInterval::None | ClosedInterval::Right => Series::gt,
        ClosedInterval::Both | ClosedInterval::Left => Series::gt_eq,
    };
    let right_cmp_op = match closed {
        ClosedInterval::None | ClosedInterval::Left => Series::lt,
        ClosedInterval::Both | ClosedInterval::Right => Series::lt_eq,
    };
    let left = left_cmp_op(s, lower)?;
    let right = right_cmp_op(s, upper)?;
    Ok(left.bitand(right))
}
