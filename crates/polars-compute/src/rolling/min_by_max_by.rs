use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::arg_min_max::ArgMinMaxWindow;
use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;

/// Rolling argmin/argmax over a `by` array, returning global indices.
///
/// Uses `ArgMinMaxWindow` (monotonic deque) for O(n) amortized complexity.
///
/// Preconditions:
/// - `starts` and `ends` must be monotonically non-decreasing (rolling window invariant).
/// - All indices in `starts`/`ends` must be within bounds of `by`.
fn rolling_arg_extremum_by<B: NativeType, P: MinMaxPolicy>(
    by: &[B],
    validity: Option<&Bitmap>,
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    assert_eq!(starts.len(), ends.len());
    let n = starts.len();

    if n == 0 || by.is_empty() {
        return PrimitiveArray::new_null(IdxSize::PRIMITIVE.into(), n);
    }

    let first_start = starts[0] as usize;
    let first_end = ends[0] as usize;

    match validity {
        None => {
            let mut window =
                <ArgMinMaxWindow<'_, B, P> as RollingAggWindowNoNulls<B, IdxSize>>::new(
                    by,
                    first_start,
                    first_end,
                    None,
                    None,
                );

            let iter = (0..n).map(|i| {
                let start = starts[i] as usize;
                let end = ends[i] as usize;

                if end <= start || (end - start) < min_periods {
                    return None;
                }

                // SAFETY: starts/ends are within bounds (rolling context).
                unsafe { RollingAggWindowNoNulls::<B, IdxSize>::update(&mut window, start, end) };

                RollingAggWindowNoNulls::<B, IdxSize>::get_agg(&window, i)
                    .map(|rel_idx| start as IdxSize + rel_idx)
            });

            PrimitiveArray::from_trusted_len_iter(iter)
        },
        Some(validity) => {
            let mut window = <ArgMinMaxWindow<'_, B, P> as RollingAggWindowNulls<B, IdxSize>>::new(
                by,
                validity,
                first_start,
                first_end,
                None,
                None,
            );

            let iter = (0..n).map(|i| {
                let start = starts[i] as usize;
                let end = ends[i] as usize;

                if end <= start {
                    return None;
                }

                // SAFETY: starts/ends are within bounds (rolling context).
                unsafe { RollingAggWindowNulls::<B, IdxSize>::update(&mut window, start, end) };

                if !RollingAggWindowNulls::<B, IdxSize>::is_valid(&window, min_periods) {
                    return None;
                }

                RollingAggWindowNulls::<B, IdxSize>::get_agg(&window, i)
                    .map(|rel_idx| start as IdxSize + rel_idx)
            });

            PrimitiveArray::from_trusted_len_iter(iter)
        },
    }
}

pub fn rolling_argmin_by<B: NativeType>(
    by: &[B],
    validity: Option<&Bitmap>,
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by::<B, MinPropagateNan>(by, validity, starts, ends, min_periods)
}

pub fn rolling_argmax_by<B: NativeType>(
    by: &[B],
    validity: Option<&Bitmap>,
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by::<B, MaxPropagateNan>(by, validity, starts, ends, min_periods)
}
