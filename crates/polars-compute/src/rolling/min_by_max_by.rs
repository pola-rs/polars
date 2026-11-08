use polars_arrow::array::PrimitiveArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::types::NativeType;
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxIgnoreNan, MinIgnoreNan, MinMaxPolicy};

use super::arg_min_max::ArgMinMaxWindow;
use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;

/// Rolling argmin/argmax over a `by` array, returning global indices.
///
/// Uses `ArgMinMaxWindow` (monotonic deque) for O(n) amortized complexity.
///
/// Preconditions:
/// - The `[start, len]` windows must be monotonically non-decreasing (rolling window invariant).
/// - All windows must be within bounds of `by`.
fn rolling_arg_extremum_by<B: NativeType, P: MinMaxPolicy>(
    by: &[B],
    validity: Option<&Bitmap>,
    windows: &[[IdxSize; 2]],
    min_samples: usize,
) -> PrimitiveArray<IdxSize> {
    let n = windows.len();

    if n == 0 || by.is_empty() {
        return PrimitiveArray::new_null(IdxSize::PRIMITIVE.into(), n);
    }

    let [first_start, first_len] = windows[0];
    let first_start = first_start as usize;
    let first_end = first_start + first_len as usize;

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

            let iter = windows.iter().enumerate().map(|(i, &[start, len])| {
                let start = start as usize;
                let end = start + len as usize;

                if end <= start || (end - start) < min_samples {
                    return None;
                }

                // SAFETY: The windows are within bounds (rolling context).
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

            let iter = windows.iter().enumerate().map(|(i, &[start, len])| {
                let start = start as usize;
                let end = start + len as usize;

                if end <= start {
                    return None;
                }

                // SAFETY: The windows are within bounds (rolling context).
                unsafe { RollingAggWindowNulls::<B, IdxSize>::update(&mut window, start, end) };

                if !RollingAggWindowNulls::<B, IdxSize>::is_valid(&window, min_samples) {
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
    windows: &[[IdxSize; 2]],
    min_samples: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by::<B, MinIgnoreNan>(by, validity, windows, min_samples)
}

pub fn rolling_argmax_by<B: NativeType>(
    by: &[B],
    validity: Option<&Bitmap>,
    windows: &[[IdxSize; 2]],
    min_samples: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by::<B, MaxIgnoreNan>(by, validity, windows, min_samples)
}
