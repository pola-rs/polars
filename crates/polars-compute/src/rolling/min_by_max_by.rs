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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_argmin_by_basic() {
        let by: Vec<i32> = vec![10, 3, 7, 1, 5];
        let starts = vec![0 as IdxSize, 0, 0, 1, 2];
        let ends = vec![1 as IdxSize, 2, 3, 4, 5];

        let result = rolling_argmin_by(&by, None, &starts, &ends, 1);
        let actual: Vec<Option<IdxSize>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(
            actual,
            vec![
                Some(0 as IdxSize),
                Some(1 as IdxSize),
                Some(1 as IdxSize),
                Some(3 as IdxSize),
                Some(3 as IdxSize)
            ]
        );
    }

    #[test]
    fn test_rolling_argmax_by_basic() {
        let by: Vec<i32> = vec![10, 3, 7, 1, 5];
        let starts = vec![0 as IdxSize, 0, 0, 1, 2];
        let ends = vec![1 as IdxSize, 2, 3, 4, 5];

        let result = rolling_argmax_by(&by, None, &starts, &ends, 1);
        let actual: Vec<Option<IdxSize>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(
            actual,
            vec![
                Some(0 as IdxSize),
                Some(0 as IdxSize),
                Some(0 as IdxSize),
                Some(2 as IdxSize),
                Some(2 as IdxSize)
            ]
        );
    }

    #[test]
    fn test_rolling_argmin_by_min_periods() {
        let by: Vec<i32> = vec![10, 3, 7, 1, 5];
        let starts = vec![0 as IdxSize, 0, 0, 1, 2];
        let ends = vec![1 as IdxSize, 2, 3, 4, 5];

        let result = rolling_argmin_by(&by, None, &starts, &ends, 3);
        let actual: Vec<Option<IdxSize>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(
            actual,
            vec![
                None,
                None,
                Some(1 as IdxSize),
                Some(3 as IdxSize),
                Some(3 as IdxSize)
            ]
        );
    }

    #[test]
    fn test_rolling_argmin_by_with_nulls() {
        // by = [10, NULL, 7, 1, 5]
        let by: Vec<i32> = vec![10, 0, 7, 1, 5];
        let validity = Bitmap::from(&[true, false, true, true, true]);
        let starts = vec![0 as IdxSize, 0, 0, 1, 2];
        let ends = vec![1 as IdxSize, 2, 3, 4, 5];

        let result = rolling_argmin_by(&by, Some(&validity), &starts, &ends, 1);
        let actual: Vec<Option<IdxSize>> = result.iter().map(|v| v.copied()).collect();
        // Window [0,1): [10] -> min=10 @ 0
        // Window [0,2): [10, NULL] -> min=10 @ 0
        // Window [0,3): [10, NULL, 7] -> min=7 @ 2
        // Window [1,4): [NULL, 7, 1] -> min=1 @ 3
        // Window [2,5): [7, 1, 5] -> min=1 @ 3
        assert_eq!(
            actual,
            vec![
                Some(0 as IdxSize),
                Some(0 as IdxSize),
                Some(2 as IdxSize),
                Some(3 as IdxSize),
                Some(3 as IdxSize)
            ]
        );
    }

    #[test]
    fn test_rolling_argmin_by_empty() {
        let by: Vec<i32> = vec![];
        let starts: Vec<IdxSize> = vec![];
        let ends: Vec<IdxSize> = vec![];

        let result = rolling_argmin_by(&by, None, &starts, &ends, 1);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_rolling_argmin_by_empty_windows() {
        let by: Vec<i32> = vec![10, 3, 7];
        let starts = vec![0 as IdxSize, 1, 1];
        let ends = vec![0 as IdxSize, 1, 1];

        let result = rolling_argmin_by(&by, None, &starts, &ends, 1);
        let actual: Vec<Option<IdxSize>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(actual, vec![None, None, None]);
    }

    #[test]
    fn test_rolling_argmin_by_float_nan() {
        let by: Vec<f64> = vec![1.0, f64::NAN, 3.0];
        let starts = vec![0 as IdxSize, 0, 1];
        let ends = vec![1 as IdxSize, 2, 3];

        let result = rolling_argmin_by(&by, None, &starts, &ends, 1);
        let actual: Vec<Option<IdxSize>> = result.iter().map(|v| v.copied()).collect();
        // NaN is "better" (propagated) for MinPropagateNan
        assert_eq!(actual[0], Some(0 as IdxSize)); // just 1.0
        assert_eq!(actual[1], Some(1 as IdxSize)); // NaN wins over 1.0
        assert_eq!(actual[2], Some(1 as IdxSize)); // NaN wins over 3.0
    }
}
