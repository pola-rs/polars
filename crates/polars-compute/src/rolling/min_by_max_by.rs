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
fn rolling_arg_extremum_by_no_nulls<B: NativeType, P: MinMaxPolicy>(
    by: &[B],
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
    let mut window = <ArgMinMaxWindow<'_, B, P> as RollingAggWindowNoNulls<B, IdxSize>>::new(
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

        // SAFETY: starts/ends are guaranteed to be within bounds by the caller (rolling context).
        unsafe { RollingAggWindowNoNulls::<B, IdxSize>::update(&mut window, start, end) };

        // get_agg returns a relative index within the window
        RollingAggWindowNoNulls::<B, IdxSize>::get_agg(&window, i)
            .map(|rel_idx| start as IdxSize + rel_idx)
    });

    PrimitiveArray::from_trusted_len_iter(iter)
}

fn rolling_arg_extremum_by_nulls<B: NativeType, P: MinMaxPolicy>(
    by: &[B],
    validity: &Bitmap,
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

        // SAFETY: starts/ends are guaranteed to be within bounds by the caller.
        unsafe { RollingAggWindowNulls::<B, IdxSize>::update(&mut window, start, end) };

        if !RollingAggWindowNulls::<B, IdxSize>::is_valid(&window, min_periods) {
            return None;
        }

        RollingAggWindowNulls::<B, IdxSize>::get_agg(&window, i)
            .map(|rel_idx| start as IdxSize + rel_idx)
    });

    PrimitiveArray::from_trusted_len_iter(iter)
}

// --- Public API ---
//
// Preconditions for all public functions below:
// - `starts` and `ends` must be monotonically non-decreasing (rolling window invariant).
// - All indices in `starts`/`ends` must be within bounds of `by`.
// Violating these corrupts the internal deque state.

pub fn rolling_argmin_by_no_nulls<B: NativeType>(
    by: &[B],
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by_no_nulls::<B, MinPropagateNan>(by, starts, ends, min_periods)
}

pub fn rolling_argmax_by_no_nulls<B: NativeType>(
    by: &[B],
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by_no_nulls::<B, MaxPropagateNan>(by, starts, ends, min_periods)
}

pub fn rolling_argmin_by_nulls<B: NativeType>(
    by: &[B],
    validity: &Bitmap,
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by_nulls::<B, MinPropagateNan>(by, validity, starts, ends, min_periods)
}

pub fn rolling_argmax_by_nulls<B: NativeType>(
    by: &[B],
    validity: &Bitmap,
    starts: &[IdxSize],
    ends: &[IdxSize],
    min_periods: usize,
) -> PrimitiveArray<IdxSize> {
    rolling_arg_extremum_by_nulls::<B, MaxPropagateNan>(by, validity, starts, ends, min_periods)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_argmin_by_no_nulls_basic() {
        // by = [10, 3, 7, 1, 5], window_size=3
        // windows: [0,1) [0,2) [0,3) [1,4) [2,5)
        let by: Vec<i32> = vec![10, 3, 7, 1, 5];
        let starts = vec![0u32, 0, 0, 1, 2];
        let ends = vec![1u32, 2, 3, 4, 5];

        let result = rolling_argmin_by_no_nulls(&by, &starts, &ends, 1);
        let expected: Vec<Option<u32>> = vec![Some(0), Some(1), Some(1), Some(3), Some(3)];
        let actual: Vec<Option<u32>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_rolling_argmax_by_no_nulls_basic() {
        let by: Vec<i32> = vec![10, 3, 7, 1, 5];
        let starts = vec![0u32, 0, 0, 1, 2];
        let ends = vec![1u32, 2, 3, 4, 5];

        let result = rolling_argmax_by_no_nulls(&by, &starts, &ends, 1);
        let expected: Vec<Option<u32>> = vec![Some(0), Some(0), Some(0), Some(2), Some(2)];
        let actual: Vec<Option<u32>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_rolling_argmin_by_min_periods() {
        let by: Vec<i32> = vec![10, 3, 7, 1, 5];
        let starts = vec![0u32, 0, 0, 1, 2];
        let ends = vec![1u32, 2, 3, 4, 5];

        let result = rolling_argmin_by_no_nulls(&by, &starts, &ends, 3);
        let expected: Vec<Option<u32>> = vec![None, None, Some(1), Some(3), Some(3)];
        let actual: Vec<Option<u32>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_rolling_argmin_by_nulls() {
        // by = [10, NULL, 7, 1, 5]
        let by: Vec<i32> = vec![10, 0, 7, 1, 5];
        let validity = Bitmap::from(&[true, false, true, true, true]);
        let starts = vec![0u32, 0, 0, 1, 2];
        let ends = vec![1u32, 2, 3, 4, 5];

        let result = rolling_argmin_by_nulls(&by, &validity, &starts, &ends, 1);
        let actual: Vec<Option<u32>> = result.iter().map(|v| v.copied()).collect();
        // Window [0,1): [10] -> min=10 @ 0
        // Window [0,2): [10, NULL] -> min=10 @ 0
        // Window [0,3): [10, NULL, 7] -> min=7 @ 2
        // Window [1,4): [NULL, 7, 1] -> min=1 @ 3
        // Window [2,5): [7, 1, 5] -> min=1 @ 3
        assert_eq!(actual, vec![Some(0), Some(0), Some(2), Some(3), Some(3)]);
    }

    #[test]
    fn test_rolling_argmin_by_empty() {
        let by: Vec<i32> = vec![];
        let starts: Vec<u32> = vec![];
        let ends: Vec<u32> = vec![];

        let result = rolling_argmin_by_no_nulls(&by, &starts, &ends, 1);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_rolling_argmin_by_empty_windows() {
        let by: Vec<i32> = vec![10, 3, 7];
        let starts = vec![0u32, 1, 1];
        let ends = vec![0u32, 1, 1]; // empty windows

        let result = rolling_argmin_by_no_nulls(&by, &starts, &ends, 1);
        let actual: Vec<Option<u32>> = result.iter().map(|v| v.copied()).collect();
        assert_eq!(actual, vec![None, None, None]);
    }

    #[test]
    fn test_rolling_argmin_by_float_nan() {
        // NaN should propagate with MinPropagateNan policy
        let by: Vec<f64> = vec![1.0, f64::NAN, 3.0];
        let starts = vec![0u32, 0, 1];
        let ends = vec![1u32, 2, 3];

        let result = rolling_argmin_by_no_nulls(&by, &starts, &ends, 1);
        let actual: Vec<Option<u32>> = result.iter().map(|v| v.copied()).collect();
        // NaN is "better" (propagated) for MinPropagateNan
        assert_eq!(actual[0], Some(0)); // just 1.0
        assert_eq!(actual[1], Some(1)); // NaN wins over 1.0
        assert_eq!(actual[2], Some(1)); // NaN wins over 3.0
    }
}
