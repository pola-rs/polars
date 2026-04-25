use arrow::legacy::error::polars_ensure;
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::super::arg_min_max::ArgMinMaxWindow;
use super::*;

// NOTE: We deliberately drive the rolling loop manually instead of going
// through `rolling_apply_agg_window` + `RollingAggWindowNoNulls`. The trait's
// `new` signature receives a single slice, while `min_by`/`max_by` need both
// the gather-target (`values`) and the comparison key (`by`). A wrapper
// struct akin to `MinMaxWindow` could be added by introducing a distinct
// `Out` type parameter and a side channel for `values`, but that widens the
// trait surface for one consumer. Re-evaluate if the trait grows
// multi-input support later.
fn rolling_min_max_by_impl<V, B, P>(
    values: &[V],
    by: &[B],
    window_size: usize,
    min_periods: usize,
    center: bool,
) -> PolarsResult<ArrayRef>
where
    V: NativeType,
    B: NativeType,
    P: MinMaxPolicy,
{
    let len = by.len();
    polars_ensure!(
        values.len() == len,
        ComputeError: "rolling_min_by/max_by: `values` and `by` must have the same length"
    );

    let offset_fn: fn(Idx, WindowSize, Len) -> (Start, End) = if center {
        det_offsets_center
    } else {
        det_offsets
    };

    let (start, end) = offset_fn(0, window_size, len);
    let mut win = <ArgMinMaxWindow<'_, B, P> as RollingAggWindowNoNulls<B, IdxSize>>::new(
        by,
        start,
        end,
        None,
        Some(window_size),
    );

    let out = (0..len).map(|idx| {
        let (start, end) = offset_fn(idx, window_size, len);
        if end - start < min_periods {
            None
        } else {
            // SAFETY: `start..end` is within `by`'s bounds by construction.
            unsafe { RollingAggWindowNoNulls::<B, IdxSize>::update(&mut win, start, end) };
            let rel = RollingAggWindowNoNulls::<B, IdxSize>::get_agg(&win, idx)? as usize;
            // SAFETY: `start + rel < end <= len == values.len()`.
            unsafe { Some(*values.get_unchecked(start + rel)) }
        }
    });

    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Ok(Box::new(arr))
}

/// Rolling `min_by`: for each window of `by`, return `values` at the index
/// where `by` is minimal.
///
/// On ties, the *earliest* index in the window wins, matching the
/// monotonic-deque semantics shared with `arg_min` / `rolling_min`.
pub fn rolling_min_by<V, B>(
    values: &[V],
    by: &[B],
    window_size: usize,
    min_periods: usize,
    center: bool,
) -> PolarsResult<ArrayRef>
where
    V: NativeType,
    B: NativeType,
{
    rolling_min_max_by_impl::<V, B, MinPropagateNan>(values, by, window_size, min_periods, center)
}

/// Rolling `max_by`: for each window of `by`, return `values` at the index
/// where `by` is maximal.
///
/// On ties, the *earliest* index in the window wins (same convention as
/// `rolling_min_by`).
pub fn rolling_max_by<V, B>(
    values: &[V],
    by: &[B],
    window_size: usize,
    min_periods: usize,
    center: bool,
) -> PolarsResult<ArrayRef>
where
    V: NativeType,
    B: NativeType,
{
    rolling_min_max_by_impl::<V, B, MaxPropagateNan>(values, by, window_size, min_periods, center)
}

#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;

    use super::*;

    fn collect<T: NativeType>(arr: ArrayRef) -> Vec<Option<T>> {
        arr.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap()
            .into_iter()
            .map(|v| v.copied())
            .collect()
    }

    #[test]
    fn rolling_min_by_basic() {
        // values: [10, 20, 30, 40]
        // by:     [3.0, 1.0, 4.0, 2.0]
        // window=2, min_periods=2 (no center)
        //   i=0  window [3.0]              -> not enough -> None
        //   i=1  window [3.0, 1.0]  argmin abs=1 -> values[1] = 20
        //   i=2  window [1.0, 4.0]  argmin abs=1 -> values[1] = 20
        //   i=3  window [4.0, 2.0]  argmin abs=3 -> values[3] = 40
        let values = &[10i32, 20, 30, 40];
        let by = &[3.0f64, 1.0, 4.0, 2.0];

        let out = rolling_min_by(values, by, 2, 2, false).unwrap();
        assert_eq!(collect::<i32>(out), &[None, Some(20), Some(20), Some(40)]);
    }

    #[test]
    fn rolling_max_by_basic() {
        // values: [10, 20, 30, 40]
        // by:     [3.0, 1.0, 4.0, 2.0]
        //   i=1  window [3.0, 1.0]  argmax abs=0 -> values[0] = 10
        //   i=2  window [1.0, 4.0]  argmax abs=2 -> values[2] = 30
        //   i=3  window [4.0, 2.0]  argmax abs=2 -> values[2] = 30
        let values = &[10i32, 20, 30, 40];
        let by = &[3.0f64, 1.0, 4.0, 2.0];

        let out = rolling_max_by(values, by, 2, 2, false).unwrap();
        assert_eq!(collect::<i32>(out), &[None, Some(10), Some(30), Some(30)]);
    }

    #[test]
    fn rolling_min_by_min_periods_one() {
        // window=2, min_periods=1 -- first window also valid
        let values = &[10i32, 20, 30, 40];
        let by = &[3.0f64, 1.0, 4.0, 2.0];

        let out = rolling_min_by(values, by, 2, 1, false).unwrap();
        assert_eq!(
            collect::<i32>(out),
            &[Some(10), Some(20), Some(20), Some(40)],
        );
    }

    #[test]
    fn rolling_min_by_window_three() {
        // values: [10, 20, 30, 40, 50]
        // by:     [3.0, 1.0, 4.0, 2.0, 5.0]
        //   i=2  window [3.0,1.0,4.0]  argmin abs=1 -> 20
        //   i=3  window [1.0,4.0,2.0]  argmin abs=1 -> 20
        //   i=4  window [4.0,2.0,5.0]  argmin abs=3 -> 40
        let values = &[10i32, 20, 30, 40, 50];
        let by = &[3.0f64, 1.0, 4.0, 2.0, 5.0];

        let out = rolling_min_by(values, by, 3, 3, false).unwrap();
        assert_eq!(
            collect::<i32>(out),
            &[None, None, Some(20), Some(20), Some(40)],
        );
    }

    #[test]
    fn rolling_min_by_length_mismatch_errors() {
        let values = &[10i32, 20, 30];
        let by = &[1.0f64, 2.0];
        let err = rolling_min_by(values, by, 2, 1, false);
        assert!(err.is_err(), "expected mismatched lengths to error");
    }

    #[test]
    fn rolling_max_by_with_ties_picks_first() {
        // Tie-breaking: monotonic-deque keeps the earliest index when later
        // equal values arrive (a value is only displaced by a strictly
        // better one).
        let values = &[100i32, 200, 300];
        let by = &[5.0f64, 5.0, 5.0];

        let out = rolling_max_by(values, by, 3, 3, false).unwrap();
        assert_eq!(collect::<i32>(out), &[None, None, Some(100)]);
    }

    #[test]
    fn rolling_min_by_window_larger_than_len() {
        // window_size > len: each effective window is `(0, idx+1)` (clipped).
        // window=5, min_periods=1 -> all positions valid, expanding-min behaviour.
        let values = &[10i32, 20, 30];
        let by = &[3.0f64, 1.0, 2.0];

        let out = rolling_min_by(values, by, 5, 1, false).unwrap();
        // i=0 [3.0] -> 10; i=1 [3.0,1.0] -> 20; i=2 [3.0,1.0,2.0] -> 20
        assert_eq!(collect::<i32>(out), &[Some(10), Some(20), Some(20)]);
    }

    #[test]
    fn rolling_min_by_min_periods_greater_than_window_is_all_none() {
        // min_periods > window_size: no window can ever satisfy min_periods.
        let values = &[10i32, 20, 30, 40];
        let by = &[3.0f64, 1.0, 4.0, 2.0];

        let out = rolling_min_by(values, by, 2, 5, false).unwrap();
        assert_eq!(collect::<i32>(out), &[None, None, None, None]);
    }

    #[test]
    fn rolling_min_by_propagates_nan_in_by() {
        // `MinPropagateNan` policy: a NaN in `by` is treated as the smallest
        // (i.e. it wins argmin). This pins the chosen NaN policy so a future
        // refactor can't silently switch to NaN-ignoring semantics.
        let values = &[10i32, 20, 30];
        let by = &[1.0f64, f64::NAN, 3.0];

        let out = rolling_min_by(values, by, 3, 1, false).unwrap();
        // i=0 [1.0] -> 10
        // i=1 [1.0, NaN] -> NaN wins -> 20
        // i=2 [1.0, NaN, 3.0] -> NaN wins -> 20
        assert_eq!(collect::<i32>(out), &[Some(10), Some(20), Some(20)]);
    }

    #[test]
    fn rolling_max_by_propagates_nan_in_by() {
        // `MaxPropagateNan`: NaN wins argmax as well.
        let values = &[10i32, 20, 30];
        let by = &[1.0f64, f64::NAN, 3.0];

        let out = rolling_max_by(values, by, 3, 1, false).unwrap();
        assert_eq!(collect::<i32>(out), &[Some(10), Some(20), Some(20)]);
    }

    #[test]
    fn rolling_min_by_centered_window() {
        // window=3 centered, min_periods=1
        // det_offsets_center(i, 3, len) -> (i-1, i+2) clipped to [0, len]
        //   i=0  window [3.0,1.0]    argmin abs=1 -> 20
        //   i=1  window [3.0,1.0,4.0] argmin abs=1 -> 20
        //   i=2  window [1.0,4.0,2.0] argmin abs=1 -> 20
        //   i=3  window [4.0,2.0,5.0] argmin abs=3 -> 40
        //   i=4  window [2.0,5.0]    argmin abs=3 -> 40
        let values = &[10i32, 20, 30, 40, 50];
        let by = &[3.0f64, 1.0, 4.0, 2.0, 5.0];

        let out = rolling_min_by(values, by, 3, 1, true).unwrap();
        assert_eq!(
            collect::<i32>(out),
            &[Some(20), Some(20), Some(20), Some(40), Some(40)],
        );
    }
}
