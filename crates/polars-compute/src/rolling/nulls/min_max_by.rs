use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::legacy::error::{PolarsResult, polars_ensure, polars_err};
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::super::arg_min_max::ArgMinMaxWindow;
use super::*;

fn rolling_min_max_by_impl<V, B, P>(
    values: &[V],
    by: &PrimitiveArray<B>,
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

    let by_values = by.values().as_slice();
    let by_validity = by.validity().ok_or_else(|| {
        polars_err!(
            ComputeError: "rolling_min_by/max_by (nulls kernel): `by` must carry a validity bitmap"
        )
    })?;

    let offset_fn: fn(Idx, WindowSize, Len) -> (Start, End) = if center {
        det_offsets_center
    } else {
        det_offsets
    };

    let (start, end) = offset_fn(0, window_size, len);
    let mut win = <ArgMinMaxWindow<'_, B, P> as RollingAggWindowNulls<B, IdxSize>>::new(
        by_values,
        by_validity,
        start,
        end,
        None,
        Some(window_size),
    );

    let mut validity =
        create_validity(min_periods, len, window_size, offset_fn).unwrap_or_else(|| {
            let mut v = MutableBitmap::with_capacity(len);
            v.extend_constant(len, true);
            v
        });

    let out: Vec<V> = (0..len)
        .map(|idx| {
            let (start, end) = offset_fn(idx, window_size, len);
            // SAFETY: `start..end` is within bounds.
            unsafe { RollingAggWindowNulls::<B, IdxSize>::update(&mut win, start, end) };
            match RollingAggWindowNulls::<B, IdxSize>::get_agg(&win, idx) {
                Some(rel) if RollingAggWindowNulls::<B, IdxSize>::is_valid(&win, min_periods) => {
                    let abs = start + rel as usize;
                    // SAFETY: `abs < end <= len == values.len()`.
                    unsafe { *values.get_unchecked(abs) }
                },
                _ => {
                    // SAFETY: `idx < len`.
                    unsafe { validity.set_unchecked(idx, false) };
                    V::default()
                },
            }
        })
        .collect_trusted::<Vec<_>>();

    Ok(Box::new(PrimitiveArray::new(
        V::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    )))
}

/// Rolling `min_by` with null support on the `by` column.
///
/// For each window of `by`, returns `values[idx]` where `idx` is the index
/// of the smallest non-null `by` value. If the window has fewer non-null
/// `by` values than `min_periods`, the output is null.
pub fn rolling_min_by<V, B>(
    values: &[V],
    by: &PrimitiveArray<B>,
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

/// Rolling `max_by` with null support on the `by` column.
///
/// Mirror of [`rolling_min_by`]: for each window of `by`, returns
/// `values[idx]` where `idx` is the index of the largest non-null `by`
/// value. If the window has fewer non-null `by` values than `min_periods`,
/// the output is null.
///
/// On ties, the *earliest* index in the window wins (monotonic-deque
/// semantics, consistent with `rolling_max`).
pub fn rolling_max_by<V, B>(
    values: &[V],
    by: &PrimitiveArray<B>,
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
    use arrow::datatypes::ArrowDataType;
    use polars_buffer::Buffer;

    use super::*;

    fn collect<T: NativeType>(arr: ArrayRef) -> Vec<Option<T>> {
        arr.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap()
            .into_iter()
            .map(|v| v.copied())
            .collect()
    }

    fn by_with_nulls() -> PrimitiveArray<f64> {
        // by:        [3.0, NULL, 4.0, 2.0, 5.0]
        // valid mask:[true, false, true, true, true]
        let buf = Buffer::from(vec![3.0, 0.0, 4.0, 2.0, 5.0]);
        PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true, true])),
        )
    }

    #[test]
    fn rolling_min_by_skips_nulls_in_by() {
        // values: [10, 20, 30, 40, 50]
        // by:     [3.0, NULL, 4.0, 2.0, 5.0]
        // window=3, min_periods=2 (no center)
        //   i=0  window {3.0}                 1 valid < 2  -> None
        //   i=1  window {3.0,NULL}            1 valid < 2  -> None
        //   i=2  window {3.0,NULL,4.0}        argmin abs=0 -> values[0] = 10
        //   i=3  window {NULL,4.0,2.0}        argmin abs=3 -> values[3] = 40
        //   i=4  window {4.0,2.0,5.0}         argmin abs=3 -> values[3] = 40
        let values = &[10i32, 20, 30, 40, 50];
        let by = by_with_nulls();

        let out = rolling_min_by(values, &by, 3, 2, false).unwrap();
        assert_eq!(
            collect::<i32>(out),
            &[None, None, Some(10), Some(40), Some(40)],
        );
    }

    #[test]
    fn rolling_max_by_skips_nulls_in_by() {
        // by:     [3.0, NULL, 4.0, 2.0, 5.0]
        //   i=2  window {3.0,NULL,4.0}        argmax abs=2 -> values[2] = 30
        //   i=3  window {NULL,4.0,2.0}        argmax abs=2 -> values[2] = 30
        //   i=4  window {4.0,2.0,5.0}         argmax abs=4 -> values[4] = 50
        let values = &[10i32, 20, 30, 40, 50];
        let by = by_with_nulls();

        let out = rolling_max_by(values, &by, 3, 2, false).unwrap();
        assert_eq!(
            collect::<i32>(out),
            &[None, None, Some(30), Some(30), Some(50)],
        );
    }

    #[test]
    fn rolling_min_by_all_nulls_window_is_null() {
        // window where all `by` entries are null -> output null
        let values = &[10i32, 20, 30];
        let buf = Buffer::from(vec![0.0f64, 0.0, 0.0]);
        let by = PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[false, false, true])),
        );

        // window=2, min_periods=1
        //   i=0  {NULL}              0 valid < 1 -> None
        //   i=1  {NULL,NULL}         0 valid < 1 -> None
        //   i=2  {NULL, 0.0}         argmin abs=2 -> values[2] = 30
        let out = rolling_min_by(values, &by, 2, 1, false).unwrap();
        assert_eq!(collect::<i32>(out), &[None, None, Some(30)]);
    }

    #[test]
    fn rolling_min_by_all_valid_matches_no_nulls_kernel() {
        // When validity is all-true, output should match the no_nulls kernel.
        let values = &[10i32, 20, 30, 40];
        let by_vals = vec![3.0f64, 1.0, 4.0, 2.0];
        let by = PrimitiveArray::new(
            ArrowDataType::Float64,
            Buffer::from(by_vals.clone()),
            Some(Bitmap::new_with_value(true, 4)),
        );

        let nulls_out = rolling_min_by(values, &by, 2, 2, false).unwrap();
        let no_nulls_out =
            super::super::no_nulls::rolling_min_by(values, &by_vals, 2, 2, false).unwrap();

        assert_eq!(collect::<i32>(nulls_out), collect::<i32>(no_nulls_out));
    }

    #[test]
    fn rolling_min_by_no_validity_errors() {
        // Caller is expected to attach an explicit validity bitmap. Mirrors
        // the contract of canonical `nulls/rolling_min`, but returns an error
        // instead of panicking.
        let values = &[10i32, 20, 30];
        let buf = Buffer::from(vec![1.0f64, 2.0, 3.0]);
        let by = PrimitiveArray::new(ArrowDataType::Float64, buf, None);

        let res = rolling_min_by(values, &by, 2, 1, false);
        assert!(res.is_err(), "expected missing validity bitmap to error");
    }

    #[test]
    fn rolling_min_by_min_periods_greater_than_window_is_all_none() {
        let values = &[10i32, 20, 30, 40, 50];
        let by = by_with_nulls();
        let out = rolling_min_by(values, &by, 2, 5, false).unwrap();
        assert_eq!(collect::<i32>(out), &[None, None, None, None, None]);
    }

    #[test]
    fn rolling_min_by_propagates_nan_in_valid_by() {
        // NaN with validity=true should still be considered (and win under
        // `MinPropagateNan`). This pins behaviour against accidental
        // NaN-ignore refactors.
        let values = &[10i32, 20, 30];
        let buf = Buffer::from(vec![1.0f64, f64::NAN, 3.0]);
        let by = PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::new_with_value(true, 3)),
        );

        let out = rolling_min_by(values, &by, 3, 1, false).unwrap();
        // i=2 [1.0, NaN, 3.0] -> NaN wins -> values[1] = 20
        let collected = collect::<i32>(out);
        assert_eq!(collected[2], Some(20));
    }

    #[test]
    fn rolling_min_by_alternating_null_pattern() {
        // Sliding non-null bookkeeping check: validity = [F,T,F,T,F,T],
        // window=3, min_periods=2.
        // Per-window non-null counts:
        //   i=0 {F}                 0 < 2 -> None
        //   i=1 {F,T}               1 < 2 -> None
        //   i=2 {F,T,F}             1 < 2 -> None
        //   i=3 {T,F,T}             2 >= 2 -> argmin over by[1]=2.0, by[3]=4.0 -> abs=1 -> values[1]=20
        //   i=4 {F,T,F}             1 < 2 -> None
        //   i=5 {T,F,T}             2 >= 2 -> argmin over by[3]=4.0, by[5]=6.0 -> abs=3 -> values[3]=40
        let values = &[10i32, 20, 30, 40, 50, 60];
        let buf = Buffer::from(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let by = PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[false, true, false, true, false, true])),
        );

        let out = rolling_min_by(values, &by, 3, 2, false).unwrap();
        assert_eq!(
            collect::<i32>(out),
            &[None, None, None, Some(20), None, Some(40)],
        );
    }

    #[test]
    fn rolling_min_by_length_mismatch_errors() {
        let values = &[10i32, 20];
        let by = by_with_nulls();
        let err = rolling_min_by(values, &by, 2, 1, false);
        assert!(err.is_err(), "expected length mismatch to error");
    }
}
