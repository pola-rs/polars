#![allow(unsafe_op_in_unsafe_fn)]
use super::super::mean::MeanWindow;
use super::*;

pub fn rolling_mean<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_samples: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType
        + IsFloat
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + NumCast
        + AddAssign
        + SubAssign
        + Div<Output = T>,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<MeanWindow<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_samples,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<MeanWindow<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_samples,
            det_offsets,
            None,
        )
    }
}

pub fn rolling_mean_exact<const FLOOR: bool, const SCALE: i64>(
    arr: &PrimitiveArray<i64>,
    window_size: usize,
    min_samples: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> ArrayRef {
    assert!(weights.is_none());
    let offset_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<super::super::ExactMeanWindow<'_, FLOOR, SCALE>, _, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_samples,
        offset_fn,
        None,
    )
}
