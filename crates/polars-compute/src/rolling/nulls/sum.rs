#![allow(unsafe_op_in_unsafe_fn)]
use super::super::sum::SumWindow;
use super::*;

pub fn rolling_sum<T>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> Box<dyn PlArray>
where
    T: NativeType
        + IsFloat
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + SubAssign
        + AddAssign
        + NumCast,
{
    // The window machines walk the values as a slice and read the mask bit by bit, so the chunk
    // is laid out here, once at the top, and only a buffer that repeats is written out.
    let arr = arr.to_flat();

    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<SumWindow<T, T>, _, _, _>(
            arr.as_slice(),
            arr.validity().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<SumWindow<T, T>, _, _, _>(
            arr.as_slice(),
            arr.validity().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}
