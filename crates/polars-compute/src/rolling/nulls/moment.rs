#![allow(unsafe_op_in_unsafe_fn)]

use num_traits::{FromPrimitive, ToPrimitive};

pub use super::super::moment::*;
use super::*;

pub fn rolling_var<T>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> Box<dyn PlArray>
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    // The window machines walk the values as a slice and read the mask bit by bit, so the chunk
    // is laid out here, once at the top, and only a buffer that repeats is written out.
    let arr = arr.to_flat();

    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, VarianceMoment>, _, _, _>(
        arr.as_slice(),
        arr.validity().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}

pub fn rolling_skew<T>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> Box<dyn PlArray>
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    // The window machines walk the values as a slice and read the mask bit by bit, so the chunk
    // is laid out here, once at the top, and only a buffer that repeats is written out.
    let arr = arr.to_flat();

    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, SkewMoment>, _, _, _>(
        arr.as_slice(),
        arr.validity().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}

pub fn rolling_kurtosis<T>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> Box<dyn PlArray>
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    // The window machines walk the values as a slice and read the mask bit by bit, so the chunk
    // is laid out here, once at the top, and only a buffer that repeats is written out.
    let arr = arr.to_flat();

    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, KurtosisMoment>, _, _, _>(
        arr.as_slice(),
        arr.validity().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}
