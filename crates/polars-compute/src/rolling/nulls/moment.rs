#![allow(unsafe_op_in_unsafe_fn)]

use num_traits::{FromPrimitive, ToPrimitive};

pub use super::super::moment::*;
use super::*;

pub fn rolling_var<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, VarianceMoment>, _, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}

pub fn rolling_skew<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, SkewMoment>, _, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}

pub fn rolling_kurtosis<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, KurtosisMoment>, _, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}
