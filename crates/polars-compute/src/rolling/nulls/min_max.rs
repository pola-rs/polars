use polars_utils::min_max::{MaxPropagateNan, MinPropagateNan};

use super::super::min_max::MinMaxWindow;

pub type MinWindow<'a, T> = MinMaxWindow<'a, T, MinPropagateNan>;
pub type MaxWindow<'a, T> = MinMaxWindow<'a, T, MaxPropagateNan>;

use super::*;

pub fn rolling_min<T>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> Box<dyn PlArray>
where
    T: NativeType + IsFloat,
{
    // The window machines walk the values as a slice and read the mask bit by bit, so the chunk
    // is laid out here, once at the top, and only a buffer that repeats is written out.
    let arr = arr.to_flat();

    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<MinMaxWindow<T, MinPropagateNan>, _, _, _>(
            arr.as_slice(),
            arr.validity().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<MinMaxWindow<T, MinPropagateNan>, _, _, _>(
            arr.as_slice(),
            arr.validity().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}

pub fn rolling_max<T>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> Box<dyn PlArray>
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd + Bounded + IsFloat,
{
    // The window machines walk the values as a slice and read the mask bit by bit, so the chunk
    // is laid out here, once at the top, and only a buffer that repeats is written out.
    let arr = arr.to_flat();

    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<MinMaxWindow<T, MaxPropagateNan>, _, _, _>(
            arr.as_slice(),
            arr.validity().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<MinMaxWindow<T, MaxPropagateNan>, _, _, _>(
            arr.as_slice(),
            arr.validity().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}
