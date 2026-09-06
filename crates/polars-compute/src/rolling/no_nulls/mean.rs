#![allow(unsafe_op_in_unsafe_fn)]

use super::super::mean::MeanWindow;
use super::*;

pub fn rolling_mean<T>(
    values: &NoNulls<PlPrimitiveArray<T>>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> PolarsResult<Box<dyn PlArray>>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    // The window machines walk their values as a slice, and this is where the chunk becomes
    // one: the representation is resolved once, out of the loop, and a buffer that already holds
    // one slot per element is handed over as it stands. No element is null here, so the mask is
    // not read at all, whatever representation it is in.
    let values = values.to_flat_values();
    let values = values.as_slice();

    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    match weights {
        None => rolling_apply_agg_window::<MeanWindow<_>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            None,
        ),
        Some(weights) => {
            let wts = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                offset_fn,
                no_nulls::compute_mean_weights,
                &wts,
                center,
            )
        },
    }
}
