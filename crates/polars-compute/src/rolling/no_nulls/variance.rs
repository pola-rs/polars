use num_traits::{FromPrimitive, ToPrimitive};
use polars_error::polars_ensure;

use super::*;
use crate::var_cov::VarState;

pub struct VarWindow<'a, T> {
    slice: &'a [T],
    var: VarState,
    ddof: u8,
    last_start: usize,
    last_end: usize,
}

impl<T: ToPrimitive + Copy> VarWindow<'_, T> {
    fn compute_var(&mut self, start: usize, end: usize) {
        self.var = VarState::default();
        for value in &self.slice[start..end] {
            let value: f64 = NumCast::from(*value).unwrap();
            self.var.insert_one(value);
        }
    }
}

impl<'a, T: NativeType + IsFloat + Float + ToPrimitive + FromPrimitive>
    RollingAggWindowNoNulls<'a, T> for VarWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize, params: Option<RollingFnParams>) -> Self {
        let mut out = Self {
            slice,
            var: VarState::default(),
            last_start: start,
            last_end: end,
            ddof: match params {
                None => 1,
                Some(pars) => {
                    let RollingFnParams::Var(pars) = pars else {
                        unreachable!("expected Var params");
                    };
                    pars.ddof
                },
            },
        };
        out.compute_var(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let recompute_var = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_var = false;
            for idx in self.last_start..start {
                // SAFETY: we are in bounds
                let leaving_value = *self.slice.get_unchecked(idx);

                // if the leaving value is nan we need to recompute the window
                if T::is_float() && !leaving_value.is_finite() {
                    recompute_var = true;
                    break;
                }
                let leaving_value: f64 = NumCast::from(leaving_value).unwrap();
                self.var.remove_one(leaving_value);
            }
            recompute_var
        };

        self.last_start = start;

        // we traverse all values and compute
        if recompute_var {
            self.compute_var(start, end);
        } else {
            for idx in self.last_end..end {
                let entering_value = *self.slice.get_unchecked(idx);
                let entering_value: f64 = NumCast::from(entering_value).unwrap();

                self.var.insert_one(entering_value);
            }
        }
        self.last_end = end;
        self.var
            .finalize(self.ddof)
            .map(|v| T::from_f64(v).unwrap())
    }
}

pub fn rolling_var<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + IsFloat + ToPrimitive + FromPrimitive + AddAssign,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    match weights {
        None => rolling_apply_agg_window::<VarWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        Some(weights) => {
            // Validate and standardize the weights like we do for the mean. This definition is fine
            // because frequency weights and unbiasing don't make sense for rolling operations.
            let mut wts = no_nulls::coerce_weights(weights);
            let wsum = wts.iter().fold(T::zero(), |acc, x| acc + *x);
            polars_ensure!(
                wsum != T::zero(),
                ComputeError: "Weighted variance is undefined if weights sum to 0"
            );
            wts.iter_mut().for_each(|w| *w = *w / wsum);
            super::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                offset_fn,
                compute_var_weights,
                &wts,
            )
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_var() {
        let values = &[1.0f64, 5.0, 3.0, 4.0];

        let out = rolling_var(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(8.0), Some(2.0), Some(0.5)]);

        let testpars = Some(RollingFnParams::Var(RollingVarParams { ddof: 0 }));
        let out = rolling_var(values, 2, 2, false, None, testpars).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(4.0), Some(1.0), Some(0.25)]);

        let out = rolling_var(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        // we cannot compare nans, so we compare the string values
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!("{:?}", &[None, Some(8.0), Some(2.0), Some(0.5)])
        );
        // test nan handling.
        let values = &[-10.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_var(values, 3, 3, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        // we cannot compare nans, so we compare the string values
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!(
                "{:?}",
                &[
                    None,
                    None,
                    Some(52.33333333333333),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(1.0)
                ]
            )
        );
    }
}
