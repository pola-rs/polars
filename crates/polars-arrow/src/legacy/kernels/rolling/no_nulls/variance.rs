use polars_error::polars_ensure;

use super::*;

pub(super) struct SumSquaredWindow<'a, T> {
    slice: &'a [T],
    sum_of_squares: T,
    last_start: usize,
    last_end: usize,
    // if we don't recompute every 'n' iterations
    // we get a accumulated error/drift
    last_recompute: u8,
}

impl<'a, T: NativeType + IsFloat + std::iter::Sum + AddAssign + SubAssign + Mul<Output = T>>
    RollingAggWindowNoNulls<'a, T> for SumSquaredWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize, _params: DynArgs) -> Self {
        let sum = slice[start..end].iter().map(|v| *v * *v).sum::<T>();
        Self {
            slice,
            sum_of_squares: sum,
            last_start: start,
            last_end: end,
            last_recompute: 0,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        // if we exceed the end, we have a completely new window
        // so we recompute
        let recompute_sum = if start >= self.last_end || self.last_recompute > 128 {
            self.last_recompute = 0;
            true
        } else {
            self.last_recompute += 1;
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let leaving_value = self.slice.get_unchecked(idx);

                if T::is_float() && !leaving_value.is_finite() {
                    recompute_sum = true;
                    break;
                }

                self.sum_of_squares -= *leaving_value * *leaving_value;
            }
            recompute_sum
        };

        self.last_start = start;

        // we traverse all values and compute
        if T::is_float() && recompute_sum {
            self.sum_of_squares = self
                .slice
                .get_unchecked(start..end)
                .iter()
                .map(|v| *v * *v)
                .sum::<T>();
        } else {
            for idx in self.last_end..end {
                let entering_value = *self.slice.get_unchecked(idx);
                self.sum_of_squares += entering_value * entering_value;
            }
        }
        self.last_end = end;
        Some(self.sum_of_squares)
    }
}

// E[(xi - E[x])^2]
// can be expanded to
// E[x^2] - E[x]^2
pub struct VarWindow<'a, T> {
    mean: MeanWindow<'a, T>,
    sum_of_squares: SumSquaredWindow<'a, T>,
    ddof: u8,
}

impl<
        'a,
        T: NativeType
            + IsFloat
            + Float
            + std::iter::Sum
            + AddAssign
            + SubAssign
            + Div<Output = T>
            + NumCast
            + One
            + Zero
            + PartialOrd
            + Sub<Output = T>,
    > RollingAggWindowNoNulls<'a, T> for VarWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize, params: DynArgs) -> Self {
        Self {
            mean: MeanWindow::new(slice, start, end, None),
            sum_of_squares: SumSquaredWindow::new(slice, start, end, None),
            ddof: match params {
                None => 1,
                Some(pars) => pars.downcast_ref::<RollingVarParams>().unwrap().ddof,
            },
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let count: T = NumCast::from(end - start).unwrap();
        let sum_of_squares = self.sum_of_squares.update(start, end).unwrap_unchecked();
        let mean = self.mean.update(start, end).unwrap_unchecked();

        let denom = count - NumCast::from(self.ddof).unwrap();
        if denom <= T::zero() {
            None
        } else if end - start == 1 {
            Some(T::zero())
        } else {
            let out = (sum_of_squares - count * mean * mean) / denom;
            // variance cannot be negative.
            // if it is negative it is due to numeric instability
            if out < T::zero() {
                Some(T::zero())
            } else {
                Some(out)
            }
        }
    }
}

pub fn rolling_var<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType
        + Float
        + IsFloat
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + One
        + Zero
        + Sub<Output = T>,
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

        let testpars = Some(Arc::new(RollingVarParams { ddof: 0 }) as Arc<dyn Any + Send + Sync>);
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
                    Some(52.333333333333336),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(1.0)
                ]
            )
        );
    }
}
