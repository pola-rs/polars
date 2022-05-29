use super::mean::MeanWindow;
use super::*;
use no_nulls::{rolling_apply_agg_window, RollingAggWindow};

pub(super) struct SumSquaredWindow<'a, T> {
    slice: &'a [T],
    sum_of_squares: T,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + std::iter::Sum + AddAssign + SubAssign + Mul<Output = T>>
    RollingAggWindow<'a, T> for SumSquaredWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let sum = slice[start..end].iter().map(|v| *v * *v).sum::<T>();
        Self {
            slice,
            sum_of_squares: sum,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // remove elements that should leave the window
        let mut recompute_sum = false;
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let leaving_value = self.slice.get_unchecked(idx);

            if T::is_float() && leaving_value.is_nan() {
                recompute_sum = true;
                break;
            }

            self.sum_of_squares -= *leaving_value * *leaving_value;
        }
        self.last_start = start;

        // we traverese all values and compute
        if T::is_float() && recompute_sum {
            self.sum_of_squares = self
                .slice
                .get_unchecked(start..end)
                .iter()
                .map(|v| *v * *v)
                .sum::<T>();
        }
        // the max has not left the window, so we only check
        // if the entering values are larger
        else {
            for idx in self.last_end..end {
                let entering_value = *self.slice.get_unchecked(idx);
                self.sum_of_squares += entering_value * entering_value;
            }
        }
        self.last_end = end;
        self.sum_of_squares
    }
}

// E[(xi - E[x])^2]
// can be expanded to
// E[x^2] - E[x]^2
struct VarWindow<'a, T> {
    mean: MeanWindow<'a, T>,
    sum_of_squares: SumSquaredWindow<'a, T>,
}

impl<
        'a,
        T: NativeType
            + IsFloat
            + std::iter::Sum
            + AddAssign
            + SubAssign
            + Div<Output = T>
            + NumCast
            + One
            + Sub<Output = T>,
    > RollingAggWindow<'a, T> for VarWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        Self {
            mean: MeanWindow::new(slice, start, end),
            sum_of_squares: SumSquaredWindow::new(slice, start, end),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        let count = NumCast::from(end - start).unwrap();
        let sum_of_squares = self.sum_of_squares.update(start, end);
        let mean_of_squares = sum_of_squares / count;
        let mean = self.mean.update(start, end);
        let var = mean_of_squares - mean * mean;
        // apply Bessel's correction
        var / (count - T::one()) * count
    }
}

pub fn rolling_var<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
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
        + Sub<Output = T>,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<VarWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<VarWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
        ),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            super::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_var_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            let weights = coerce_weights(weights);
            super::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_var_weights,
                &weights,
            )
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_var() {
        let values = &[1.0f64, 5.0, 3.0, 4.0];

        let out = rolling_var(values, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(8.0), Some(2.0), Some(0.5)]);

        let out = rolling_var(values, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out
            .into_iter()
            .map(|v| v.copied().unwrap())
            .collect::<Vec<_>>();
        // we cannot compare nans, so we compare the string values
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!("{:?}", &[f64::nan(), 8.0, 2.0, 0.5])
        );
        // test nan handling.
        let values = &[-10.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_var(values, 3, 3, false, None);
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
                    Some(0.9999999999999964)
                ]
            )
        );
    }
}
