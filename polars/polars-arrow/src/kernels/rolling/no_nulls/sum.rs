use no_nulls;
use no_nulls::{rolling_apply_agg_window, RollingAggWindowNoNulls};

use super::*;

pub struct SumWindow<'a, T> {
    slice: &'a [T],
    sum: T,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + std::iter::Sum + AddAssign + SubAssign>
    RollingAggWindowNoNulls<'a, T> for SumWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let sum = slice[start..end].iter().copied().sum::<T>();
        Self {
            slice,
            sum,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // if we exceed the end, we have a completely new window
        // so we recompute
        let recompute_sum = if start >= self.last_end {
            true
        } else {
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

                self.sum -= *leaving_value;
            }
            recompute_sum
        };
        self.last_start = start;

        // we traverse all values and compute
        if recompute_sum {
            self.sum = self
                .slice
                .get_unchecked(start..end)
                .iter()
                .copied()
                .sum::<T>();
        }
        // remove leaving values.
        else {
            for idx in self.last_end..end {
                self.sum += *self.slice.get_unchecked(idx);
            }
        }
        self.last_end = end;
        self.sum
    }
}

pub fn rolling_sum<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + NumCast + Mul<Output = T> + AddAssign + SubAssign + IsFloat,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<SumWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<SumWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
        ),
        (true, Some(weights)) => {
            let weights = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                no_nulls::compute_sum_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            let weights = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                no_nulls::compute_sum_weights,
                &weights,
            )
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rolling_sum() {
        let values = &[1.0f64, 2.0, 3.0, 4.0];

        let out = rolling_sum(values, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(6.0), Some(10.0)]);

        let out = rolling_sum(values, 4, 1, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(3.0), Some(6.0), Some(10.0), Some(9.0)]);

        let out = rolling_sum(values, 4, 4, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, Some(10.0), None]);

        // test nan handling.
        let values = &[1.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_sum(values, 3, 3, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();

        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!(
                "{:?}",
                &[
                    None,
                    None,
                    Some(6.0),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(18.0)
                ]
            )
        );
    }
}
