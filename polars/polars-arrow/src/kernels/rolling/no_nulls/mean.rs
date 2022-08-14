use no_nulls::{rolling_apply_agg_window, RollingAggWindowNoNulls};

use super::sum::SumWindow;
use super::*;

pub struct MeanWindow<'a, T> {
    sum: SumWindow<'a, T>,
}

impl<
        'a,
        T: NativeType + IsFloat + std::iter::Sum + AddAssign + SubAssign + Div<Output = T> + NumCast,
    > RollingAggWindowNoNulls<'a, T> for MeanWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        Self {
            sum: SumWindow::new(slice, start, end),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        let sum = self.sum.update(start, end);
        sum / NumCast::from(end - start).unwrap()
    }
}

pub fn rolling_mean<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<MeanWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<MeanWindow<_>, _, _>(
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
                no_nulls::compute_mean_weights,
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
                no_nulls::compute_mean_weights,
                &weights,
            )
        }
    }
}
