use no_nulls::{rolling_apply_agg_window, RollingAggWindowNoNulls};
use num::pow::Pow;

use super::mean::MeanWindow;
use super::variance::SumPowersWindow;
use super::*;
pub struct KurtosisWindow<'a, T> {
    mean: MeanWindow<'a, T>,
    sum_of_squares: SumPowersWindow<'a, T>,
    sum_of_cubes: SumPowersWindow<'a, T>,
    sum_of_fourths: SumPowersWindow<'a, T>,
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
            + Zero
            + Sub<Output = T>
            + Pow<i8, Output = T>,
    > RollingAggWindowNoNulls<'a, T> for KurtosisWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        // TODO: when window < 4, no kurtosis

        Self {
            mean: MeanWindow::new(slice, start, end),
            sum_of_squares: SumPowersWindow::new(slice, start, end),
            sum_of_cubes: SumPowersWindow::new(slice, start, end),
            sum_of_fourths: SumPowersWindow::new(slice, start, end),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // based on https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        let count: T = NumCast::from(end - start).unwrap();

        // TODO: if count < 3 -> NAN

        let mean = self.mean.update(start, end);
        let sum_of_squares = self.sum_of_squares.update(start, end);
        let sum_of_cubes = self.sum_of_cubes.update(start, end);
        let sum_of_fourths = self.sum_of_fourths.update(start, end);

        // compute moments
        let a = mean / count;
        let mut r = a * a;

        let b = sum_of_squares / count - r;
        // if b <= 1e-14 {
        //     T::zero()
        // }

        r = r * a;

        // QUESTION??? how to multiply below?
        let one: T = NumCast::from(1).unwrap();
        let two: T = NumCast::from(2).unwrap();
        let three: T = NumCast::from(3).unwrap();
        let four: T = NumCast::from(4).unwrap();
        let six: T = NumCast::from(6).unwrap();

        let c = sum_of_cubes / count - r - three * a * b;
        r = r * a;

        let d = sum_of_fourths / count - r - six * b * a * a - (four * c * a);

        if end - start == 1 {
            T::zero()
        } else {
            let k = ((count * count) - one) * d / (b * b) - three * ((count - one) * (count - one));
            // TODO if self.fisher , subtract 3
            k / ((count - two) * (count - three))
        }
    }
}

pub fn rolling_kurtosis<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
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
        + Zero
        + Sub<Output = T>
        + Pow<i8, Output = T>,
{
    match (weights) {
        (None) => rolling_apply_agg_window::<KurtosisWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (Some(weights)) => rolling_apply_agg_window::<KurtosisWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
        ),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_kurtosis() {
        let values = &[1.0f64, 5.0, 3.0, 4.0];

        let out = rolling_kurtosis(values, 2, 2, None);
    }
}
