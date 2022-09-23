use core::panic;

use no_nulls::{rolling_apply_agg_window, RollingAggWindowNoNulls};
use num::pow::Pow;

use super::variance::SumPowersWindow;
use super::*;
pub struct KurtosisWindow<'a, T> {
    sum: SumPowersWindow<'a, T>,
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
            sum: SumPowersWindow::new_with_base(slice, 1, start, end),
            sum_of_squares: SumPowersWindow::new_with_base(slice, 2, start, end),
            sum_of_cubes: SumPowersWindow::new_with_base(slice, 3, start, end),
            sum_of_fourths: SumPowersWindow::new_with_base(slice, 4, start, end),
        }
    }

    fn new_with_base(slice: &'a [T], _base: i8, start: usize, end: usize) -> Self {
        Self::new(slice, start, end)
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // based on
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        // https://github.com/pandas-dev/pandas/blob/3937fbe1106c5b30b9072243920a8521d36dc301/pandas/core/nanops.py#L1250
        // https://github.com/ajcr/rolling/commit/591e2eb91988d75c74f5704eecf7f12f1a0b63e8
        let count: T = NumCast::from(end - start).unwrap();

        // TODO: if count < 3 -> NAN
        if count.to_usize().unwrap() < 3 {
            return Default::default();
        }

        let sum = self.sum.update(start, end);
        let sum_of_squares = self.sum_of_squares.update(start, end);
        let sum_of_cubes = self.sum_of_cubes.update(start, end);
        let sum_of_fourths = self.sum_of_fourths.update(start, end);

        // compute moments
        let a = sum / count;
        let mut r = a * a;

        let b = sum_of_squares / count - r;
        if b.to_f64().unwrap() <= 1e-14 {
            Default::default()
        }

        r = r * a;

        // QUESTION??? how to multiply below?
        let one = T::one();
        let two = one + one;
        let three = two + one;
        let four = three + one;
        let six = four + one + one;

        let c = ((sum_of_cubes / count) - r) - (three * a * b);
        r = r * a;

        let d = ((sum_of_fourths / count) - r) - (six * b * a * a) - (four * c * a);

        if end - start == 1 {
            T::zero()
        } else {
            let k =
                ((((count * count) - one) * d) / (b * b)) - three * ((count - one) * (count - one));

            k / ((count - two) * (count - three))
            // out
            // if fisher {
            //     out - three
            // } else {
            //     out
            // }
        }
    }
}

pub fn rolling_kurtosis<T>(
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
        + Zero
        + Sub<Output = T>
        + Pow<i8, Output = T>,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<KurtosisWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<KurtosisWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
        ),
        _ => panic!("not yet implemented"),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_kurtosis() {
        let values = &[3.0f64, 1., 4., 1., 5., 9., 2.];

        let out = rolling_kurtosis(values, 4, 4, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();

        assert_eq!(
            out,
            &[
                None,
                None,
                None,
                Some(-3.901234567901234),
                Some(-4.858131487889274),
                Some(1.1656663364605784),
                Some(-0.5818938605619142)
            ]
        )
    }
}
