#![allow(unsafe_op_in_unsafe_fn)]
use super::*;

pub struct SumWindow<'a, T, S> {
    slice: &'a [T],
    sum: S,
    err: S,
    non_finite_count: usize, // NaN or infinity.
    pos_inf_count: usize,
    neg_inf_count: usize,
    last_start: usize,
    last_end: usize,
}

impl<T, S> SumWindow<'_, T, S>
where
    T: NativeType + IsFloat + Sub<Output = T> + NumCast + PartialOrd,
    S: NativeType + AddAssign + SubAssign + Sub<Output = S> + Add<Output = S> + NumCast,
{
    fn add_finite_kahan(&mut self, val: T) {
        let val: S = NumCast::from(val).unwrap();
        let y = val - self.err;
        let new_sum = self.sum + y;
        self.err = (new_sum - self.sum) - y;
        self.sum = new_sum;
    }

    fn add(&mut self, val: T) {
        if T::is_float() {
            if val.is_finite() {
                self.add_finite_kahan(val);
            } else {
                self.non_finite_count += 1;
                self.pos_inf_count += (val > T::zeroed()) as usize;
                self.neg_inf_count += (val < T::zeroed()) as usize;
            }
        } else {
            let val: S = NumCast::from(val).unwrap();
            self.sum += val;
        }
    }

    fn sub(&mut self, val: T) {
        if T::is_float() {
            if val.is_finite() {
                self.add_finite_kahan(T::zeroed() - val);
            } else {
                self.non_finite_count -= 1;
                self.pos_inf_count -= (val > T::zeroed()) as usize;
                self.neg_inf_count -= (val < T::zeroed()) as usize;
            }
        } else {
            let val: S = NumCast::from(val).unwrap();
            self.sum -= val;
        }
    }
}

impl<'a, T, S> RollingAggWindowNoNulls<'a, T> for SumWindow<'a, T, S>
where
    T: NativeType + IsFloat + Sub<Output = T> + NumCast + PartialOrd,
    S: NativeType + AddAssign + SubAssign + Sub<Output = S> + Add<Output = S> + NumCast,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        _params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self {
        let mut out = Self {
            slice,
            sum: S::zeroed(),
            err: S::zeroed(),
            non_finite_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            last_start: 0,
            last_end: 0,
        };
        unsafe { out.update(start, end) };
        out
    }

    // # Safety
    // The start, end range must be in-bounds.
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        if start >= self.last_end {
            self.sum = S::zeroed();
            self.err = S::zeroed();
            self.non_finite_count = 0;
            self.pos_inf_count = 0;
            self.neg_inf_count = 0;
            self.last_start = start;
            self.last_end = start;
        }

        for val in &self.slice[self.last_start..start] {
            self.sub(*val);
        }

        for val in &self.slice[self.last_end..end] {
            self.add(*val);
        }

        self.last_start = start;
        self.last_end = end;
        if self.non_finite_count == 0 {
            NumCast::from(self.sum)
        } else if self.non_finite_count == self.pos_inf_count {
            Some(T::pos_inf_value())
        } else if self.non_finite_count == self.neg_inf_count {
            Some(T::neg_inf_value())
        } else {
            Some(T::nan_value())
        }
    }
}

pub fn rolling_sum<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType
        + std::iter::Sum
        + NumCast
        + Mul<Output = T>
        + AddAssign
        + SubAssign
        + IsFloat
        + Num
        + PartialOrd,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<SumWindow<T, T>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            None,
        ),
        (false, None) => rolling_apply_agg_window::<SumWindow<T, T>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
            None,
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
                center,
            )
        },
        (false, Some(weights)) => {
            let weights = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                no_nulls::compute_sum_weights,
                &weights,
                center,
            )
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rolling_sum() {
        let values = &[1.0f64, 2.0, 3.0, 4.0];

        let out = rolling_sum(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 4, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(6.0), Some(10.0)]);

        let out = rolling_sum(values, 4, 1, true, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(3.0), Some(6.0), Some(10.0), Some(9.0)]);

        let out = rolling_sum(values, 4, 4, true, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, Some(10.0), None]);

        // test nan handling.
        let values = &[1.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_sum(values, 3, 3, false, None, None).unwrap();
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
