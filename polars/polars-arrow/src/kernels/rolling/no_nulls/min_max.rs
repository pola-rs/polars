use super::*;
use no_nulls;
use no_nulls::{rolling_apply_agg_window, RollingAggWindowNoNulls};

pub struct MinWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    slice: &'a [T],
    min: T,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNoNulls<'a, T> for MinWindow<'a, T> {
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let min = *slice[start..end]
            .iter()
            .min_by(|a, b| compare_fn_nan_min(*a, *b))
            .unwrap_or(&slice[start]);
        Self {
            slice,
            min,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // recompute min
        if start >= self.last_end {
            self.min = *self
                .slice
                .get_unchecked(start..end)
                .iter()
                .min_by(|a, b| compare_fn_nan_min(*a, *b))
                .unwrap_or(&self.slice[start]);

            self.last_start = start;
            self.last_end = end;

            return self.min;
        }

        let mut recompute_min = false;
        // remove elements that should leave the window
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let leaving_value = self.slice.get_unchecked(idx);

            // if the leaving value is the
            // max value, we need to recompute the max.
            if matches!(
                compare_fn_nan_min(leaving_value, &self.min),
                Ordering::Equal
            ) {
                recompute_min = true;
                break;
            }
        }

        let entering_min = self
            .slice
            .get_unchecked(self.last_end..end)
            .iter()
            .min_by(|a, b| compare_fn_nan_min(*a, *b))
            .unwrap_or(
                &self.slice[std::cmp::max(self.last_start, self.last_end.saturating_sub(1))],
            );

        if recompute_min {
            match compare_fn_nan_min(&self.min, entering_min) {
                // do nothing
                Ordering::Equal => {}
                // leaving < entering
                Ordering::Less => {
                    // leaving value could be the smallest, we might need to recompute

                    // just a random value in the window to prevent O(n^2) behavior
                    // that can occur when all values in the window are the same
                    let remaining_value1 = self.slice.get(start).unwrap_unchecked();
                    let remaining_value2 = self.slice.get(end.saturating_sub(1)).unwrap();

                    // we check those two value in the window, if they are equal to leaving, we know
                    // we don't need to traverse all to compote the window
                    if !matches!(
                        compare_fn_nan_min(remaining_value1, &self.min),
                        Ordering::Equal
                    ) && !matches!(
                        compare_fn_nan_min(remaining_value2, &self.min),
                        Ordering::Equal
                    ) {
                        // the minimum value int the window we did not yet compute
                        let min_in_between = self
                            .slice
                            .get_unchecked(start..self.last_end)
                            .iter()
                            .min_by(|a, b| compare_fn_nan_min(*a, *b))
                            .unwrap_or(&self.slice[start]);

                        if matches!(
                            compare_fn_nan_min(min_in_between, entering_min),
                            Ordering::Less
                        ) {
                            self.min = *min_in_between
                        } else {
                            self.min = *entering_min
                        }
                    }
                }
                // leaving > entering
                Ordering::Greater => {
                    if matches!(compare_fn_nan_min(entering_min, &self.min), Ordering::Less) {
                        self.min = *entering_min
                    }
                }
            }
        } else if matches!(compare_fn_nan_min(entering_min, &self.min), Ordering::Less) {
            self.min = *entering_min
        }

        self.last_start = start;
        self.last_end = end;
        self.min
    }
}

pub struct MaxWindow<'a, T: NativeType> {
    slice: &'a [T],
    max: T,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNoNulls<'a, T> for MaxWindow<'a, T> {
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let max = *slice[start..end]
            .iter()
            .max_by(|a, b| compare_fn_nan_max(*a, *b))
            .unwrap_or(&slice[start]);
        Self {
            slice,
            max,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // recompute max
        if start >= self.last_end {
            self.max = *self
                .slice
                .get_unchecked(start..end)
                .iter()
                .max_by(|a, b| compare_fn_nan_max(*a, *b))
                .unwrap_or(&self.slice[start]);

            self.last_start = start;
            self.last_end = end;

            return self.max;
        }

        let mut recompute_max = false;
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let leaving_value = self.slice.get_unchecked(idx);
            // if the leaving value is the max value, we need to recompute the max.
            if matches!(
                compare_fn_nan_max(leaving_value, &self.max),
                Ordering::Equal
            ) {
                recompute_max = true;
                break;
            }
        }

        let entering_max = self
            .slice
            .get_unchecked(self.last_end..end)
            .iter()
            .max_by(|a, b| compare_fn_nan_max(*a, *b))
            .unwrap_or(
                &self.slice[std::cmp::max(self.last_start, self.last_end.saturating_sub(1))],
            );

        if recompute_max {
            match compare_fn_nan_max(&self.max, entering_max) {
                // do nothing
                Ordering::Equal => {}
                // leaving < entering
                Ordering::Less => {
                    if matches!(
                        compare_fn_nan_max(entering_max, &self.max),
                        Ordering::Greater
                    ) {
                        self.max = *entering_max
                    }
                }
                // leaving > entering
                Ordering::Greater => {
                    // leaving value could be the largest, we might need to recompute

                    // just a random value in the window to prevent O(n^2) behavior
                    // that can occur when all values in the window are the same
                    let remaining_value1 = self.slice.get(start).unwrap_unchecked();
                    let remaining_value2 = self.slice.get(end.saturating_sub(1)).unwrap();

                    // we check those two value in the window, if they are equal to leaving, we know
                    // we don't need to traverse all to compote the window
                    if !matches!(
                        compare_fn_nan_max(remaining_value1, &self.max),
                        Ordering::Equal
                    ) && !matches!(
                        compare_fn_nan_max(remaining_value2, &self.max),
                        Ordering::Equal
                    ) {
                        // the maximum value int the window we did not yet compute
                        let max_in_between = self
                            .slice
                            .get_unchecked(start..self.last_end)
                            .iter()
                            .max_by(|a, b| compare_fn_nan_max(*a, *b))
                            .unwrap_or(&self.slice[start]);

                        if matches!(
                            compare_fn_nan_max(max_in_between, entering_max),
                            Ordering::Greater
                        ) {
                            self.max = *max_in_between
                        } else {
                            self.max = *entering_max
                        }
                    }
                }
            }
        } else if matches!(
            compare_fn_nan_max(entering_max, &self.max),
            Ordering::Greater
        ) {
            self.max = *entering_max
        }
        self.last_start = start;
        self.last_end = end;
        self.max
    }
}

pub(crate) fn compute_min_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: NativeType + PartialOrd + std::ops::Mul<Output = T>,
{
    values
        .iter()
        .zip(weights)
        .map(|(v, w)| *v * *w)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

pub(crate) fn compute_max_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: NativeType + PartialOrd + IsFloat + Bounded + Mul<Output = T>,
{
    let mut max = T::min_value();
    for v in values.iter().zip(weights).map(|(v, w)| *v * *w) {
        if T::is_float() && v.is_nan() {
            return v;
        }
        if v > max {
            max = v
        }
    }

    max
}

pub fn rolling_max<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<MaxWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<MaxWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
        ),
        (true, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_max_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_max_weights,
                &weights,
            )
        }
    }
}

pub fn rolling_min<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + PartialOrd + NumCast + Mul<Output = T> + Bounded + IsFloat,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<MinWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
        ),
        (false, None) => rolling_apply_agg_window::<MinWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
        ),
        (true, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_min_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_min_weights,
                &weights,
            )
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_min_max() {
        let values = &[1.0f64, 5.0, 3.0, 4.0];

        let out = rolling_min(values, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(1.0), Some(3.0), Some(3.0)]);
        let out = rolling_max(values, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(5.0), Some(5.0), Some(4.0)]);

        let out = rolling_min(values, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.0)]);
        let out = rolling_max(values, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(5.0), Some(5.0), Some(4.0)]);

        let out = rolling_max(values, 3, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(5.0), Some(5.0), Some(5.0)]);

        // test nan handling.
        let values = &[1.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_min(values, 3, 3, false, None);
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
                    Some(1.0),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(5.0)
                ]
            )
        );

        let out = rolling_max(values, 3, 3, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!(
                "{:?}",
                &[
                    None,
                    None,
                    Some(3.0),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(7.0)
                ]
            )
        );
    }
}
