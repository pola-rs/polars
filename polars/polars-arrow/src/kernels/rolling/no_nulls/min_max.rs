use super::*;
use no_nulls;
use no_nulls::{rolling_apply_agg_window, RollingAggWindow};

struct MinWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    slice: &'a [T],
    min: T,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindow<'a, T> for MinWindow<'a, T> {
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let min = *slice[start..end]
            .iter()
            .min_by(|a, b| compare_fn_nan_min(*a, *b))
            .unwrap();
        Self {
            slice,
            min,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // remove elements that should leave the window
        let mut recompute_min = false;
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
        self.last_start = start;

        // we traverse all values and compute
        if recompute_min {
            self.min = *self
                .slice
                .get_unchecked(start..end)
                .iter()
                .min_by(|a, b| compare_fn_nan_min(*a, *b))
                .unwrap();
        }
        // the max has not left the window, so we only check
        // if the entering values are larger
        else if end > self.last_end {
            let min_entering = self
                .slice
                .get_unchecked(self.last_end..end)
                .iter()
                .min_by(|a, b| compare_fn_nan_min(*a, *b))
                .unwrap_unchecked();
            if matches!(compare_fn_nan_min(min_entering, &self.min), Ordering::Less) {
                self.min = *min_entering
            }
        }
        self.last_end = end;
        self.min
    }
}

struct MaxWindow<'a, T: NativeType> {
    slice: &'a [T],
    max: T,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindow<'a, T> for MaxWindow<'a, T> {
    fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let max = *slice[start..end]
            .iter()
            .max_by(|a, b| compare_fn_nan_max(*a, *b))
            .unwrap();
        Self {
            slice,
            max,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        // remove elements that should leave the window
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
        self.last_start = start;

        // we traverese all values and compute
        if recompute_max {
            self.max = *self
                .slice
                .get_unchecked(start..end)
                .iter()
                .max_by(|a, b| compare_fn_nan_max(*a, *b))
                .unwrap_unchecked();
        }
        // the max has not left the window, so we only check
        // if the entering values are larger
        else if end > self.last_end {
            let max_entering = self
                .slice
                .get_unchecked(self.last_end..end)
                .iter()
                .max_by(|a, b| compare_fn_nan_max(*a, *b))
                .unwrap_unchecked();
            if matches!(
                compare_fn_nan_max(max_entering, &self.max),
                Ordering::Greater
            ) {
                self.max = *max_entering
            }
        }
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
