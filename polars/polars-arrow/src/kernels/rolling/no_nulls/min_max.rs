use no_nulls;
use no_nulls::{rolling_apply_agg_window, RollingAggWindowNoNulls};

use super::*;

pub struct SortedMinMax<'a, T: NativeType> {
    slice: &'a [T],
}

impl<'a, T: NativeType> RollingAggWindowNoNulls<'a, T> for SortedMinMax<'a, T> {
    fn new(slice: &'a [T], _start: usize, _end: usize, _params: DynArgs) -> Self {
        Self { slice }
    }

    #[inline]
    unsafe fn update(&mut self, start: usize, _end: usize) -> T {
        *self.slice.get_unchecked(start)
    }
}

#[inline]
unsafe fn get_min_and_idx<T>(slice: &[T], start: usize, end: usize) -> Option<(usize, &T)>
where
    T: NativeType + IsFloat + PartialOrd,
{
    // Reversed because min_by returns the first min if there's a tie but we want the last
    slice
        .get_unchecked(start..end)
        .iter()
        .enumerate()
        .rev()
        .min_by(|&a, &b| compare_fn_nan_min(a.1, b.1))
}

pub struct MinWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    slice: &'a [T],
    min: T,
    min_idx: usize,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNoNulls<'a, T> for MinWindow<'a, T> {
    fn new(slice: &'a [T], start: usize, end: usize, _params: DynArgs) -> Self {
        let (idx, min) =
            unsafe { get_min_and_idx(slice, start, end).unwrap_or((0, &slice[start])) };
        Self {
            slice,
            min: *min,
            min_idx: start + idx,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        //For details see: https://github.com/pola-rs/polars/pull/9277#issuecomment-1581401692
        self.last_start = start; // Don't care where the last one started
        let old_last_end = self.last_end; // But we need this
        self.last_end = end;

        let entering_start = std::cmp::max(old_last_end, start);
        let entering = get_min_and_idx(self.slice, entering_start, end);
        let empty_overlap = old_last_end <= start;

        if entering.is_some_and(|em| compare_fn_nan_min(&self.min, em.1).is_ge() || empty_overlap) {
            // If the entering min <= the current min return early, since no value in the overlap can be smaller than either.
            self.min = *entering.unwrap().1;
            self.min_idx = entering_start + entering.unwrap().0;
            return self.min;
        } else if self.min_idx >= start || empty_overlap {
            // If the entering min isn't the smallest but the current min is between start and end we can still ignore the overlap
            return self.min;
        }
        // Otherwise get the min of the overlapping window and the entering min
        match (get_min_and_idx(self.slice, start, old_last_end), entering) {
            (Some(pm), Some(em)) => {
                if compare_fn_nan_min(pm.1, em.1).is_ge() {
                    self.min = *em.1;
                    self.min_idx = entering_start + em.0;
                } else {
                    self.min = *pm.1;
                    self.min_idx = start + pm.0;
                }
            }
            (Some(pm), None) => {
                self.min = *pm.1;
                self.min_idx = start + pm.0;
            }
            (None, Some(em)) => {
                self.min = *em.1;
                self.min_idx = entering_start + em.0;
            }
            // We shouldn't reach this, but it means
            (None, None) => {}
        }

        self.min
    }
}

#[inline]
unsafe fn get_max_and_idx<T>(slice: &[T], start: usize, end: usize) -> Option<(usize, &T)>
where
    T: NativeType + IsFloat + PartialOrd,
{
    slice
        .get_unchecked(start..end)
        .iter()
        .enumerate()
        .max_by(|&a, &b| compare_fn_nan_max(a.1, b.1))
}

pub struct MaxWindow<'a, T: NativeType> {
    slice: &'a [T],
    max: T,
    max_idx: usize,
    last_start: usize,
    last_end: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNoNulls<'a, T> for MaxWindow<'a, T> {
    fn new(slice: &'a [T], start: usize, end: usize, _params: DynArgs) -> Self {
        let (idx, max) =
            unsafe { get_max_and_idx(slice, start, end).unwrap_or((0, &slice[start])) };
        Self {
            slice,
            max: *max,
            max_idx: start + idx,
            last_start: start,
            last_end: end,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        self.last_start = start; // Don't care where the last one started
        let old_last_end = self.last_end; // But we need this
        self.last_end = end;

        let entering_start = std::cmp::max(old_last_end, start);
        let entering = get_max_and_idx(self.slice, entering_start, end);
        let empty_overlap = old_last_end < start;

        if entering.is_some_and(|em| compare_fn_nan_max(&self.max, em.1).is_le() || empty_overlap) {
            // If the entering max >= the current max return early, since no value in the overlap can be larger than either.
            self.max = *entering.unwrap().1;
            self.max_idx = entering_start + entering.unwrap().0;
            return self.max;
        } else if self.max_idx >= start || empty_overlap {
            // If the entering max isn't the largest but the current max is between start and end we can still ignore the overlap
            return self.max;
        }
        // Otherwise get the max of the overlapping window and the entering max
        match (get_max_and_idx(self.slice, start, old_last_end), entering) {
            (Some(pm), Some(em)) => {
                if compare_fn_nan_max(pm.1, em.1).is_le() {
                    self.max = *em.1;
                    self.max_idx = entering_start + em.0;
                } else {
                    self.max = *pm.1;
                    self.max_idx = start + pm.0;
                }
            }
            (Some(pm), None) => {
                self.max = *pm.1;
                self.max_idx = start + pm.0;
            }
            (None, Some(em)) => {
                self.max = *em.1;
                self.max_idx = entering_start + em.0;
            }
            // We shouldn't reach this, but it means
            (None, None) => {}
        }

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

pub fn is_reverse_sorted_max<T: NativeType + PartialOrd + IsFloat>(values: &[T]) -> bool {
    values
        .windows(2)
        .all(|w| match compare_fn_nan_min(&w[0], &w[1]) {
            Ordering::Equal => true,
            Ordering::Greater => true,
            Ordering::Less => false,
        })
}

pub fn rolling_max<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    match (center, weights) {
        (true, None) => {
            // will be O(n2) if we don't take this path we hope that we hit an early return on not sorted data
            if is_reverse_sorted_max(values) {
                rolling_apply_agg_window::<SortedMinMax<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets_center,
                    None,
                )
            } else {
                rolling_apply_agg_window::<MaxWindow<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets_center,
                    None,
                )
            }
        }
        (false, None) => {
            if is_reverse_sorted_max(values) {
                rolling_apply_agg_window::<SortedMinMax<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets,
                    None,
                )
            } else {
                rolling_apply_agg_window::<MaxWindow<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets,
                    None,
                )
            }
        }
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

pub fn is_sorted_min<T: NativeType + PartialOrd + IsFloat>(values: &[T]) -> bool {
    values
        .windows(2)
        .all(|w| match compare_fn_nan_min(&w[0], &w[1]) {
            Ordering::Equal => true,
            Ordering::Less => true,
            Ordering::Greater => false,
        })
}

pub fn rolling_min<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + NumCast + Mul<Output = T> + Bounded + IsFloat,
{
    match (center, weights) {
        (true, None) => {
            // will be O(n2) if we don't take this path we hope that we hit an early return on not sorted data
            if is_sorted_min(values) {
                rolling_apply_agg_window::<SortedMinMax<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets_center,
                    None,
                )
            } else {
                rolling_apply_agg_window::<MinWindow<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets_center,
                    None,
                )
            }
        }
        (false, None) => {
            // will be O(n2)
            if is_sorted_min(values) {
                rolling_apply_agg_window::<SortedMinMax<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets,
                    None,
                )
            } else {
                rolling_apply_agg_window::<MinWindow<_>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    det_offsets,
                    None,
                )
            }
        }
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

        let out = rolling_min(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(1.0), Some(3.0), Some(3.0)]);
        let out = rolling_max(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(5.0), Some(5.0), Some(4.0)]);

        let out = rolling_min(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.0)]);
        let out = rolling_max(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(5.0), Some(5.0), Some(4.0)]);

        let out = rolling_max(values, 3, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(5.0), Some(5.0), Some(5.0)]);

        // test nan handling.
        let values = &[1.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_min(values, 3, 3, false, None, None).unwrap();
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

        let out = rolling_max(values, 3, 3, false, None, None).unwrap();
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
