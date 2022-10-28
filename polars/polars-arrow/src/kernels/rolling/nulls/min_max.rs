use arrow::bitmap::utils::{count_zeros, ZipValidityIter};
use nulls;
use nulls::{rolling_apply_agg_window, RollingAggWindowNulls};

use super::*;

pub fn is_reverse_sorted_max_nulls<T: NativeType + PartialOrd + IsFloat>(
    values: &[T],
    validity: &Bitmap,
) -> bool {
    let mut current_max = None;
    for opt_v in ZipValidityIter::new(values.iter(), validity.iter()) {
        match (current_max, opt_v) {
            // do nothing
            (None, None) => {}
            (None, Some(v)) => current_max = Some(*v),
            (Some(current), Some(val)) => {
                match compare_fn_nan_min(&current, val) {
                    Ordering::Greater => {
                        current_max = Some(*val);
                    }
                    // allowed
                    Ordering::Equal => {}
                    // not sorted
                    Ordering::Less => return false,
                }
            }
            (Some(_current), None) => {}
        }
    }

    true
}

pub struct SortedMinMax<'a, T: NativeType> {
    slice: &'a [T],
    validity: &'a Bitmap,
    last_start: usize,
    last_end: usize,
    null_count: usize,
}

impl<'a, T: NativeType> SortedMinMax<'a, T> {
    fn count_nulls(&self, start: usize, end: usize) -> usize {
        let (bytes, offset, _) = self.validity.as_slice();
        count_zeros(bytes, offset + start, end - start)
    }
}

impl<'a, T: NativeType> RollingAggWindowNulls<'a, T> for SortedMinMax<'a, T> {
    unsafe fn new(slice: &'a [T], validity: &'a Bitmap, start: usize, end: usize) -> Self {
        let mut out = Self {
            slice,
            validity,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        let nulls = out.count_nulls(start, end);
        out.null_count = nulls;
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        self.null_count -= self.count_nulls(self.last_start, start);
        self.null_count += self.count_nulls(self.last_end, end);

        self.last_start = start;
        self.last_end = end;

        // return first non null
        for idx in start..end {
            let valid = self.validity.get_bit_unchecked(idx);

            if valid {
                return Some(*self.slice.get_unchecked(idx));
            }
        }

        None
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

/// Generic `Min` / `Max` kernel.
pub struct MinMaxWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    slice: &'a [T],
    validity: &'a Bitmap,
    extremum: Option<T>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
    compare_fn_nan: fn(&T, &T) -> Ordering,
    take_extremum: fn(T, T) -> T,
    // ordering on which the window needs to act.
    // for min kernel this is Less
    // for max kernel this is Greater
    agg_ordering: Ordering,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> MinMaxWindow<'a, T> {
    unsafe fn compute_extremum_in_between_leaving_and_entering(&self, start: usize) -> Option<T> {
        // check the values in between the window that remains e.g. is not leaving
        // this between `start..last_end`
        //
        // because we know the current `min` (which might be leaving), we know we can stop
        // searching if any value is equal to current `min`.
        let mut extremum_in_between = None;
        for idx in start..self.last_end {
            let valid = self.validity.get_bit_unchecked(idx);
            let value = self.slice.get_unchecked(idx);

            if valid {
                // early return
                if let Some(current_min) = self.extremum {
                    if matches!(compare_fn_nan_min(value, &current_min), Ordering::Equal) {
                        return Some(current_min);
                    }
                }

                match extremum_in_between {
                    None => extremum_in_between = Some(*value),
                    Some(current) => {
                        extremum_in_between = Some((self.take_extremum)(*value, current))
                    }
                }
            }
        }
        extremum_in_between
    }

    // compute min from the entire window
    unsafe fn compute_extremum_and_update_null_count(
        &mut self,
        start: usize,
        end: usize,
    ) -> Option<T> {
        let mut extremum = None;
        let mut idx = start;
        for value in &self.slice[start..end] {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                match extremum {
                    None => extremum = Some(*value),
                    Some(current) => extremum = Some((self.take_extremum)(*value, current)),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        extremum
    }

    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        compare_fn: fn(&T, &T) -> Ordering,
        take_extremum: fn(T, T) -> T,
        agg_ordering: Ordering,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            extremum: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            compare_fn_nan: compare_fn,
            take_extremum,
            agg_ordering,
        };
        let extremum = out.compute_extremum_and_update_null_count(start, end);
        out.extremum = extremum;
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        // recompute min
        if start >= self.last_end {
            self.extremum = self.compute_extremum_and_update_null_count(start, end);
            self.last_end = end;
            self.last_start = start;
            return self.extremum;
        }

        // remove elements that should leave the window
        let mut recompute_extremum = false;
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                let leaving_value = self.slice.get_unchecked(idx);

                // if the leaving value is the
                // min value, we need to recompute the min.
                if matches!(
                    (self.compare_fn_nan)(leaving_value, &self.extremum.unwrap()),
                    Ordering::Equal
                ) {
                    recompute_extremum = true;
                    break;
                }
            } else {
                // null value leaving the window
                self.null_count -= 1;

                // self.min is None and the leaving value is None
                // if the entering value is valid, we might get a new min.
                if self.extremum.is_none() {
                    recompute_extremum = true;
                    break;
                }
            }
        }

        let entering_extremum = self.compute_extremum_and_update_null_count(self.last_end, end);

        match (self.extremum, entering_extremum) {
            // all remains `None`
            (None, None) => {}
            (None, Some(new_min)) => self.extremum = Some(new_min),
            // entering min is `None` and the `min` is leaving, so the `in_between` min is the new
            // minimum.
            // if min is not leaving, we don't do anything
            (Some(_current_min), None) => {
                if recompute_extremum {
                    self.extremum = self.compute_extremum_in_between_leaving_and_entering(start);
                }
            }
            (Some(current_extremum), Some(entering_extremum)) => {
                if recompute_extremum {
                    match (self.compare_fn_nan)(&current_extremum, &entering_extremum) {
                        // do nothing
                        Ordering::Equal => {}
                        // leaving < entering
                        ord if ord == self.agg_ordering => {
                            // leaving value could be the smallest, we might need to recompute

                            let min_in_between =
                                self.compute_extremum_in_between_leaving_and_entering(start);
                            match min_in_between {
                                None => self.extremum = Some(entering_extremum),
                                Some(extremum_in_between) => {
                                    if (self.compare_fn_nan)(
                                        &extremum_in_between,
                                        &entering_extremum,
                                    ) == self.agg_ordering
                                    {
                                        self.extremum = Some(extremum_in_between)
                                    } else {
                                        self.extremum = Some(entering_extremum)
                                    }
                                }
                            }
                        }
                        // leaving > entering
                        _ => {
                            if (self.compare_fn_nan)(&entering_extremum, &current_extremum)
                                == self.agg_ordering
                            {
                                self.extremum = Some(entering_extremum)
                            }
                        }
                    }
                } else if (self.compare_fn_nan)(&entering_extremum, &current_extremum)
                    == self.agg_ordering
                {
                    self.extremum = Some(entering_extremum)
                }
            }
        }
        self.last_start = start;
        self.last_end = end;
        self.extremum
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

pub struct MinWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    inner: MinMaxWindow<'a, T>,
}

fn take_min<T: NativeType + IsFloat + PartialOrd>(a: T, b: T) -> T {
    std::cmp::min_by(a, b, compare_fn_nan_min)
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T> for MinWindow<'a, T> {
    unsafe fn new(slice: &'a [T], validity: &'a Bitmap, start: usize, end: usize) -> Self {
        Self {
            inner: MinMaxWindow::new(
                slice,
                validity,
                start,
                end,
                compare_fn_nan_min,
                take_min,
                Ordering::Less,
            ),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        self.inner.update(start, end)
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.inner.is_valid(min_periods)
    }
}

pub fn rolling_min<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd + Bounded + IsFloat,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<MinWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
        )
    } else {
        rolling_apply_agg_window::<MinWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
        )
    }
}

pub struct MaxWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    inner: MinMaxWindow<'a, T>,
}

fn take_max<T: NativeType + IsFloat + PartialOrd>(a: T, b: T) -> T {
    std::cmp::max_by(a, b, compare_fn_nan_max)
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T> for MaxWindow<'a, T> {
    unsafe fn new(slice: &'a [T], validity: &'a Bitmap, start: usize, end: usize) -> Self {
        Self {
            inner: MinMaxWindow::new(
                slice,
                validity,
                start,
                end,
                compare_fn_nan_max,
                take_max,
                Ordering::Greater,
            ),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        self.inner.update(start, end)
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.inner.is_valid(min_periods)
    }
}

pub fn rolling_max<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd + Bounded + IsFloat,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        if is_reverse_sorted_max_nulls(arr.values().as_slice(), arr.validity().as_ref().unwrap()) {
            rolling_apply_agg_window::<SortedMinMax<_>, _, _>(
                arr.values().as_slice(),
                arr.validity().as_ref().unwrap(),
                window_size,
                min_periods,
                det_offsets_center,
            )
        } else {
            rolling_apply_agg_window::<MaxWindow<_>, _, _>(
                arr.values().as_slice(),
                arr.validity().as_ref().unwrap(),
                window_size,
                min_periods,
                det_offsets_center,
            )
        }
    } else if is_reverse_sorted_max_nulls(arr.values().as_slice(), arr.validity().as_ref().unwrap())
    {
        rolling_apply_agg_window::<SortedMinMax<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
        )
    } else {
        rolling_apply_agg_window::<MaxWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
        )
    }
}
