use super::*;
use nulls;
use nulls::{rolling_apply_agg_window, RollingAggWindowNulls};

pub struct MinWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    slice: &'a [T],
    validity: &'a Bitmap,
    min: Option<T>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> MinWindow<'a, T> {
    // compute min from the entire window
    unsafe fn compute_min(&mut self, start: usize, end: usize) -> Option<T> {
        let mut min = None;
        let mut idx = start;
        self.null_count = 0;
        for value in (&self.slice[start..end]).iter() {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                match min {
                    None => min = Some(*value),
                    Some(current) => {
                        min = Some(std::cmp::min_by(*value, current, compare_fn_nan_min))
                    }
                }
            }
            idx += 1;
        }
        min
    }

    unsafe fn compute_min_in_between_leaving_and_entering(&self, start: usize) -> Option<T> {
        // check the values in between the window that remains e.g. is not leaving
        // this between `start..last_end`
        //
        // because we know the current `min` (which might be leaving), we know we can stop
        // searching if any value is equal to current `min`.
        let mut min_in_between = None;
        for idx in start..self.last_end {
            let valid = self.validity.get_bit_unchecked(idx);
            let value = self.slice.get_unchecked(idx);

            if valid {
                // early return
                if let Some(current_min) = self.min {
                    if matches!(compare_fn_nan_min(value, &current_min), Ordering::Equal) {
                        return Some(current_min);
                    }
                }

                match min_in_between {
                    None => min_in_between = Some(*value),
                    Some(current) => {
                        min_in_between = Some(std::cmp::min_by(*value, current, compare_fn_nan_min))
                    }
                }
            }
        }
        min_in_between
    }

    // compute min from the entire window
    unsafe fn compute_min_and_update_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut min = None;
        let mut idx = start;
        for value in (&self.slice[start..end]).iter() {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                match min {
                    None => min = Some(*value),
                    Some(current) => {
                        min = Some(std::cmp::min_by(*value, current, compare_fn_nan_min))
                    }
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        min
    }
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T> for MinWindow<'a, T> {
    unsafe fn new(slice: &'a [T], validity: &'a Bitmap, start: usize, end: usize) -> Self {
        let mut out = Self {
            slice,
            validity,
            min: None,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        let min = out.compute_min_and_update_null_count(start, end);
        out.min = min;
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        // recompute min
        if start >= self.last_end {
            self.min = self.compute_min_and_update_null_count(start, end);
            return self.min;
        }

        // remove elements that should leave the window
        let mut recompute_min = false;
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                let leaving_value = self.slice.get_unchecked(idx);

                // if the leaving value is the
                // min value, we need to recompute the min.
                if matches!(
                    compare_fn_nan_min(leaving_value, &self.min.unwrap()),
                    Ordering::Equal
                ) {
                    recompute_min = true;
                    break;
                }
            } else {
                // null value leaving the window
                self.null_count -= 1;

                // self.min is None and the leaving value is None
                // if the entering value is valid, we might get a new min.
                if self.min.is_none() {
                    recompute_min = true;
                    break;
                }
            }
        }

        let entering_min = self.compute_min_and_update_null_count(self.last_end, end);

        match (self.min, entering_min) {
            // all remains `None`
            (None, None) => {}
            (None, Some(new_min)) => self.min = Some(new_min),
            // entering min is `None` and the `min` is leaving, so the `in_between` min is the new
            // minimum.
            // if min is not leaving, we don't do anything
            (Some(_current_min), None) => {
                if recompute_min {
                    self.min = self.compute_min_in_between_leaving_and_entering(start);
                }
            }
            (Some(current_min), Some(entering_min)) => {
                if recompute_min {
                    match compare_fn_nan_min(&current_min, &entering_min) {
                        // do nothing
                        Ordering::Equal => {}
                        // leaving < entering
                        Ordering::Less => {
                            // leaving value could be the smallest, we might need to recompute

                            let min_in_between =
                                self.compute_min_in_between_leaving_and_entering(start);
                            match min_in_between {
                                None => self.min = Some(entering_min),
                                Some(min_in_between) => {
                                    if matches!(
                                        compare_fn_nan_min(&min_in_between, &entering_min),
                                        Ordering::Less
                                    ) {
                                        self.min = Some(min_in_between)
                                    } else {
                                        self.min = Some(entering_min)
                                    }
                                }
                            }
                        }
                        // leaving > entering
                        Ordering::Greater => {
                            if matches!(
                                compare_fn_nan_min(&entering_min, &current_min),
                                Ordering::Less
                            ) {
                                self.min = Some(entering_min)
                            }
                        }
                    }
                } else if matches!(
                    compare_fn_nan_min(&entering_min, &current_min),
                    Ordering::Less
                ) {
                    self.min = Some(entering_min)
                }
            }
        }
        self.last_start = start;
        self.last_end = end;
        self.min
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        !(((self.last_end - self.last_start) - self.null_count) < min_periods)
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
    slice: &'a [T],
    validity: &'a Bitmap,
    max: Option<T>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> MaxWindow<'a, T> {
    // compute max from the entire window
    unsafe fn compute_max_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut max = None;
        let mut idx = start;
        self.null_count = 0;
        for value in (&self.slice[start..end]).iter() {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                match max {
                    None => max = Some(*value),
                    Some(current) => {
                        max = Some(std::cmp::max_by(*value, current, compare_fn_nan_max))
                    }
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        self.max = max;
        max
    }
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T> for MaxWindow<'a, T> {
    unsafe fn new(slice: &'a [T], validity: &'a Bitmap, start: usize, end: usize) -> Self {
        let mut out = Self {
            slice,
            validity,
            max: None,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        out.compute_max_and_null_count(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        // if we exceed the end, we have a completely new window
        // so we recompute
        let recompute_max = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_max = false;
            for idx in self.last_start..start {
                // safety
                // we are in bounds
                let valid = self.validity.get_bit_unchecked(idx);
                if valid {
                    let leaving_value = self.slice.get_unchecked(idx);

                    // if the leaving value is the
                    // max value, we need to recompute the max.
                    if matches!(
                        compare_fn_nan_max(leaving_value, &self.max.unwrap()),
                        Ordering::Equal
                    ) {
                        recompute_max = true;
                        break;
                    }
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.max is None and the leaving value is None
                    // if the entering value is valid, we might get a new max.
                    if self.max.is_none() {
                        recompute_max = true;
                        break;
                    }
                }
            }
            recompute_max
        };

        self.last_start = start;

        // we traverese all values and compute
        if recompute_max {
            self.compute_max_and_null_count(start, end);
        } else {
            // the max has not left the window, so we only check
            // if the entering values are larger
            for idx in self.last_end..end {
                let valid = self.validity.get_bit_unchecked(idx);

                if valid {
                    let value = *self.slice.get_unchecked(idx);
                    match self.max {
                        None => self.max = Some(value),
                        Some(current) => {
                            self.max = Some(std::cmp::max_by(value, current, compare_fn_nan_max))
                        }
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.max
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        !(((self.last_end - self.last_start) - self.null_count) < min_periods)
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
        rolling_apply_agg_window::<MaxWindow<_>, _, _>(
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
            det_offsets,
        )
    }
}
