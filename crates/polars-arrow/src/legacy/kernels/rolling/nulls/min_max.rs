use super::*;
use crate::array::iterator::NonNullValuesIter;
use crate::bitmap::utils::count_zeros;

pub fn is_reverse_sorted_max_nulls<T: NativeType>(values: &[T], validity: &Bitmap) -> bool {
    let mut it = NonNullValuesIter::new(values, Some(validity));
    let Some(mut prev) = it.next() else {
        return true;
    };
    for v in it {
        if prev.tot_lt(&v) {
            return false;
        }
        prev = v
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
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        _params: DynArgs,
    ) -> Self {
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
    is_better: fn(&T, &T) -> bool,
    take_extremum: fn(T, T) -> T,
    // ordering on which the window needs to act.
    // for min kernel this is Less
    // for max kernel this is Greater
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
                    if value.tot_eq(&current_min) {
                        return Some(current_min);
                    }
                }

                match extremum_in_between {
                    None => extremum_in_between = Some(*value),
                    Some(current) => {
                        extremum_in_between = Some((self.take_extremum)(*value, current))
                    },
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
        is_better: fn(&T, &T) -> bool,
        take_extremum: fn(T, T) -> T,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            extremum: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            is_better,
            take_extremum,
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
            // SAFETY:
            // we are in bounds
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                let leaving_value = self.slice.get_unchecked(idx);

                // if the leaving value is the
                // min value, we need to recompute the min.
                if leaving_value.tot_eq(&self.extremum.unwrap()) {
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
            (None, None) => {},
            (None, Some(new_min)) => self.extremum = Some(new_min),
            // entering min is `None` and the `min` is leaving, so the `in_between` min is the new
            // minimum.
            // if min is not leaving, we don't do anything
            (Some(_current_min), None) => {
                if recompute_extremum {
                    self.extremum = self.compute_extremum_in_between_leaving_and_entering(start);
                }
            },
            (Some(current_extremum), Some(entering_extremum)) => {
                if (self.is_better)(&entering_extremum, &current_extremum) {
                    self.extremum = Some(entering_extremum)
                } else if recompute_extremum
                    && (self.is_better)(&current_extremum, &entering_extremum)
                {
                    // leaving value could be the smallest, we might need to recompute
                    let min_in_between =
                        self.compute_extremum_in_between_leaving_and_entering(start);
                    match min_in_between {
                        None => self.extremum = Some(entering_extremum),
                        Some(extremum_in_between) => {
                            self.extremum =
                                Some((self.take_extremum)(extremum_in_between, entering_extremum));
                        },
                    }
                }
            },
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

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T> for MinWindow<'a, T> {
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        _params: DynArgs,
    ) -> Self {
        Self {
            inner: MinMaxWindow::new(
                slice,
                validity,
                start,
                end,
                |a, b| a.nan_max_lt(b),
                |a, b| a.min_ignore_nan(b),
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
    _params: DynArgs,
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
            None,
        )
    } else {
        rolling_apply_agg_window::<MinWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}

pub struct MaxWindow<'a, T: NativeType + PartialOrd + IsFloat> {
    inner: MinMaxWindow<'a, T>,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> RollingAggWindowNulls<'a, T> for MaxWindow<'a, T> {
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        _params: DynArgs,
    ) -> Self {
        Self {
            inner: MinMaxWindow::new(
                slice,
                validity,
                start,
                end,
                |a, b| b.nan_min_lt(a),
                |a, b| a.max_ignore_nan(b),
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
    _params: DynArgs,
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
                None,
            )
        } else {
            rolling_apply_agg_window::<MaxWindow<_>, _, _>(
                arr.values().as_slice(),
                arr.validity().as_ref().unwrap(),
                window_size,
                min_periods,
                det_offsets_center,
                None,
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
            None,
        )
    } else {
        rolling_apply_agg_window::<MaxWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}
