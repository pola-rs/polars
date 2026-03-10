use std::ops::{Add, AddAssign, Sub, SubAssign};

use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;
use super::*;

pub struct SumWindow<'a, T, S> {
    slice: &'a [T],
    validity: Option<&'a Bitmap>,
    sum: S,
    err_add: S,
    err_sub: S,
    non_finite_count: usize, // NaN or infinity.
    pos_inf_count: usize,
    neg_inf_count: usize,
    pub(super) null_count: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

impl<'a, T, S> SumWindow<'a, T, S>
where
    T: NativeType + IsFloat + Sub<Output = T> + NumCast + PartialOrd,
    S: NativeType + AddAssign + SubAssign + Sub<Output = S> + Add<Output = S> + NumCast,
{
    fn new_impl(slice: &'a [T], validity: Option<&'a Bitmap>) -> Self {
        Self {
            slice,
            validity,
            sum: S::zeroed(),
            err_add: S::zeroed(),
            err_sub: S::zeroed(),
            non_finite_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            null_count: 0,
            start: 0,
            end: 0,
        }
    }

    fn reset(&mut self) {
        self.sum = S::zeroed();
        self.err_add = S::zeroed();
        self.err_sub = S::zeroed();
        self.non_finite_count = 0;
        self.pos_inf_count = 0;
        self.neg_inf_count = 0;
        self.null_count = 0;
    }

    fn add_finite_kahan(&mut self, val: T) {
        let val: S = NumCast::from(val).unwrap();
        let y = val - self.err_add;
        let new_sum = self.sum + y;
        self.err_add = (new_sum - self.sum) - y;
        self.sum = new_sum;
    }

    fn sub_finite_kahan(&mut self, val: T) {
        let val: S = NumCast::from(T::zeroed() - val).unwrap();
        let y = val - self.err_sub;
        let new_sum = self.sum + y;
        self.err_sub = (new_sum - self.sum) - y;
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
                self.sub_finite_kahan(val);
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

    fn get_sum(&self) -> Option<T> {
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

impl<T, S> RollingAggWindowNoNulls<T> for SumWindow<'_, T, S>
where
    T: NativeType + IsFloat + Sub<Output = T> + NumCast + PartialOrd,
    S: NativeType + AddAssign + SubAssign + Sub<Output = S> + Add<Output = S> + NumCast,
{
    type This<'a> = SumWindow<'a, T, S>;

    fn new<'a>(
        slice: &'a [T],
        start: usize,
        end: usize,
        _params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        let mut out = SumWindow::new_impl(slice, None);
        unsafe { RollingAggWindowNoNulls::update(&mut out, start, end) };
        out
    }

    // # Safety
    // The start, end range must be in-bounds.
    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        if new_start >= self.end {
            self.reset();
            self.start = new_start;
            self.end = new_start;
        }

        for val in &self.slice[self.start..new_start] {
            self.sub(*val);
        }

        for val in &self.slice[self.end..new_end] {
            self.add(*val);
        }

        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.get_sum()
    }

    fn slice_len(&self) -> usize {
        self.slice.len()
    }
}

impl<T, S> RollingAggWindowNulls<T> for SumWindow<'_, T, S>
where
    T: NativeType + IsFloat + Sub<Output = T> + NumCast + PartialOrd,
    S: NativeType + AddAssign + SubAssign + Sub<Output = S> + Add<Output = S> + NumCast,
{
    type This<'a> = SumWindow<'a, T, S>;

    fn new<'a>(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        _params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(start <= slice.len() && end <= slice.len() && start <= end);
        let mut out = SumWindow::new_impl(slice, Some(validity));
        // SAFETY: We bounds checked `start` and `end`.
        unsafe { RollingAggWindowNulls::update(&mut out, start, end) };
        out
    }

    // # Safety
    // The start, end range must be in-bounds.
    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        let validity = unsafe { self.validity.unwrap_unchecked() };

        if new_start >= self.end {
            self.reset();
            self.start = new_start;
            self.end = new_start;
        }

        for idx in self.start..new_start {
            let valid = unsafe { validity.get_bit_unchecked(idx) };
            if valid {
                self.sub(unsafe { *self.slice.get_unchecked(idx) });
            } else {
                self.null_count -= 1;
            }
        }

        for idx in self.end..new_end {
            let valid = unsafe { validity.get_bit_unchecked(idx) };
            if valid {
                self.add(unsafe { *self.slice.get_unchecked(idx) });
            } else {
                self.null_count += 1;
            }
        }

        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.get_sum()
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.end - self.start) - self.null_count) >= min_periods
    }

    fn slice_len(&self) -> usize {
        self.slice.len()
    }
}
