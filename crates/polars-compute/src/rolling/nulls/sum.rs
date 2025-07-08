#![allow(unsafe_op_in_unsafe_fn)]
use super::*;

pub struct SumWindow<'a, T, S> {
    slice: &'a [T],
    validity: &'a Bitmap,
    sum: S,
    err: S,
    non_finite_count: usize, // NaN or infinity.
    pos_inf_count: usize,
    neg_inf_count: usize,
    pub(super) null_count: usize,
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

impl<'a, T, S> RollingAggWindowNulls<'a, T> for SumWindow<'a, T, S>
where
    T: NativeType + IsFloat + Sub<Output = T> + NumCast + PartialOrd,
    S: NativeType + AddAssign + SubAssign + Sub<Output = S> + Add<Output = S> + NumCast,
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        _params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            sum: S::zeroed(),
            err: S::zeroed(),
            non_finite_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            last_start: 0,
            last_end: 0,
            null_count: 0,
        };
        out.update(start, end);
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
            self.null_count = 0;
            self.last_start = start;
            self.last_end = start;
        }

        for idx in self.last_start..start {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                self.sub(unsafe { *self.slice.get_unchecked(idx) });
            } else {
                self.null_count -= 1;
            }
        }

        for idx in self.last_end..end {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                self.add(unsafe { *self.slice.get_unchecked(idx) });
            } else {
                self.null_count += 1;
            }
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

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

pub fn rolling_sum<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType
        + IsFloat
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + SubAssign
        + AddAssign
        + NumCast,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<SumWindow<T, T>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<SumWindow<T, T>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}
