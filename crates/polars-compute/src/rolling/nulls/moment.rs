#![allow(unsafe_op_in_unsafe_fn)]

use num_traits::{FromPrimitive, ToPrimitive};

pub use super::super::moment::*;
use super::*;

pub struct MomentWindow<'a, T, M: StateUpdate> {
    slice: &'a [T],
    validity: &'a Bitmap,
    moment: Option<M>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
    params: Option<RollingFnParams>,
}

impl<T: NativeType + ToPrimitive, M: StateUpdate> MomentWindow<'_, T, M> {
    // compute sum from the entire window
    unsafe fn compute_moment_and_null_count(&mut self, start: usize, end: usize) {
        self.moment = None;
        let mut idx = start;
        self.null_count = 0;
        for value in &self.slice[start..end] {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                let value: f64 = NumCast::from(*value).unwrap();
                self.moment
                    .get_or_insert_with(|| M::new(self.params))
                    .insert_one(value);
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
    }
}

impl<'a, T: NativeType + ToPrimitive + IsFloat + FromPrimitive, M: StateUpdate>
    RollingAggWindowNulls<'a, T> for MomentWindow<'a, T, M>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            moment: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            params,
        };
        out.compute_moment_and_null_count(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let recompute_var = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_var = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let valid = self.validity.get_bit_unchecked(idx);
                if valid {
                    let leaving_value = *self.slice.get_unchecked(idx);

                    // if the leaving value is nan we need to recompute the window
                    if T::is_float() && !leaving_value.is_finite() {
                        recompute_var = true;
                        break;
                    }
                    let leaving_value: f64 = NumCast::from(leaving_value).unwrap();
                    if let Some(v) = self.moment.as_mut() {
                        v.remove_one(leaving_value)
                    }
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.sum is None and the leaving value is None
                    // if the entering value is valid, we might get a new sum.
                    if self.moment.is_none() {
                        recompute_var = true;
                        break;
                    }
                }
            }
            recompute_var
        };

        self.last_start = start;

        // we traverse all values and compute
        if recompute_var {
            self.compute_moment_and_null_count(start, end);
        } else {
            for idx in self.last_end..end {
                let valid = self.validity.get_bit_unchecked(idx);

                if valid {
                    let entering_value = *self.slice.get_unchecked(idx);
                    let entering_value: f64 = NumCast::from(entering_value).unwrap();
                    self.moment
                        .get_or_insert_with(|| M::new(self.params))
                        .insert_one(entering_value);
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.moment.as_ref().and_then(|v| {
            let out = v.finalize();
            out.map(|v| T::from_f64(v).unwrap())
        })
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

pub fn rolling_var<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, VarianceMoment>, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}

pub fn rolling_skew<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, SkewMoment>, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}

pub fn rolling_kurtosis<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + FromPrimitive + IsFloat + Float,
{
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<MomentWindow<_, KurtosisMoment>, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}
