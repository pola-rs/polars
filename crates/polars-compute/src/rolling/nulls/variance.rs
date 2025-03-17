#![allow(unsafe_op_in_unsafe_fn)]
use num_traits::{FromPrimitive, ToPrimitive};

use super::*;
use crate::var_cov::VarState;

pub struct VarWindow<'a, T> {
    slice: &'a [T],
    validity: &'a Bitmap,
    var: Option<VarState>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
    ddof: u8,
}

impl<T: NativeType + ToPrimitive> VarWindow<'_, T> {
    // compute sum from the entire window
    unsafe fn compute_var_and_null_count(&mut self, start: usize, end: usize) {
        let mut var = None;
        let mut idx = start;
        self.null_count = 0;
        for value in &self.slice[start..end] {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                let value: f64 = NumCast::from(*value).unwrap();
                match &mut var {
                    None => var = Some(VarState::new_single(value)),
                    Some(current) => current.insert_one(value),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        self.var = var;
    }
}

impl<'a, T: NativeType + ToPrimitive + IsFloat + FromPrimitive> RollingAggWindowNulls<'a, T>
    for VarWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
    ) -> Self {
        let ddof = if let Some(RollingFnParams::Var(params)) = params {
            params.ddof
        } else {
            1
        };

        let mut out = Self {
            slice,
            validity,
            var: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            ddof,
        };
        out.compute_var_and_null_count(start, end);
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
                    if let Some(v) = self.var.as_mut() {
                        v.remove_one(leaving_value)
                    }
                } else {
                    // null value leaving the window
                    self.null_count -= 1;

                    // self.sum is None and the leaving value is None
                    // if the entering value is valid, we might get a new sum.
                    if self.var.is_none() {
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
            self.compute_var_and_null_count(start, end);
        } else {
            for idx in self.last_end..end {
                let valid = self.validity.get_bit_unchecked(idx);

                if valid {
                    let entering_value = *self.slice.get_unchecked(idx);
                    let entering_value: f64 = NumCast::from(entering_value).unwrap();

                    match &mut self.var {
                        None => self.var = Some(VarState::new_single(entering_value)),
                        Some(current) => current.insert_one(entering_value),
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        self.var.as_ref().and_then(|v| {
            let out = v.finalize(self.ddof);
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
    rolling_apply_agg_window::<VarWindow<_>, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}
