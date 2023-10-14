pub mod no_nulls;
pub mod nulls;
mod window;

use std::any::Any;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use std::sync::Arc;

use num_traits::{Bounded, Float, NumCast, One, Zero};
use window::*;

use crate::array::PrimitiveArray;
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::legacy::data_types::IsFloat;
use crate::legacy::prelude::*;
use crate::legacy::utils::CustomIterTools;
use crate::types::NativeType;

type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;
pub type DynArgs = Option<Arc<dyn Any + Sync + Send>>;

#[inline]
/// NaN will be smaller than every valid value
pub fn compare_fn_nan_min<T>(a: &T, b: &T) -> Ordering
where
    T: PartialOrd + IsFloat,
{
    // this branch should be optimized away for integers
    if T::is_float() {
        match (a.is_nan(), b.is_nan()) {
            // safety: we checked nans
            (false, false) => unsafe { a.partial_cmp(b).unwrap_unchecked() },
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
        }
    } else {
        // Safety:
        // all integers are Ord
        unsafe { a.partial_cmp(b).unwrap_unchecked() }
    }
}

#[inline]
/// NaN will be larger than every valid value
pub fn compare_fn_nan_max<T>(a: &T, b: &T) -> Ordering
where
    T: PartialOrd + IsFloat,
{
    // this branch should be optimized away for integers
    if T::is_float() {
        match (a.is_nan(), b.is_nan()) {
            // safety: we checked nans
            (false, false) => unsafe { a.partial_cmp(b).unwrap_unchecked() },
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
        }
    } else {
        // Safety:
        // all integers are Ord
        unsafe { a.partial_cmp(b).unwrap_unchecked() }
    }
}

fn det_offsets(i: Idx, window_size: WindowSize, _len: Len) -> (usize, usize) {
    (i.saturating_sub(window_size - 1), i + 1)
}
fn det_offsets_center(i: Idx, window_size: WindowSize, len: Len) -> (usize, usize) {
    let right_window = (window_size + 1) / 2;
    (
        i.saturating_sub(window_size - right_window),
        std::cmp::min(len, i + right_window),
    )
}

fn create_validity<Fo>(
    min_periods: usize,
    len: usize,
    window_size: usize,
    det_offsets_fn: Fo,
) -> Option<MutableBitmap>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
{
    if min_periods > 1 {
        let mut validity = MutableBitmap::with_capacity(len);
        validity.extend_constant(len, true);

        // set the null values at the boundaries

        // head
        for i in 0..len {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }
        // tail
        for i in (0..len).rev() {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }

        Some(validity)
    } else {
        None
    }
}
pub(super) fn sort_buf<T>(buf: &mut [T])
where
    T: IsFloat + NativeType + PartialOrd,
{
    if T::is_float() {
        buf.sort_by(|a, b| {
            match (a.is_nan(), b.is_nan()) {
                // safety: we checked nans
                (false, false) => unsafe { a.partial_cmp(b).unwrap_unchecked() },
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
            }
        });
    } else {
        // Safety:
        // all integers are Ord
        unsafe { buf.sort_by(|a, b| a.partial_cmp(b).unwrap_unchecked()) };
    }
}

//Parameters allowed for rolling operations.
#[derive(Clone, Copy, Debug)]
pub struct RollingVarParams {
    pub ddof: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct RollingQuantileParams {
    pub prob: f64,
    pub interpol: QuantileInterpolOptions,
}
