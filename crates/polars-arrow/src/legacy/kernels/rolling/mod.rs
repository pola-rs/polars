pub mod no_nulls;
pub mod nulls;
pub mod quantile_filter;
mod window;

use std::any::Any;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use std::sync::Arc;

use num_traits::{Bounded, Float, NumCast, One, Zero};
use polars_utils::float::IsFloat;
use polars_utils::ord::{compare_fn_nan_max, compare_fn_nan_min};
use window::*;

use crate::array::{ArrayRef, PrimitiveArray};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::legacy::prelude::*;
use crate::legacy::utils::CustomIterTools;
use crate::types::NativeType;

type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;
pub type DynArgs = Option<Arc<dyn Any + Sync + Send>>;

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

        // Set the null values at the boundaries

        // Head.
        for i in 0..len {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }
        // Tail.
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

// Parameters allowed for rolling operations.
#[derive(Clone, Copy, Debug)]
pub struct RollingVarParams {
    pub ddof: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct RollingQuantileParams {
    pub prob: f64,
    pub interpol: QuantileInterpolOptions,
}
