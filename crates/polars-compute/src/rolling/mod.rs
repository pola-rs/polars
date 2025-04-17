mod min_max;
pub mod no_nulls;
pub mod nulls;
pub mod quantile_filter;
mod window;

use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
use num_traits::{Bounded, Float, NumCast, One, Zero};
use polars_utils::float::IsFloat;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;
use window::*;

type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum QuantileMethod {
    #[default]
    Nearest,
    Lower,
    Higher,
    Midpoint,
    Linear,
    Equiprobable,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct QuantileOptions {
    pub prob: f64,
    pub method: QuantileMethod,
}

impl Eq for QuantileOptions {}

impl Hash for QuantileOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prob.to_bits().hash(state);
        self.method.hash(state);
    }
}

#[deprecated(note = "use QuantileMethod instead")]
pub type QuantileInterpolOptions = QuantileMethod;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RollingFnParams {
    Quantile(RollingQuantileParams),
    Var(RollingVarParams),
}

fn det_offsets(i: Idx, window_size: WindowSize, _len: Len) -> (usize, usize) {
    (i.saturating_sub(window_size - 1), i + 1)
}
fn det_offsets_center(i: Idx, window_size: WindowSize, len: Len) -> (usize, usize) {
    let right_window = window_size.div_ceil(2);
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
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingVarParams {
    pub ddof: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingQuantileParams {
    pub prob: f64,
    pub method: QuantileMethod,
}
