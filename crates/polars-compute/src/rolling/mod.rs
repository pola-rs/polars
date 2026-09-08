mod mean;
mod min_max;
mod moment;
pub mod no_nulls;
pub mod nulls;
pub mod quantile_filter;
mod rank;
mod sum;

mod arg_min_max;
mod min_by_max_by;
pub use min_by_max_by::*;
pub(super) mod window;
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

pub use arg_min_max::{ArgMaxWindow, ArgMinMaxWindow, ArgMinWindow};
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
pub use mean::MeanWindow;
use num_traits::{Bounded, Float, NumCast, One, Zero};
use polars_utils::float::IsFloat;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;
pub use sum::SumWindow;
use window::*;

type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RollingFnParams {
    Quantile(RollingQuantileParams),
    Var(RollingVarParams),
    Rank {
        method: RollingRankMethod,
        seed: Option<u64>,
    },
    Skew {
        bias: bool,
    },
    Kurtosis {
        fisher: bool,
        bias: bool,
    },
}

fn det_offsets(i: Idx, window_size: WindowSize, _len: Len) -> (usize, usize) {
    if window_size == 0 {
        return (i, i);
    }
    (i.saturating_sub(window_size - 1), i + 1)
}
fn det_offsets_center(i: Idx, window_size: WindowSize, len: Len) -> (usize, usize) {
    if window_size == 0 {
        return (i, i);
    }
    let right_window = window_size.div_ceil(2);
    (
        i.saturating_sub(window_size - right_window),
        std::cmp::min(len, i + right_window),
    )
}

/// Compute the validity for a rolling aggregation.
///
/// `weights` may only be passed if the weights as a whole don't sum to zero; the caller is
/// expected to reject that up front, since it makes the aggregation undefined everywhere.
fn create_validity<Fo>(
    min_periods: usize,
    len: usize,
    window_size: usize,
    det_offsets_fn: Fo,
    weights: Option<&[f64]>,
    centered: bool,
) -> Option<MutableBitmap>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
{
    // Short path:
    // If there are no zero weights, then there can be no invalid values due to weights.
    let weights = weights.filter(|w| w.iter().any(|&w| w == 0.0));

    if min_periods <= 1 && weights.is_none() {
        return None;
    }

    let mut validity = MutableBitmap::with_capacity(len);
    validity.extend_constant(len, true);

    if min_periods > 1 {
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
    }

    // ASSUMPTION: the sum of *all* weights is not 0.
    // This should be caught by the DSL.
    // This only leaves an invalid possibility if a truncated window's sums are zero.
    // Moving along from a boundary the 'valid' window grows, so each window covers a superset of the
    // weights of the one before it. Once a window covers a nonzero weight, so does every window
    // after it
    if let Some(weights) = weights {
        let covers_only_zero_weights = |i: usize| {
            let (start, end) = det_offsets_fn(i, window_size, len);
            let win_len = end - start;
            let weights_start =
                no_nulls::det_weights_start(centered, window_size, i, start, win_len);
            weights[weights_start..weights_start + win_len]
                .iter()
                .all(|&w| w == 0.0)
        };

        // Head.
        for i in 0..len {
            if !covers_only_zero_weights(i) {
                break;
            }
            validity.set(i, false)
        }
        // Tail.
        for i in (0..len).rev() {
            if !covers_only_zero_weights(i) {
                break;
            }
            validity.set(i, false)
        }
    }

    Some(validity)
}

// Parameters allowed for rolling operations.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RollingVarParams {
    pub ddof: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RollingQuantileParams {
    pub prob: f64,
    pub method: QuantileMethod,
}

impl Hash for RollingQuantileParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Will not be NaN, so hash + eq symmetry will hold.
        self.prob.to_bits().hash(state);
        self.method.hash(state);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum RollingRankMethod {
    #[default]
    Average,
    Min,
    Max,
    Dense,
    Random,
}
