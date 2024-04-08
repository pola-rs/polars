mod mean;
mod min_max;
mod quantile;
mod sum;
mod variance;
use std::fmt::Debug;

pub use mean::*;
pub use min_max::*;
use num_traits::{Float, Num, NumCast};
pub use quantile::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use sum::*;
pub use variance::*;

use super::*;
use crate::array::PrimitiveArray;
use crate::datatypes::ArrowDataType;
use crate::legacy::error::PolarsResult;
use crate::types::NativeType;

pub trait RollingAggWindowNoNulls<'a, T: NativeType> {
    fn new(slice: &'a [T], start: usize, end: usize, params: DynArgs) -> Self;

    /// Update and recompute the window
    ///
    /// # Safety
    /// `start` and `end` must be within the windows bounds
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T>;
}

// Use an aggregation window that maintains the state
pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Agg: RollingAggWindowNoNulls<'a, T>,
    T: Debug + NativeType + Num,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    let mut agg_window = Agg::new(values, start, end, params);
    if let Some(validity) = create_validity(min_periods, len, window_size, &det_offsets_fn) {
        if validity.iter().all(|x| !x) {
            return Ok(Box::new(PrimitiveArray::<T>::new_null(
                T::PRIMITIVE.into(),
                len,
            )));
        }
    }

    let out = (0..len).map(|idx| {
        let (start, end) = det_offsets_fn(idx, window_size, len);
        if end - start < min_periods {
            None
        } else {
            // SAFETY:
            // we are in bounds
            unsafe { agg_window.update(start, end) }
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Ok(Box::new(arr))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QuantileInterpolOptions {
    #[default]
    Nearest,
    Lower,
    Higher,
    Midpoint,
    Linear,
}

pub(super) fn rolling_apply_weights<T, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[T],
) -> PolarsResult<ArrayRef>
where
    T: NativeType,
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], &[T]) -> T,
{
    assert_eq!(weights.len(), window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len, window_size, det_offsets_fn);
    Ok(Box::new(PrimitiveArray::new(
        ArrowDataType::from(T::PRIMITIVE),
        out.into(),
        validity.map(|b| b.into()),
    )))
}

fn compute_var_weights<T>(vals: &[T], weights: &[T]) -> T
where
    T: Float + std::ops::AddAssign,
{
    // Assumes the weights have already been standardized to 1
    debug_assert!(
        weights.iter().fold(T::zero(), |acc, x| acc + *x) == T::one(),
        "Rolling weighted variance Weights don't sum to 1"
    );
    let (wssq, wmean) = vals
        .iter()
        .zip(weights)
        .fold((T::zero(), T::zero()), |(wssq, wsum), (&v, &w)| {
            (wssq + v * v * w, wsum + v * w)
        });

    wssq - wmean * wmean
}

pub(crate) fn compute_sum_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: std::iter::Sum<T> + Copy + std::ops::Mul<Output = T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum()
}

pub(super) fn coerce_weights<T: NumCast>(weights: &[f64]) -> Vec<T>
where
{
    weights
        .iter()
        .map(|v| NumCast::from(*v).unwrap())
        .collect::<Vec<_>>()
}
