//! This module implements logic shared between nulls and no_nulls.

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use bytemuck::allocation::zeroed_vec;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
use polars_compute::rolling::no_nulls::RollingAggWindowNoNulls;
use polars_compute::rolling::nulls::RollingAggWindowNulls;
use polars_core::prelude::*;

use crate::windows::duration::Duration;
use crate::windows::group_by::{ClosedWindow, group_by_values_iter};

pub(crate) trait RollingAggWindow<T: NativeType, Out: NativeType> {
    /// # Safety
    /// `start` and `end` must be in bounds of `slice` and associated structures.
    unsafe fn update(&mut self, start: usize, end: usize);

    /// Get the aggregate of the current window relative to the value at `idx`.
    fn get_agg(&self, idx: usize) -> Option<Out>;

    /// Returns the length of the underlying input.
    fn slice_len(&self) -> usize;
}

#[repr(transparent)]
pub(crate) struct RollingAggWindowNoNullsWrapper<T>(pub T);
#[repr(transparent)]
pub(crate) struct RollingAggWindowNullsWrapper<T>(pub T);

impl<T: NativeType, Out: NativeType, Agg: RollingAggWindowNoNulls<T, Out>> RollingAggWindow<T, Out>
    for RollingAggWindowNoNullsWrapper<Agg>
{
    unsafe fn update(&mut self, start: usize, end: usize) {
        // SAFETY: Caller MUST uphold function safety contract.
        unsafe { self.0.update(start, end) }
    }

    fn get_agg(&self, idx: usize) -> Option<Out> {
        self.0.get_agg(idx)
    }

    fn slice_len(&self) -> usize {
        self.0.slice_len()
    }
}

impl<T: NativeType, Out: NativeType, Agg: RollingAggWindowNulls<T, Out>> RollingAggWindow<T, Out>
    for RollingAggWindowNullsWrapper<Agg>
{
    unsafe fn update(&mut self, start: usize, end: usize) {
        // SAFETY: Caller MUST uphold function safety contract.
        unsafe { self.0.update(start, end) }
    }

    fn get_agg(&self, idx: usize) -> Option<Out> {
        self.0.get_agg(idx)
    }

    fn slice_len(&self) -> usize {
        self.0.slice_len()
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn rolling_apply_agg<T, Out, Agg>(
    agg_window: &mut Agg,
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType,
    Out: NativeType,
    Agg: RollingAggWindow<T, Out>,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;

    if let Some(indices) = sorting_indices {
        rolling_apply_agg_window(agg_window, offset_iter, min_periods, indices)
    } else {
        rolling_apply_agg_window_sorted(agg_window, offset_iter, min_periods)
    }
}

// Use an aggregation window that maintains the state.
// Fastpath if values were known to already be sorted by time.
fn rolling_apply_agg_window_sorted<Agg, O, T, Out>(
    agg_window: &mut Agg,
    offsets: O,
    min_periods: usize,
) -> PolarsResult<ArrayRef>
where
    Agg: RollingAggWindow<T, Out>,
    O: Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen,
    T: NativeType,
    Out: NativeType,
{
    let out = offsets
        .enumerate()
        .map(|(idx, result)| {
            result.map(|(start, len)| {
                let end = start + len;

                // On the Python side, if `min_periods` wasn't specified, it is set to
                // `1`. In that case, this condition is the same as checking
                // `if start == end`.
                if len < (min_periods as IdxSize) {
                    None
                } else {
                    // SAFETY: we are in bounds
                    unsafe { agg_window.update(start as usize, end as usize) }
                    agg_window.get_agg(idx)
                }
            })
        })
        .collect::<PolarsResult<PrimitiveArray<Out>>>()?;

    Ok(Box::new(out))
}

// Use an aggregation window that maintains the state
fn rolling_apply_agg_window<Agg, O, T, Out>(
    agg_window: &mut Agg,
    offsets: O,
    min_periods: usize,
    sorting_indices: &[IdxSize],
) -> PolarsResult<ArrayRef>
where
    Agg: RollingAggWindow<T, Out>,
    O: Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen,
    T: NativeType,
    Out: NativeType,
{
    let mut out = zeroed_vec(agg_window.slice_len());
    let mut validity: Option<MutableBitmap> = None;
    offsets.enumerate().try_for_each(|(idx, result)| {
        let (start, len) = result?;
        let end = start + len;
        let out_idx = unsafe { sorting_indices.get_unchecked(idx) };

        // On the Python side, if `min_periods` wasn't specified, it is set to
        // `1`. In that case, this condition is the same as checking
        // `if start == end`.
        if len >= (min_periods as IdxSize) {
            // SAFETY:
            // we are in bound
            unsafe { agg_window.update(start as usize, end as usize) };
            let res = agg_window.get_agg(*out_idx as usize);

            if let Some(res) = res {
                // SAFETY: `idx` is in bounds because `sorting_indices` was just taken from
                // `by`, which has already been checked to be the same length as the values.
                unsafe { *out.get_unchecked_mut(*out_idx as usize) = res };
            } else {
                instantiate_bitmap_if_null_and_set_false_at_idx(
                    &mut validity,
                    agg_window.slice_len(),
                    *out_idx as usize,
                )
            }
        } else {
            instantiate_bitmap_if_null_and_set_false_at_idx(
                &mut validity,
                agg_window.slice_len(),
                *out_idx as usize,
            )
        }
        Ok::<(), PolarsError>(())
    })?;

    let out = PrimitiveArray::<Out>::from_vec(out).with_validity(validity.map(|x| x.into()));

    Ok(Box::new(out))
}

// Instantiate a bitmap when the first null value is encountered.
// Set the validity at index `idx` to `false`.
fn instantiate_bitmap_if_null_and_set_false_at_idx(
    validity: &mut Option<MutableBitmap>,
    len: usize,
    idx: usize,
) {
    let bitmap = validity.get_or_insert_with(|| {
        let mut bitmap = MutableBitmap::with_capacity(len);
        bitmap.extend_constant(len, true);
        bitmap
    });
    bitmap.set(idx, false);
}
