//! This module implements logic shared between nulls and no_nulls.

use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_arrow::bitmap::MutableBitmap;
use polars_arrow::types::NativeType;
use polars_compute::rolling::no_nulls::RollingAggWindowNoNulls;
use polars_compute::rolling::nulls::RollingAggWindowNulls;
use polars_core::prelude::*;

pub(crate) trait RollingAggWindow<T: NativeType, Out: NativeType> {
    /// # Safety
    /// `start` and `end` must be in bounds of `slice` and associated structures.
    unsafe fn update(&mut self, start: usize, end: usize);

    /// Get the aggregate of the current window relative to the value at `idx`.
    fn get_agg(&self, idx: usize) -> Option<Out>;

    fn is_valid(&self, min_samples: usize) -> bool;
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

    fn is_valid(&self, _min_samples: usize) -> bool {
        true
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

    fn is_valid(&self, min_samples: usize) -> bool {
        self.0.is_valid(min_samples)
    }
}

/// Aggregates each window `[start, len]`.
pub(crate) fn rolling_apply_agg<T, Out, Agg>(
    agg_window: &mut Agg,
    windows: &[[IdxSize; 2]],
    min_samples: usize,
) -> ArrayRef
where
    T: NativeType,
    Out: NativeType,
    Agg: RollingAggWindow<T, Out>,
{
    let len = windows.len();
    let mut out = Vec::with_capacity(len);
    let mut validity: Option<MutableBitmap> = None;
    for (idx, &[start, window_len]) in windows.iter().enumerate() {
        // On the Python side, if `min_samples` wasn't specified, it is set to
        // `1`. In that case, this condition is the same as checking
        // `if start == end`.
        let res = if window_len >= min_samples as IdxSize {
            // SAFETY: the windows are in bounds of the values.
            unsafe { agg_window.update(start as usize, (start + window_len) as usize) };
            // idx is the position of the current element in the window, because the values
            // were sorted by 'by' before.
            if agg_window.is_valid(min_samples) {
                agg_window.get_agg(idx)
            } else {
                None
            }
        } else {
            None
        };

        match res {
            Some(res) => out.push(res),
            None => {
                out.push(Out::default());
                instantiate_bitmap_if_null_and_set_false_at_idx(&mut validity, len, idx)
            },
        }
    }

    let out = PrimitiveArray::<Out>::from_vec(out).with_validity(validity.map(|x| x.into()));
    Box::new(out)
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
