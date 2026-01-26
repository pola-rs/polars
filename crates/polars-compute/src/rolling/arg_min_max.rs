use std::collections::VecDeque;
use std::marker::PhantomData;

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::RollingFnParams;
use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;

// Algorithm: https://cs.stackexchange.com/questions/120915/interview-question-with-arrays-and-consecutive-subintervals/120936#120936
// Modified to return the argmin/argmax instead of the value:
pub struct ArgMinMaxWindow<'a, T, P> {
    pub(crate) values: &'a [T],
    validity: Option<&'a Bitmap>,
    // values[monotonic_idxs[i]] is better than values[monotonic_idxs[i+1]] for
    // all i, as per the policy.
    monotonic_idxs: VecDeque<usize>,
    nonnulls_in_window: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    policy: PhantomData<P>,
}

impl<T: NativeType, P: MinMaxPolicy> ArgMinMaxWindow<'_, T, P> {
    /// # Safety
    /// The index must be in-bounds.
    unsafe fn insert_nonnull_value(&mut self, idx: usize) {
        unsafe {
            let value = self.values.get_unchecked(idx);

            // Remove values which are older and worse.
            while let Some(&tail_idx) = self.monotonic_idxs.back() {
                let tail_value = self.values.get_unchecked(tail_idx);
                if !P::is_better(value, tail_value) {
                    break;
                }
                self.monotonic_idxs.pop_back();
            }

            self.monotonic_idxs.push_back(idx);
            self.nonnulls_in_window += 1;
        }
    }

    fn remove_old_values(&mut self, window_start: usize) {
        // Remove values which have fallen outside the window start.
        while let Some(&head_idx) = self.monotonic_idxs.front() {
            if head_idx >= window_start {
                break;
            }
            self.monotonic_idxs.pop_front();
        }
    }
}

impl<T: NativeType, P: MinMaxPolicy> RollingAggWindowNulls<T, IdxSize>
    for ArgMinMaxWindow<'_, T, P>
{
    type This<'a> = ArgMinMaxWindow<'a, T, P>;

    fn new<'a>(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(params.is_none());
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let mut this = ArgMinMaxWindow {
            values: slice,
            validity: Some(validity),
            monotonic_idxs: VecDeque::new(),
            nonnulls_in_window: 0,
            start: 0,
            end: 0,
            policy: PhantomData,
        };
        // SAFETY: We bounds checked `start` and `end`.
        unsafe { RollingAggWindowNulls::update(&mut this, start, end) };
        this
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe {
            let v = self.validity.unwrap_unchecked();
            self.remove_old_values(new_start);
            for i in self.start..new_start.min(self.end) {
                self.nonnulls_in_window -= v.get_bit_unchecked(i) as usize;
            }
            for i in new_start.max(self.end)..new_end {
                if v.get_bit_unchecked(i) {
                    self.insert_nonnull_value(i);
                }
            }
        };
        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<IdxSize> {
        self.monotonic_idxs
            .front()
            .map(|&best_abs| (best_abs - self.start) as IdxSize)
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.nonnulls_in_window >= min_periods
    }

    fn slice_len(&self) -> usize {
        self.values.len()
    }
}

impl<T: NativeType, P: MinMaxPolicy> RollingAggWindowNoNulls<T, IdxSize>
    for ArgMinMaxWindow<'_, T, P>
{
    type This<'a> = ArgMinMaxWindow<'a, T, P>;

    fn new<'a>(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(params.is_none());
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let mut this = ArgMinMaxWindow {
            values: slice,
            validity: None,
            monotonic_idxs: VecDeque::new(),
            nonnulls_in_window: 0,
            start: 0,
            end: 0,
            policy: PhantomData,
        };

        // SAFETY: We bounds checked `start` and `end`.
        unsafe { RollingAggWindowNoNulls::update(&mut this, start, end) };
        this
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe {
            self.remove_old_values(new_start);

            for i in new_start.max(self.end)..new_end {
                self.insert_nonnull_value(i);
            }
        };
        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<IdxSize> {
        self.monotonic_idxs
            .front()
            .map(|&best_abs| (best_abs - self.start) as IdxSize)
    }

    fn slice_len(&self) -> usize {
        self.values.len()
    }
}

pub type ArgMinWindow<'a, T> = ArgMinMaxWindow<'a, T, MinPropagateNan>;
pub type ArgMaxWindow<'a, T> = ArgMinMaxWindow<'a, T, MaxPropagateNan>;
