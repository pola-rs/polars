use std::collections::VecDeque;
use std::marker::PhantomData;

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::min_max::MinMaxPolicy;

use super::RollingFnParams;
use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;

// Algorithm: https://cs.stackexchange.com/questions/120915/interview-question-with-arrays-and-consecutive-subintervals/120936#120936
pub struct MinMaxWindow<'a, T, P> {
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    // values[monotonic_idxs[i]] is better than values[monotonic_idxs[i+1]] for
    // all i, as per the policy.
    monotonic_idxs: VecDeque<usize>,
    nonnulls_in_window: usize,
    last_end: usize,
    policy: PhantomData<P>,
}

impl<T: NativeType, P: MinMaxPolicy> MinMaxWindow<'_, T, P> {
    /// # Safety
    /// The index must be in-bounds.
    unsafe fn insert_nonnull_value(&mut self, idx: usize) {
        unsafe {
            let value = self.values.get_unchecked(idx);

            // Remove values which are older and worse.
            while let Some(tail_idx) = self.monotonic_idxs.back() {
                let tail_value = self.values.get_unchecked(*tail_idx);
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
        while let Some(head_idx) = self.monotonic_idxs.front() {
            if *head_idx >= window_start {
                break;
            }
            self.monotonic_idxs.pop_front();
            self.nonnulls_in_window -= 1;
        }
    }
}

impl<'a, T: NativeType, P: MinMaxPolicy> RollingAggWindowNulls<'a, T> for MinMaxWindow<'a, T, P> {
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self {
        assert!(params.is_none());
        let mut slf = Self {
            values: slice,
            validity: Some(validity),
            monotonic_idxs: VecDeque::new(),
            nonnulls_in_window: 0,
            last_end: 0,
            policy: PhantomData,
        };
        unsafe {
            RollingAggWindowNulls::update(&mut slf, start, end);
        }
        slf
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        unsafe {
            let v = self.validity.unwrap_unchecked();
            self.remove_old_values(start);
            for i in start.max(self.last_end)..end {
                if v.get_bit_unchecked(i) {
                    self.insert_nonnull_value(i);
                }
            }
            self.last_end = end;
            self.monotonic_idxs
                .front()
                .map(|idx| *self.values.get_unchecked(*idx))
        }
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.nonnulls_in_window >= min_periods
    }
}

impl<'a, T: NativeType, P: MinMaxPolicy> RollingAggWindowNoNulls<'a, T> for MinMaxWindow<'a, T, P> {
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self {
        assert!(params.is_none());
        let mut slf = Self {
            values: slice,
            validity: None,
            monotonic_idxs: VecDeque::new(),
            nonnulls_in_window: 0,
            last_end: 0,
            policy: PhantomData,
        };
        unsafe {
            RollingAggWindowNoNulls::update(&mut slf, start, end);
        }
        slf
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        unsafe {
            self.remove_old_values(start);
            for i in start.max(self.last_end)..end {
                self.insert_nonnull_value(i);
            }
            self.last_end = end;
            self.monotonic_idxs
                .front()
                .map(|idx| *self.values.get_unchecked(*idx))
        }
    }
}
