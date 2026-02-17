use polars_utils::order_statistic_tree::OrderStatisticTree;
use polars_utils::total_ord::TotalOrd;

use super::*;

pub(super) struct SortedBuf<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: OrderStatisticTree<T>,
}

impl<'a, T: NativeType + PartialOrd + Copy> SortedBuf<'a, T> {
    pub(super) fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        max_window_size: Option<usize>,
    ) -> Self {
        let buf = if let Some(max_window_size) = max_window_size {
            OrderStatisticTree::with_capacity(max_window_size, TotalOrd::tot_cmp)
        } else {
            OrderStatisticTree::new(TotalOrd::tot_cmp)
        };
        let mut out = Self {
            slice,
            last_start: start,
            last_end: end,
            buf,
        };
        let init = &slice[start..end];
        out.reset(init);
        out
    }

    fn reset(&mut self, slice: &[T]) {
        self.buf.clear();
        self.buf.extend(slice.iter().copied());
    }

    /// Update the window position by setting the `start` index and the `end` index.
    ///
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    ///
    pub(super) unsafe fn update(&mut self, start: usize, end: usize) {
        // swap the whole buffer
        if start >= self.last_end {
            self.buf.clear();
            let new_window = unsafe { self.slice.get_unchecked(start..end) };
            self.reset(new_window);
        } else {
            // remove elements that should leave the window
            for idx in self.last_start..start {
                // SAFETY:
                // in bounds
                let val = unsafe { self.slice.get_unchecked(idx) };
                self.buf.remove(val);
            }

            // insert elements that enter the window, but insert them sorted
            for idx in self.last_end..end {
                // SAFETY:
                // we are in bounds
                let val = unsafe { *self.slice.get_unchecked(idx) };
                self.buf.insert(val);
            }
        }
        self.last_start = start;
        self.last_end = end;
    }

    pub(super) fn get(&self, index: usize) -> T {
        self.buf.get(index).copied().unwrap()
    }

    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }

    pub(super) fn slice_len(&self) -> usize {
        self.slice.len()
    }
}

pub(super) struct SortedBufNulls<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    validity: &'a Bitmap,
    start: usize,
    end: usize,
    // non-null values within the window that we keep sorted
    buf: OrderStatisticTree<T>,
    pub null_count: usize,
}

impl<'a, T: NativeType + PartialOrd> SortedBufNulls<'a, T> {
    unsafe fn fill_and_sort_buf(&mut self, start: usize, end: usize) {
        self.null_count = 0;
        let iter = (start..end).flat_map(|idx| unsafe {
            if self.validity.get_bit_unchecked(idx) {
                Some(*self.slice.get_unchecked(idx))
            } else {
                self.null_count += 1;
                None
            }
        });

        self.buf.clear();
        self.buf.extend(iter);
    }

    pub fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        max_window_size: Option<usize>,
    ) -> Self {
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let buf = if let Some(max_window_size) = max_window_size {
            OrderStatisticTree::with_capacity(max_window_size, TotalOrd::tot_cmp)
        } else {
            OrderStatisticTree::new(TotalOrd::tot_cmp)
        };

        // sort_opt_buf(&mut buf);
        let mut out = Self {
            slice,
            validity,
            start,
            end,
            buf,
            null_count: 0,
        };
        // SAFETY: We bounds checked `start` and `end`.
        unsafe { out.fill_and_sort_buf(start, end) };
        out
    }

    /// Update the window position by setting the `start` index and the `end` index.
    ///
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    pub unsafe fn update(&mut self, new_start: usize, new_end: usize) -> usize {
        // Swap the whole buffer.
        if new_start >= self.end {
            unsafe { self.fill_and_sort_buf(new_start, new_end) };
        } else {
            // Remove elements that should leave the window.
            for idx in self.start..new_start {
                // SAFETY: we are in bounds.
                if unsafe { self.validity.get_bit_unchecked(idx) } {
                    self.buf.remove(unsafe { self.slice.get_unchecked(idx) });
                } else {
                    self.null_count -= 1;
                }
            }

            // Insert elements that enter the window, but insert them sorted.
            for idx in self.end..new_end {
                // SAFETY: we are in bounds.
                if unsafe { self.validity.get_bit_unchecked(idx) } {
                    self.buf.insert(unsafe { *self.slice.get_unchecked(idx) });
                } else {
                    self.null_count += 1;
                }
            }
        }

        self.start = new_start;
        self.end = new_end;
        self.null_count
    }

    pub fn is_valid(&self, min_periods: usize) -> bool {
        ((self.end - self.start) - self.null_count) >= min_periods
    }

    pub fn len(&self) -> usize {
        self.null_count + self.buf.len()
    }

    pub fn slice_len(&self) -> usize {
        self.slice.len()
    }

    pub fn get(&self, idx: usize) -> Option<T> {
        if idx >= self.null_count {
            Some(self.buf.get(idx - self.null_count).copied().unwrap())
        } else {
            None
        }
    }
}
