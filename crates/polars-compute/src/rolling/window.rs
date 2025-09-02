use ::skiplist::OrderedSkipList;
use polars_utils::total_ord::TotalOrd;

use super::*;

pub(super) struct SortedBuf<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: OrderedSkipList<T>,
}

impl<'a, T: NativeType + PartialOrd + Copy> SortedBuf<'a, T> {
    pub(super) fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        max_window_size: Option<usize>,
    ) -> Self {
        let mut buf = if let Some(max_window_size) = max_window_size {
            OrderedSkipList::with_capacity(max_window_size)
        } else {
            OrderedSkipList::new()
        };
        unsafe { buf.sort_by(TotalOrd::tot_cmp) };
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
        self.buf[index]
    }

    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }
    // Note: range is not inclusive
    pub(super) fn index_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> skiplist::ordered_skiplist::Iter<'_, T> {
        self.buf.index_range(range)
    }
}

pub(super) struct SortedBufNulls<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    validity: &'a Bitmap,
    last_start: usize,
    last_end: usize,
    // non-null values within the window that we keep sorted
    buf: OrderedSkipList<T>,
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

    pub unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        max_window_size: Option<usize>,
    ) -> Self {
        let mut buf = if let Some(max_window_size) = max_window_size {
            OrderedSkipList::with_capacity(max_window_size)
        } else {
            OrderedSkipList::new()
        };
        unsafe { buf.sort_by(TotalOrd::tot_cmp) };

        // sort_opt_buf(&mut buf);
        let mut out = Self {
            slice,
            validity,
            last_start: start,
            last_end: end,
            buf,
            null_count: 0,
        };
        unsafe { out.fill_and_sort_buf(start, end) };
        out
    }

    /// Update the window position by setting the `start` index and the `end` index.
    ///
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    pub unsafe fn update(&mut self, start: usize, end: usize) -> usize {
        // Swap the whole buffer.
        if start >= self.last_end {
            unsafe { self.fill_and_sort_buf(start, end) };
        } else {
            // Vemove elements that should leave the window.
            for idx in self.last_start..start {
                // SAFETY: we are in bounds.
                if unsafe { self.validity.get_bit_unchecked(idx) } {
                    self.buf.remove(unsafe { self.slice.get_unchecked(idx) });
                } else {
                    self.null_count -= 1;
                }
            }

            // Insert elements that enter the window, but insert them sorted.
            for idx in self.last_end..end {
                // SAFETY: we are in bounds.
                if unsafe { self.validity.get_bit_unchecked(idx) } {
                    self.buf.insert(unsafe { *self.slice.get_unchecked(idx) });
                } else {
                    self.null_count += 1;
                }
            }
        }

        self.last_start = start;
        self.last_end = end;
        self.null_count
    }

    pub fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }

    pub fn len(&self) -> usize {
        self.null_count + self.buf.len()
    }

    pub fn get(&self, idx: usize) -> Option<T> {
        if idx >= self.null_count {
            Some(self.buf[idx - self.null_count])
        } else {
            None
        }
    }

    // Note: range is not inclusive
    pub fn index_range(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = Option<T>> {
        let nonnull_range =
            range.start.saturating_sub(self.null_count)..range.end.saturating_sub(self.null_count);
        (0..range.len() - nonnull_range.len())
            .map(|_| None)
            .chain(self.buf.index_range(nonnull_range).map(|x| Some(*x)))
    }
}
