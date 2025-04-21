use ::skiplist::OrderedSkipList;
use polars_utils::total_ord::TotalOrd;

use super::*;

pub(super) struct SortedBuf<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    pub buf: OrderedSkipList<T>,
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
}

pub(super) struct SortedBufNulls<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    validity: &'a Bitmap,
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: OrderedSkipList<Option<T>>,
    pub null_count: usize,
}

impl<'a, T: NativeType + PartialOrd> SortedBufNulls<'a, T> {
    unsafe fn fill_and_sort_buf(&mut self, start: usize, end: usize) {
        self.null_count = 0;
        let iter = (start..end).map(|idx| unsafe {
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

    pub(super) unsafe fn new(
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
    ///
    pub(super) unsafe fn update(&mut self, start: usize, end: usize) -> usize {
        // swap the whole buffer
        if start >= self.last_end {
            unsafe { self.fill_and_sort_buf(start, end) };
        } else {
            // remove elements that should leave the window
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let val = if unsafe { self.validity.get_bit_unchecked(idx) } {
                    unsafe { Some(*self.slice.get_unchecked(idx)) }
                } else {
                    self.null_count -= 1;
                    None
                };
                self.buf.remove(&val);
            }

            // insert elements that enter the window, but insert them sorted
            for idx in self.last_end..end {
                // SAFETY:
                // we are in bounds
                let val = if unsafe { self.validity.get_bit_unchecked(idx) } {
                    unsafe { Some(*self.slice.get_unchecked(idx)) }
                } else {
                    self.null_count += 1;
                    None
                };

                self.buf.insert(val);
            }
        }
        self.last_start = start;
        self.last_end = end;
        self.null_count
    }

    pub(super) fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }

    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }

    pub(super) fn get(&self, idx: usize) -> Option<T> {
        self.buf[idx]
    }
}
