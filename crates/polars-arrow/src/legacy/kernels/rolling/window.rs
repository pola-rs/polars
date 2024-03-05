use polars_utils::total_ord::TotalOrd;

use super::*;

pub(super) struct SortedBuf<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: Vec<T>,
}

impl<'a, T: NativeType> SortedBuf<'a, T> {
    pub(super) fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let mut buf = slice[start..end].to_vec();
        buf.sort_by(TotalOrd::tot_cmp);
        Self {
            slice,
            last_start: start,
            last_end: end,
            buf,
        }
    }

    /// Update the window position by setting the `start` index and the `end` index.
    ///
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    ///
    pub(super) unsafe fn update(&mut self, start: usize, end: usize) -> &[T] {
        // swap the whole buffer
        if start >= self.last_end {
            self.buf.clear();
            let new_window = self.slice.get_unchecked(start..end);
            self.buf.extend_from_slice(new_window);
            self.buf.sort_by(TotalOrd::tot_cmp);
        } else {
            // remove elements that should leave the window
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let val = self.slice.get_unchecked(idx);
                // SAFETY:
                // value is present in buf
                let remove_idx = self
                    .buf
                    .binary_search_by(|a| a.tot_cmp(val))
                    .unwrap_unchecked();
                // this is O(n) but we need a sorted window
                self.buf.remove(remove_idx);
            }

            // insert elements that enter the window, but insert them sorted
            for idx in self.last_end..end {
                // SAFETY:
                // we are in bounds
                let val = *self.slice.get_unchecked(idx);
                let insertion_idx = self
                    .buf
                    .binary_search_by(|a| a.tot_cmp(&val))
                    .unwrap_or_else(|insertion_idx| insertion_idx);

                // this is O(n) but we need a sorted window
                self.buf.insert(insertion_idx, val);
            }
        }
        self.last_start = start;
        self.last_end = end;
        &self.buf
    }
}

pub(super) struct SortedBufNulls<'a, T: NativeType> {
    // slice over which the window slides
    slice: &'a [T],
    validity: &'a Bitmap,
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: Vec<Option<T>>,
    pub null_count: usize,
}

impl<'a, T: NativeType> SortedBufNulls<'a, T> {
    unsafe fn fill_and_sort_buf(&mut self, start: usize, end: usize) {
        self.null_count = 0;
        let iter = (start..end).map(|idx| {
            if self.validity.get_bit_unchecked(idx) {
                Some(*self.slice.get_unchecked(idx))
            } else {
                self.null_count += 1;
                None
            }
        });

        self.buf.clear();
        self.buf.extend(iter);
        self.buf.sort_by(TotalOrd::tot_cmp);
    }

    pub(super) unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
    ) -> Self {
        let buf = Vec::with_capacity(end - start);

        // sort_opt_buf(&mut buf);
        let mut out = Self {
            slice,
            validity,
            last_start: start,
            last_end: end,
            buf,
            null_count: 0,
        };
        out.fill_and_sort_buf(start, end);
        out
    }

    /// Update the window position by setting the `start` index and the `end` index.
    ///
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    ///
    pub(super) unsafe fn update(&mut self, start: usize, end: usize) -> (&[Option<T>], usize) {
        // swap the whole buffer
        if start >= self.last_end {
            self.fill_and_sort_buf(start, end);
        } else {
            // remove elements that should leave the window
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let val = if self.validity.get_bit_unchecked(idx) {
                    Some(*self.slice.get_unchecked(idx))
                } else {
                    self.null_count -= 1;
                    None
                };

                // SAFETY:
                // value is present in buf
                let remove_idx = self
                    .buf
                    .binary_search_by(|a| a.tot_cmp(&val))
                    .unwrap_unchecked();
                // this is O(n) but we need a sorted window
                self.buf.remove(remove_idx);
            }

            // insert elements that enter the window, but insert them sorted
            for idx in self.last_end..end {
                // SAFETY:
                // we are in bounds
                let val = if self.validity.get_bit_unchecked(idx) {
                    Some(*self.slice.get_unchecked(idx))
                } else {
                    self.null_count += 1;
                    None
                };
                let insertion_idx = self
                    .buf
                    .binary_search_by(|a| a.tot_cmp(&val))
                    .unwrap_or_else(|insertion_idx| insertion_idx);

                // this is O(n) but we need a sorted window
                self.buf.insert(insertion_idx, val);
            }
        }
        self.last_start = start;
        self.last_end = end;
        (&self.buf, self.null_count)
    }

    pub(super) fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sorted_buf() {
        unsafe {
            let values = &[1, 3, 4, 6, 2, -1, 9];

            let mut sorted_window = SortedBuf::new(values, 0, 3);
            let window = sorted_window.update(1, 4);
            assert_eq!(window, &[3, 4, 6]);
            let window = sorted_window.update(2, 5);
            assert_eq!(window, &[2, 4, 6]);
            let window = sorted_window.update(3, 6);
            assert_eq!(window, &[-1, 2, 6]);
            let window = sorted_window.update(3, 7);
            assert_eq!(window, &[-1, 2, 6, 9]);
            let window = sorted_window.update(4, 7);
            assert_eq!(window, &[-1, 2, 9]);
        }
    }
}
