use super::*;

pub(super) struct SortedBuf<'a, T: NativeType + IsFloat + PartialOrd> {
    // slice over which the window slides
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: Vec<T>,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> SortedBuf<'a, T> {
    pub(super) fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let mut buf = slice[start..end].to_vec();
        sort_buf(&mut buf);
        Self {
            slice,
            last_start: start,
            last_end: end,
            buf,
        }
    }

    /// Update the window position by setting the `start` index and the `end` index.
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    ///
    pub(super) unsafe fn update(&mut self, start: usize, end: usize) -> &[T] {
        // remove elements that should leave the window
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let val = self.slice.get_unchecked(idx);
            // safety
            // value is present in buf
            let remove_idx = self
                .buf
                .binary_search_by(|a| compare_fn(a, val))
                .unwrap_unchecked();
            // this is O(n) but we need a sorted window
            self.buf.remove(remove_idx);
        }
        self.last_start = start;

        // insert elements that enter the window, but insert them sorted
        for idx in self.last_end..end {
            // safety
            // we are in bounds
            let val = *self.slice.get_unchecked(idx);
            let insertion_idx = self
                .buf
                .binary_search_by(|a| compare_fn(a, &val))
                .unwrap_or_else(|insertion_idx| insertion_idx);

            // this is O(n) but we need a sorted window
            self.buf.insert(insertion_idx, val);
        }
        self.last_end = end;
        &self.buf
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
