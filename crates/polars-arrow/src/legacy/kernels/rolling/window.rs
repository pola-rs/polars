use super::*;

pub(super) struct SortedBuf<'a, T: NativeType + IsFloat + PartialOrd> {
    // slice over which the window slides
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: Vec<T>,
    scratch_out: Vec<T>,
    scratch_in: Vec<T>,
    scratch_tmp: Vec<T>,
}

impl<'a, T: NativeType + IsFloat + PartialOrd + Zero> SortedBuf<'a, T> {
    pub(super) fn new(slice: &'a [T], start: usize, end: usize) -> Self {
        let mut buf = slice[start..end].to_vec();
        sort_buf(&mut buf);
        let scratch_out = vec![T::zero(); buf.len()];
        let scratch_in = vec![T::zero(); buf.len()];
        let scratch_tmp = vec![T::zero(); buf.len()];
        Self {
            slice,
            last_start: start,
            last_end: end,
            buf,
            scratch_out,
            scratch_in,
            scratch_tmp,
        }
    }

    /// Update the window position by setting the `start` index and the `end` index.
    /// # Safety
    /// The caller must ensure that `start` and `end` are within bounds of `self.slice`
    ///
    pub(super) unsafe fn update(&mut self, start: usize, end: usize) -> &[T] {
        // swap the whole buffer
        if start >= self.last_end {
            // SAFETY: all types are copy.
            self.buf.set_len(0);
            let new_window = self.slice.get_unchecked(start..end);
            self.buf.extend_from_slice(new_window);
            sort_buf(&mut self.buf);
        } else {
            self.scratch_out.set_len(0);
            let values_out = self.slice.get_unchecked(self.last_start..start);
            self.scratch_out.extend_from_slice(values_out);
            sort_buf(&mut self.scratch_out);

            self.scratch_in.set_len(0);
            let values_in = self.slice.get_unchecked(self.last_end..end);
            self.scratch_in.extend_from_slice(values_in);
            sort_buf(&mut self.scratch_in);

            let mut iter_out = self.scratch_out.iter();
            let mut iter_in = self.scratch_in.iter();
            let mut iter_window = self.buf.iter();

            self.scratch_tmp.set_len(0);
            let mut out_value = iter_out.next();
            let mut in_value = iter_in.next();
            let mut window_value = iter_window.next();
            loop {
                match (in_value, window_value) {
                    (Some(i), Some(w)) => {
                        if let Some(o) = out_value {
                            // This value is leaving
                            if matches!(compare_fn(*o, *w), Ordering::Equal) {
                                out_value = iter_out.next();
                                window_value = iter_window.next();
                                continue;
                            }
                        }

                        // Choose the lowest
                        if matches!(compare_fn(*i, *w), Ordering::Less) {
                            self.scratch_tmp.push(*i);
                            in_value = iter_in.next();
                        } else {
                            self.scratch_tmp.push(*w);
                            window_value = iter_window.next();
                        }
                    },
                    (None, Some(w)) => {
                        match out_value {
                            Some(o) => {
                                // This value is leaving
                                if matches!(compare_fn(*o, *w), Ordering::Equal) {
                                    out_value = iter_out.next();
                                    window_value = iter_window.next();
                                    continue;
                                }
                            },
                            // Both input, and output depleted
                            // deplete the buffer.
                            None => {
                                // -1 because the iterator already took the next value, but we didn't push it yet.
                                let offset = self.buf.len() - iter_window.size_hint().0 - 1;
                                self.scratch_tmp.extend_from_slice(&self.buf[offset..]);
                                break;
                            },
                        }
                        self.scratch_tmp.push(*w);
                        window_value = iter_window.next();
                    },
                    (Some(i), None) => {
                        self.scratch_tmp.push(*i);
                        in_value = iter_in.next();
                        continue;
                    },
                    (None, None) => break,
                }
            }
            std::mem::swap(&mut self.scratch_tmp, &mut self.buf);
        }
        self.last_start = start;
        self.last_end = end;
        &self.buf
    }
}

pub(super) fn sort_opt_buf<T>(buf: &mut [Option<T>])
where
    T: IsFloat + NativeType + PartialOrd,
{
    if T::is_float() {
        buf.sort_by(|a, b| compare_opt_fn(*a, *b));
    } else {
        // Safety:
        // all integers are Ord
        unsafe { buf.sort_by(|a, b| a.partial_cmp(b).unwrap_unchecked()) };
    }
}

#[inline]
fn compare_fn<T>(a: T, b: T) -> Ordering
where
    T: PartialOrd + IsFloat + NativeType,
{
    if T::is_float() {
        match (a.is_nan(), b.is_nan()) {
            // safety: we checked nans
            (false, false) => unsafe { a.partial_cmp(&b).unwrap_unchecked() },
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
        }
    } else {
        // Safety:
        // all integers are Ord
        unsafe { a.partial_cmp(&b).unwrap_unchecked() }
    }
}

#[inline]
fn compare_opt_fn<T>(a: Option<T>, b: Option<T>) -> Ordering
where
    T: PartialOrd + IsFloat + NativeType,
{
    if T::is_float() {
        match (a, b) {
            (Some(a), Some(b)) => compare_fn(a, b),
            _ => a.partial_cmp(&b).unwrap(),
        }
    } else {
        // Safety:
        // all integers are Ord
        unsafe { a.partial_cmp(&b).unwrap_unchecked() }
    }
}

pub(super) struct SortedBufNulls<'a, T: NativeType + IsFloat + PartialOrd> {
    // slice over which the window slides
    slice: &'a [T],
    validity: &'a Bitmap,
    last_start: usize,
    last_end: usize,
    // values within the window that we keep sorted
    buf: Vec<Option<T>>,
    pub null_count: usize,
}

impl<'a, T: NativeType + IsFloat + PartialOrd> SortedBufNulls<'a, T> {
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
        sort_opt_buf(&mut self.buf);
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
                // safety
                // we are in bounds
                let val = if self.validity.get_bit_unchecked(idx) {
                    Some(*self.slice.get_unchecked(idx))
                } else {
                    self.null_count -= 1;
                    None
                };

                // safety
                // value is present in buf
                let remove_idx = self
                    .buf
                    .binary_search_by(|a| compare_opt_fn(*a, val))
                    .unwrap_unchecked();
                // this is O(n) but we need a sorted window
                self.buf.remove(remove_idx);
            }

            // insert elements that enter the window, but insert them sorted
            for idx in self.last_end..end {
                // safety
                // we are in bounds
                let val = if self.validity.get_bit_unchecked(idx) {
                    Some(*self.slice.get_unchecked(idx))
                } else {
                    self.null_count += 1;
                    None
                };
                let insertion_idx = self
                    .buf
                    .binary_search_by(|a| compare_opt_fn(*a, val))
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
