use std::collections::VecDeque;

use crate::parquet::indexes::Interval;

/// An iterator adapter that converts an iterator over items into an iterator over slices of
/// those N items.
///
/// This iterator is best used with iterators that implement `nth` since skipping items
/// allows this iterator to skip sequences of items without having to call each of them.
#[derive(Debug, Clone)]
pub struct SliceFilteredIter<I> {
    pub(crate) iter: I,
    selected_rows: VecDeque<Interval>,
    current_remaining: usize,
    current: usize, // position in the slice
    total_length: usize,
}

impl<I> SliceFilteredIter<I> {
    /// Return a new [`SliceFilteredIter`]
    pub fn new(iter: I, selected_rows: VecDeque<Interval>) -> Self {
        let total_length = selected_rows.iter().map(|i| i.length).sum();
        Self {
            iter,
            selected_rows,
            current_remaining: 0,
            current: 0,
            total_length,
        }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for SliceFilteredIter<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_remaining == 0 {
            if let Some(interval) = self.selected_rows.pop_front() {
                // skip the hole between the previous start and this start
                // (start + length) - start
                let item = self.iter.nth(interval.start - self.current);
                self.current = interval.start + interval.length;
                self.current_remaining = interval.length - 1;
                self.total_length -= 1;
                item
            } else {
                None
            }
        } else {
            self.current_remaining -= 1;
            self.total_length -= 1;
            self.iter.next()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_length, Some(self.total_length))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        let iter = 0..=100;

        let intervals = vec![
            Interval::new(0, 2),
            Interval::new(20, 11),
            Interval::new(31, 1),
        ];

        let a: VecDeque<Interval> = intervals.clone().into_iter().collect();
        let mut a = SliceFilteredIter::new(iter, a);

        let expected: Vec<usize> = intervals
            .into_iter()
            .flat_map(|interval| interval.start..(interval.start + interval.length))
            .collect();

        assert_eq!(expected, a.by_ref().collect::<Vec<_>>());
        assert_eq!((0, Some(0)), a.size_hint());
    }
}
