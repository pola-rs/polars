use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};

/// Find the index of the first element of `arr` that is greater
/// or equal to `val`.
/// Assumes that `arr` is sorted.
pub fn find_first_ge_index<T>(arr: &[T], val: T) -> usize
where
    T: Ord,
{
    match arr.binary_search(&val) {
        Ok(x) => x,
        Err(x) => x,
    }
}

/// Find the index of the first element of `arr` that is greater
/// than `val`.
/// Assumes that `arr` is sorted.
pub fn find_first_gt_index<T>(arr: &[T], val: T) -> usize
where
    T: Ord,
{
    match arr.binary_search(&val) {
        Ok(x) => x + 1,
        Err(x) => x,
    }
}

// https://en.wikipedia.org/wiki/Exponential_search
// Use if you expect matches to be close by. Otherwise use binary search.
pub trait ExponentialSearch<T> {
    fn exponential_search_by<F>(&self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering;

    fn partition_point_exponential<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        self.exponential_search_by(|x| if pred(x) { Less } else { Greater })
            .unwrap_or_else(|i| i)
    }
}

impl<T: std::fmt::Debug> ExponentialSearch<T> for &[T] {
    fn exponential_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        if self.is_empty() {
            return Err(0);
        }

        let mut bound = 1;

        while bound < self.len() {
            // SAFETY
            // Bound is always >=0 and < len.
            let cmp = f(unsafe { self.get_unchecked(bound) });

            if cmp == Greater {
                break;
            }
            bound *= 2
        }
        let end_bound = std::cmp::min(self.len(), bound);
        // SAFETY:
        // We checked the end bound and previous bound was within slice as per the `while` condition.
        let prev_bound = bound / 2;

        let slice = unsafe { self.get_unchecked(prev_bound..end_bound) };

        match slice.binary_search_by(f) {
            Ok(i) => Ok(i + prev_bound),
            Err(i) => Err(i + prev_bound),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_partition_point() {
        let v = [1, 2, 3, 3, 5, 6, 7];
        let i = v.as_slice().partition_point_exponential(|&x| x < 5);
        assert_eq!(i, 4);
    }
}
