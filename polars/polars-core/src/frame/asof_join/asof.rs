use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Sub;

use num_traits::Bounded;
use polars_arrow::index::IdxSize;

// We 
//It turns out to be clearer to iterate over elements of R and find their preimage. The direction of
//iteration depends on whether we are rolling forward or backward.


#[inline]
unsafe fn find_next_le<T: PartialOrd + Copy + Debug>(slice: &[T], val: &T) -> usize {
    // This is a tradeoff between a linear and a binary search. Linear is wasteful if we have long
    // runs of values in the left index mapping to a single value in the right, but binary search is
    // slower if a right value only maps to a few on the left.
    let mut last_idx = 0;
    let mut idx = 1;
    while slice.get_unchecked(idx - 1) <= val && idx < slice.len() {
        last_idx = idx;
        idx = std::cmp::min(2 * idx, slice.len());
    }
    last_idx + slice[last_idx..idx].partition_point(|x| x <= val)
}

pub(super) fn join_asof_forward<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
        let mut out = Vec::with_capacity(left.len());
        let mut slice = left;
    
        for (i, &val_r) in right.iter().enumerate() {
            if slice.is_empty() {
                break;
            } else if val_r < slice[0] {
                // Skip repeated values or values in right too small to match
                continue;
            }
            let j = unsafe { find_next_le(slice, &val_r) };
            out.extend(std::iter::repeat(Some(i as IdxSize)).take(j));
            slice = &slice[j..];
        }
    
        out.extend(std::iter::repeat(None).take(left.len() - out.len()));
        out
}

pub(super) fn join_asof_forward_with_tolerance<T: PartialOrd + Copy + Debug + Sub<Output = T>>(
    left: &[T],
    right: &[T],
    tolerance: T,
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());
    let mut slice = left;

    for (i, &val_r) in right.iter().enumerate() {
        if slice.is_empty() {
            break;
        } else if val_r < slice[0] {
            continue;
        }
        let j = unsafe { find_next_le(slice, &val_r) };
        // How many preceding values are NOT within the tolerance.
        let k = slice[..j].partition_point(|&x| x < val_r - tolerance);
        out.extend(std::iter::repeat(None).take(k));
        out.extend(std::iter::repeat(Some(i as IdxSize)).take(j - k));
        slice = &slice[j..];
    }

    out.extend(std::iter::repeat(None).take(left.len() - out.len()));
    out
}

#[inline]
unsafe fn find_prev_gt<T: PartialOrd + Copy + Debug>(slice: &[T], val: &T) -> usize {
    // See `find_next_le` This is basically the same but in reverse for rolling backward
    let mut last_idx = slice.len();
    let mut nback = 1;
    let mut idx = last_idx - nback;
    while slice.get_unchecked(idx) >= val && idx > 0 {
        last_idx = idx;
        idx = idx.saturating_sub(nback);
        nback *= 2;
    }
    idx + slice[idx..last_idx].partition_point(|x| x < val)
}

pub(super) fn join_asof_backward<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    if left.is_empty() {
        return vec![];
    }
    let mut out = VecDeque::with_capacity(left.len());
    let mut slice = left;

    // We go in reverse because we are looking for values in left >= val_r, they're both sorted
    // ascending, and when we have duplicate values in right, we match to the LAST one.
    for (i, &val_r) in right.iter().enumerate().rev() {
        if slice.is_empty() {
            break;
        } else if &val_r > slice.last().unwrap() {
            // Skip repeated values or values in right too large to match
            continue;
        }
        let j = unsafe { find_prev_gt(slice, &val_r) };
        for _ in j..slice.len() {
            out.push_front(Some(i as IdxSize));
        }
        slice = &slice[..j];
    }
    
    for _ in out.len()..left.len() {
        out.push_front(None);
    }
    out.into()
}

pub(super) fn join_asof_backward_with_tolerance<T: PartialOrd + Copy + Debug + Sub<Output = T>>(
    left: &[T],
    right: &[T],
    tolerance: T,
) -> Vec<Option<IdxSize>> {
    if left.is_empty() {
        return vec![];
    }
    let mut out = VecDeque::with_capacity(left.len());
    let mut slice = left;
    
    for (i, &val_r) in right.iter().enumerate().rev() {
        if slice.is_empty() {
            break;
        } else if &val_r > slice.last().unwrap() {
            continue;
        }
        let j = unsafe { find_prev_gt(slice, &val_r) };
        // How many following value ARE within the tolerance
        let k = slice[j..].partition_point(|&x| tolerance >= x - val_r);
        // Since left is sorted, farther out values are the ones outisde of the tolerance.
        for _ in (j + k)..slice.len() {
            out.push_front(None);
        }
        for _ in 0..k {
            out.push_front(Some(i as IdxSize));
        }
        slice = &slice[..j];
    }
    
    for _ in out.len()..left.len() {
        out.push_front(None);
    }
    out.into()
}


pub(super) fn join_asof_nearest<T: PartialOrd + Copy + Debug + Sub<Output = T> + Bounded>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());
    let mut offset = 0 as IdxSize;
    let max_value = <T as num_traits::Bounded>::max_value();
    let mut dist: T = max_value;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    // This is (val_r - val_l).abs(), but works on strings/dates
                    let dist_curr = if val_r > val_l {
                        val_r - val_l
                    } else {
                        val_l - val_r
                    };
                    if dist_curr <= dist {
                        // candidate for match
                        dist = dist_curr;
                        offset += 1;
                    } else {
                        // distance has increased, we're now farther away, so previous element was closest
                        out.push(Some(offset - 1));

                        // reset distance
                        dist = max_value;

                        // The next left-item may match on the same item, so we need to rewind the offset
                        offset -= 1;
                        break;
                    }
                }

                None => {
                    if offset > 1 {
                        // we've reached the end with no matches, so the last item is the nearest for all remaining
                        out.extend(
                            std::iter::repeat(Some(offset - 1)).take(left.len() - out.len()),
                        );
                    } else {
                        // this is only hit when the right frame is empty
                        out.extend(std::iter::repeat(None).take(left.len() - out.len()));
                    }
                    return out;
                }
            }
        }
    }

    out
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_asof_backward() {
        let a = [-1, 2, 3, 3, 3, 4];
        let b = [1, 2, 3, 3];

        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples, &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]);

        let b = [1, 2, 4, 5];
        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples, &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]);

        let a = [2, 4, 4, 4];
        let b = [1, 2, 3, 3];
        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples, &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_backward_tolerance() {
        let a = [-1, 20, 25, 30, 30, 40];
        let b = [10, 20, 30, 30];
        let tuples = join_asof_backward_with_tolerance(&a, &b, 4);
        assert_eq!(tuples, &[None, Some(1), None, Some(3), Some(3), None]);
    }

    #[test]
    fn test_asof_forward_tolerance() {
        let a = [-1, 20, 25, 30, 30, 40, 52];
        let b = [10, 20, 33, 55];
        let tuples = join_asof_forward_with_tolerance(&a, &b, 4);
        assert_eq!(
            tuples,
            &[None, Some(1), None, Some(2), Some(2), None, Some(3)]
        );
    }

    #[test]
    fn test_asof_forward() {
        let a = [-1, 1, 2, 4, 6];
        let b = [1, 2, 4, 5];

        let tuples = join_asof_forward(&a, &b);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples, &[Some(0), Some(0), Some(1), Some(2), None]);
    }
}
