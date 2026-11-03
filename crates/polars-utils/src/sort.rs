use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::ops::Deref;

use num_traits::FromPrimitive;

use crate::itertools::Itertools;
use crate::nulls::IsNull;
use crate::total_ord::TotalOrd;

unsafe fn assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe { &mut *(slice as *mut [MaybeUninit<T>] as *mut [T]) }
}

pub fn arg_sort_ascending<'a, T: TotalOrd + Copy + 'a, Idx, I: IntoIterator<Item = T>>(
    v: I,
    scratch: &'a mut Vec<u8>,
    n: usize,
) -> &'a mut [Idx]
where
    Idx: FromPrimitive + Copy,
{
    let upper_bound = size_of::<(T, Idx)>() * n + align_of::<(T, Idx)>();
    scratch.reserve(upper_bound);
    let scratch_slice = unsafe {
        let cap_slice = scratch.spare_capacity_mut();
        let (_, scratch_slice, _) = cap_slice.align_to_mut::<MaybeUninit<(T, Idx)>>();
        &mut scratch_slice[..n]
    };

    for ((i, v), dst) in v.into_iter().enumerate().zip_eq(scratch_slice.iter_mut()) {
        *dst = MaybeUninit::new((v, Idx::from_usize(i).unwrap()));
    }

    let scratch_slice = unsafe { assume_init_mut(scratch_slice) };
    scratch_slice.sort_by(|key1, key2| key1.0.tot_cmp(&key2.0));

    // Now we write the indexes in the same array. So from (T, Idxsize) to just IdxSize.
    unsafe {
        let src = scratch_slice.as_ptr();
        let (_, scratch_slice_aligned_to_idx, _) = scratch_slice.align_to_mut::<Idx>();
        let dst = scratch_slice_aligned_to_idx.as_mut_ptr();

        for i in 0..n {
            dst.add(i).write((*src.add(i)).1);
        }

        &mut scratch_slice_aligned_to_idx[..n]
    }
}

/// Writes the value at each of the `k` strictly ascending `ranks` (all below `v.len()`) of `v`
/// sorted by `cmp` to `out`, reordering `v`, in O(n log k).
pub fn select_many_unstable_by<T, F>(v: &mut [T], ranks: &[usize], out: &mut [T], cmp: F)
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    check_ranks(v.len(), ranks, out);
    select_many_seq(v, ranks, 0, out, &cmp);
}

/// Writes the value at each of the ascending `ranks`, less `offset`, of `v` sorted by `cmp` to
/// `out`, reordering `v`, sequentially in O(n log k).
fn select_many_seq<T: Copy, F>(v: &mut [T], ranks: &[usize], offset: usize, out: &mut [T], cmp: &F)
where
    F: Fn(&T, &T) -> Ordering,
{
    // Sorting a slice this short costs about one selection and places every rank at once.
    const SORT_LEN: usize = 16;
    match ranks {
        [] => {},
        _ if v.len() <= SORT_LEN => {
            v.sort_unstable_by(|a, b| cmp(a, b));
            read_ranks(v, ranks, offset, out);
        },
        // A rank at either end needs only a scan, as for the upper of two adjacent ranks.
        [r] if r - offset == 0 => out[0] = extreme(v, |a, b| cmp(a, b).is_lt()),
        [r] if r - offset == v.len() - 1 => out[0] = extreme(v, |a, b| cmp(a, b).is_gt()),
        _ => {
            // Splitting on the middle rank bounds the recursion depth by log2(k).
            let mid = (ranks.len() - 1) / 2;
            let pivot = ranks[mid] - offset;
            // Passing `cmp` itself, a reference, makes std's selection about 20% slower on
            // integers.
            let (left, value, right) = v.select_nth_unstable_by(pivot, |a, b| cmp(a, b));
            let (out_left, rest) = out.split_at_mut(mid);
            rest[0] = *value;
            select_many_seq(left, &ranks[..mid], offset, out_left, cmp);
            let right_offset = offset + pivot + 1;
            select_many_seq(right, &ranks[mid + 1..], right_offset, &mut rest[1..], cmp);
        },
    }
}

/// A value of the non-empty `v` that no other value `beats`.
fn extreme<T: Copy>(v: &[T], beats: impl Fn(&T, &T) -> bool) -> T {
    // Independent running extremes keep each comparison from waiting on the one before. More than
    // four keeps 64-bit integers from vectorising on AVX2, which makes the scan 5x slower.
    const LANES: usize = 4;
    let mut best = [v[0]; LANES];
    let lanes = v.chunks_exact(LANES);
    let rest = lanes.remainder();
    for chunk in lanes {
        for (b, x) in best.iter_mut().zip(chunk) {
            if beats(x, b) {
                *b = *x;
            }
        }
    }
    best[1..]
        .iter()
        .chain(rest)
        .fold(best[0], |b, x| if beats(x, &b) { *x } else { b })
}

fn check_ranks<T>(len: usize, ranks: &[usize], out: &[T]) {
    assert!(ranks.is_sorted_by(|a, b| a < b));
    assert!(ranks.last().is_none_or(|&r| r < len));
    assert_eq!(ranks.len(), out.len());
}

fn read_ranks<T: Copy>(v: &[T], ranks: &[usize], offset: usize, out: &mut [T]) {
    for (o, &r) in out.iter_mut().zip(ranks) {
        *o = v[r - offset];
    }
}

#[derive(PartialEq, Eq, Clone, Hash)]
#[repr(transparent)]
pub struct ReorderWithNulls<T, const DESCENDING: bool, const NULLS_LAST: bool>(pub Option<T>);

impl<T, const DESCENDING: bool, const NULLS_LAST: bool>
    ReorderWithNulls<T, DESCENDING, NULLS_LAST>
{
    pub fn as_deref(&self) -> ReorderWithNulls<&<T as Deref>::Target, DESCENDING, NULLS_LAST>
    where
        T: Deref,
    {
        let x = self.0.as_deref();
        ReorderWithNulls(x)
    }
}

impl<T: PartialOrd, const DESCENDING: bool, const NULLS_LAST: bool> PartialOrd
    for ReorderWithNulls<T, DESCENDING, NULLS_LAST>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (&self.0, &other.0) {
            (None, None) => Some(Ordering::Equal),
            (None, Some(_)) => {
                if NULLS_LAST {
                    Some(Ordering::Greater)
                } else {
                    Some(Ordering::Less)
                }
            },
            (Some(_), None) => {
                if NULLS_LAST {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            },
            (Some(l), Some(r)) => {
                if DESCENDING {
                    r.partial_cmp(l)
                } else {
                    l.partial_cmp(r)
                }
            },
        }
    }
}

impl<T: Ord, const DESCENDING: bool, const NULLS_LAST: bool> Ord
    for ReorderWithNulls<T, DESCENDING, NULLS_LAST>
{
    fn cmp(&self, other: &Self) -> Ordering {
        reorder_cmp(&self.0, &other.0, DESCENDING, NULLS_LAST)
    }
}

/// Compare two values with support for sort direction and nulls position.
///
/// # Panics
///
/// Panics if `T::partial_cmp(lhs, rhs)` returns `None`.
#[inline]
pub fn reorder_cmp<T: PartialOrd + IsNull>(
    lhs: &T,
    rhs: &T,
    descending: bool,
    nulls_last: bool,
) -> Ordering {
    match PartialOrd::partial_cmp(lhs, rhs).expect("expected total ordering") {
        Ordering::Equal => Ordering::Equal,
        _ if lhs.is_null() && nulls_last => Ordering::Greater,
        _ if rhs.is_null() && nulls_last => Ordering::Less,
        _ if lhs.is_null() => Ordering::Less,
        _ if rhs.is_null() => Ordering::Greater,
        ord if descending => ord.reverse(),
        ord => ord,
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    fn check(v: &[i64], ranks: &[usize]) {
        let mut sorted = v.to_vec();
        sorted.sort();
        let expected: Vec<i64> = ranks.iter().map(|&r| sorted[r]).collect();

        let mut reordered = v.to_vec();
        let mut out = vec![0; ranks.len()];
        select_many_unstable_by(&mut reordered, ranks, &mut out, Ord::cmp);
        assert_eq!(out, expected);
        reordered.sort();
        assert_eq!(reordered, sorted);
    }

    #[test]
    fn test_select_many() {
        let mut rng = SmallRng::seed_from_u64(0);
        for n in [1, 2, 3, 10, 17, 100, 1000] {
            let inputs: [Vec<i64>; 6] = [
                (0..n).map(|_| rng.random_range(-1000..1000)).collect(),
                (0..n).map(|_| rng.random_range(0..2)).collect(),
                (0..n).map(|_| rng.random_range(0..3)).collect(),
                vec![7; n],
                (0..n as i64).rev().collect(),
                (0..n)
                    .map(|_| rng.random_range(i64::MIN..i64::MAX))
                    .collect(),
            ];
            for v in &inputs {
                let all: Vec<usize> = (0..n).collect();
                let mut some: Vec<usize> = (0..10).map(|_| rng.random_range(0..n)).collect();
                some.sort();
                some.dedup();
                let (mid, far) = ([n / 2, n / 2 + 1], [n / 10, n - n / 10]);
                for ranks in [
                    &[][..],
                    &[0],
                    &[n - 1],
                    &[0, n - 1],
                    &all,
                    &some,
                    &mid,
                    &far,
                ] {
                    if ranks.is_sorted_by(|a, b| a < b) && ranks.last().is_none_or(|&r| r < n) {
                        check(v, ranks);
                    }
                }
            }
        }
    }
}
