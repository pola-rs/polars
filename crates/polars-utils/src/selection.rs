use std::cmp::Ordering;

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
