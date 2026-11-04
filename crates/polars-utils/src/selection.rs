use std::cmp::Ordering;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// The unit of parallel work: 256 KiB of `f64` fits a core's L2 cache, and a fixed length makes
/// which of several equal values is selected the same on every machine and thread count.
pub const SELECT_BLOCK_LEN: usize = 1 << 15;
/// Below two blocks, entering the thread pool costs more than the parallel passes save.
pub const SELECT_MIN_PAR_LEN: usize = 2 * SELECT_BLOCK_LEN;
/// Values sampled per bracket: the bracket shrinks as the square root of this, while drawing and
/// sorting the sample is serial.
const SELECT_SAMPLE_LEN: usize = 1024;
/// At least half the drawn values are valid when mostly-null chunks are filtered first, so a
/// sample takes about `2 * SELECT_SAMPLE_LEN` draws; the cap only stops inputs that are not.
const SELECT_MAX_DRAWS: usize = 64 * SELECT_SAMPLE_LEN;
/// Three standard deviations of the position in the sample of the value at a rank, at most
/// `sqrt(SELECT_SAMPLE_LEN) / 2` each.
const SELECT_SLACK: usize = 3 * SELECT_SAMPLE_LEN.isqrt() / 2;

/// Writes the value at each of the `k` strictly ascending `ranks` (all below `v.len()`) of `v`
/// sorted by `cmp` to `out`, reordering `v`, in O(n log k).
///
/// Large inputs run in parallel on the current rayon pool, so call it inside polars' `RAYON.install`.
pub fn select_many_unstable_by<T, F>(v: &mut [T], ranks: &[usize], out: &mut [T], cmp: F)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    check_ranks(v.len(), ranks, out);
    if v.len() < SELECT_MIN_PAR_LEN {
        select_many_seq(v, ranks, 0, out, &cmp);
        return;
    }
    let len = v.len();
    let segs = v.chunks_mut(SELECT_BLOCK_LEN).collect();
    select_many_rec(segs, len, ranks, 0, out, &cmp);
}

/// Values of which, if `validity` is given, only those whose bit is set count.
///
/// `validity` holds the bytes of an Arrow validity bitmap and the bit offset of the first value.
#[derive(Clone, Copy)]
pub struct MaskedSlice<'a, T> {
    pub values: &'a [T],
    pub validity: Option<(&'a [u8], usize)>,
}

impl<'a, T> MaskedSlice<'a, T> {
    fn is_valid(&self, i: usize) -> bool {
        self.validity.is_none_or(|(bytes, offset)| {
            let bit = offset + i;
            (bytes[bit / 8] >> (bit % 8)) & 1 != 0
        })
    }

    fn slice(&self, start: usize, len: usize) -> Self {
        Self {
            values: &self.values[start..start + len],
            validity: self.validity.map(|(bytes, offset)| (bytes, offset + start)),
        }
    }
}

/// [`select_many_unstable_by`] on the `n_valid` values that count in the concatenation of
/// `chunks`, which it leaves untouched.
pub fn select_many_by<T, F>(
    chunks: &[MaskedSlice<'_, T>],
    n_valid: usize,
    ranks: &[usize],
    out: &mut [T],
    cmp: F,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    select_many_by_path(chunks, n_valid, ranks, out, cmp);
}

/// How [`select_many_by`] started the selection, which lets tests check that each path runs.
#[derive(Debug, PartialEq, Eq)]
enum SelectPath {
    /// Too few values to select in parallel.
    Sequential,
    /// Copied only the values in the sampled bracket.
    Bracket,
    /// Split every block in three around the sampled bracket.
    Split,
    /// Too few drawn values counted to sample, so copied those that count.
    Unsampled,
}

/// [`select_many_by`], returning the path it took.
fn select_many_by_path<T, F>(
    chunks: &[MaskedSlice<'_, T>],
    n_valid: usize,
    ranks: &[usize],
    out: &mut [T],
    cmp: F,
) -> SelectPath
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    check_ranks(n_valid, ranks, out);
    let raw_len: usize = chunks.iter().map(|c| c.values.len()).sum();
    if ranks.is_empty() || raw_len < SELECT_MIN_PAR_LEN {
        let mut valid = Vec::with_capacity(n_valid);
        for chunk in chunks {
            valid.extend(copy_valid(chunk));
        }
        assert_eq!(valid.len(), n_valid);
        select_many_unstable_by(&mut valid, ranks, out, cmp);
        return SelectPath::Sequential;
    }
    let blocks: Vec<MaskedSlice<'_, T>> = chunks
        .iter()
        .flat_map(|c| {
            (0..c.values.len())
                .step_by(SELECT_BLOCK_LEN)
                .map(|start| c.slice(start, SELECT_BLOCK_LEN.min(c.values.len() - start)))
        })
        .collect();
    let rank = ranks[ranks.len() / 2];
    if let Some(bracket) = sample_bracket(&blocks, n_valid, rank, 0, &cmp) {
        return select_blocks(&blocks, bracket, n_valid, ranks, out, &cmp);
    }
    // Too few of the drawn values counted, so select from a copy of only those that do.
    let mut valid: Vec<Vec<T>> = blocks.par_iter().map(copy_valid).collect();
    assert_eq!(valid.iter().map(Vec::len).sum::<usize>(), n_valid);
    let segs = valid.iter_mut().map(|v| &mut v[..]).collect();
    select_many_rec(segs, n_valid, ranks, 0, out, &cmp);
    SelectPath::Unsampled
}

/// [`select_many_by`] on `blocks` of at most [`SELECT_BLOCK_LEN`] values, given the
/// [`sample_bracket`] of their middle rank.
fn select_blocks<T, F>(
    blocks: &[MaskedSlice<'_, T>],
    (lo, hi, repeats): (T, T, bool),
    n_valid: usize,
    ranks: &[usize],
    out: &mut [T],
    cmp: &F,
) -> SelectPath
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let len = n_valid;
    let mid = ranks.len() / 2;

    // Ranks within half the slack of the middle one keep at least half the slack (1.5 standard
    // deviations) on either side of the bracket, so they most likely lie from `lo` to `hi` and only
    // the values between them need copying; with repeated values, those equal to either are only
    // counted.
    // Compared in `u64`, as the products overflow a 32-bit `usize` from a few million values.
    let near = |r: usize| {
        2 * r.abs_diff(ranks[mid]) as u64 * SELECT_SAMPLE_LEN as u64
            <= len as u64 * SELECT_SLACK as u64
    };
    if near(ranks[0]) && near(ranks[ranks.len() - 1]) {
        let mut between: Vec<(Vec<T>, [usize; 5])> = blocks
            .par_iter()
            .map_init(
                || Vec::with_capacity(SELECT_BLOCK_LEN),
                |scratch, block| copy_between(block, &lo, &hi, repeats, scratch, cmp),
            )
            .collect();
        let [below, eq_lo, n_between, eq_hi, valid] = between
            .iter()
            .fold([0; 5], |acc, (_, n)| std::array::from_fn(|i| acc[i] + n[i]));
        assert_eq!(valid, len);
        let (to_between, to_hi) = (below + eq_lo, below + eq_lo + n_between);
        if below <= ranks[0] && ranks[ranks.len() - 1] < to_hi + eq_hi {
            let first_between = ranks.partition_point(|&r| r < to_between);
            let first_hi = ranks.partition_point(|&r| r < to_hi);
            let (out_lo, rest) = out.split_at_mut(first_between);
            let (out_between, out_hi) = rest.split_at_mut(first_hi - first_between);
            out_lo.fill(lo);
            out_hi.fill(hi);
            let segs = between
                .iter_mut()
                .filter(|(v, _)| !v.is_empty())
                .map(|(v, _)| &mut v[..])
                .collect();
            let ranks_between = &ranks[first_between..first_hi];
            let k = ranks.len();
            select_part(
                segs,
                n_between,
                len,
                ranks_between,
                k,
                to_between,
                out_between,
                cmp,
            );
            return SelectPath::Bracket;
        }
    }

    let mut copies: Vec<(Vec<T>, [usize; 3])> = blocks
        .par_iter()
        .map(|block| {
            let mut copy = copy_valid(block);
            let sizes = split_block(&mut copy, &lo, &hi, cmp);
            (copy, sizes)
        })
        .collect();
    assert_eq!(copies.iter().map(|(c, _)| c.len()).sum::<usize>(), len);
    let mut parts: [Vec<&mut [T]>; 3] = Default::default();
    for (copy, sizes) in &mut copies {
        push_split(&mut parts, copy, *sizes);
    }
    select_split(parts, &lo, &hi, len, ranks, 0, out, cmp);
    SelectPath::Split
}

/// [`select_many_unstable_by`] on the concatenated `segs`, which hold `len` values whose sorted
/// order starts at rank `offset`.
fn select_many_rec<T, F>(
    segs: Vec<&mut [T]>,
    len: usize,
    ranks: &[usize],
    offset: usize,
    out: &mut [T],
    cmp: &F,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    if ranks.is_empty() {
        return;
    }
    if len < SELECT_MIN_PAR_LEN {
        return select_many_gathered(segs, ranks, offset, out, cmp);
    }
    let mid = ranks.len() / 2;
    let shared: Vec<MaskedSlice<'_, T>> = segs
        .iter()
        .map(|s| MaskedSlice {
            values: s,
            validity: None,
        })
        .collect();
    let (lo, hi, _) = sample_bracket(&shared, len, ranks[mid] - offset, offset, cmp)
        .expect("every drawn value counts");
    let split: Vec<(&mut [T], [usize; 3])> = segs
        .into_par_iter()
        .map(|seg| {
            let sizes = split_block(seg, &lo, &hi, cmp);
            (seg, sizes)
        })
        .collect();
    let mut parts: [Vec<&mut [T]>; 3] = Default::default();
    for (seg, sizes) in split {
        push_split(&mut parts, seg, sizes);
    }
    select_split(parts, &lo, &hi, len, ranks, offset, out, cmp);
}

/// Selects `ranks` from the values below, inside and above the bracket of `lo` and `hi` (see
/// [`below_hi`]) in `parts`, which together hold the `len` values whose sorted order starts at
/// rank `offset`.
#[allow(clippy::too_many_arguments)]
fn select_split<T, F>(
    parts: [Vec<&mut [T]>; 3],
    lo: &T,
    hi: &T,
    len: usize,
    ranks: &[usize],
    offset: usize,
    out: &mut [T],
    cmp: &F,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let sizes = parts
        .each_ref()
        .map(|p| p.iter().map(|s| s.len()).sum::<usize>());
    let below = ranks.partition_point(|&r| r < offset + sizes[0]);
    let not_above = ranks.partition_point(|&r| r < offset + sizes[0] + sizes[1]);
    let (ranks_a, rest) = ranks.split_at(below);
    let (ranks_b, ranks_c) = rest.split_at(not_above - below);
    let (out_a, rest) = out.split_at_mut(below);
    let (out_b, out_c) = rest.split_at_mut(not_above - below);
    let [a, b, c] = parts;
    let k = ranks.len();
    rayon::join(
        || select_part(a, sizes[0], len, ranks_a, k, offset, out_a, cmp),
        || {
            rayon::join(
                || {
                    // Values between equal bounds all equal them.
                    if cmp(lo, hi).is_eq() {
                        out_b.fill(*lo)
                    } else {
                        let b_offset = offset + sizes[0];
                        select_part(b, sizes[1], len, ranks_b, k, b_offset, out_b, cmp)
                    }
                },
                || {
                    let c_offset = offset + sizes[0] + sizes[1];
                    select_part(c, sizes[2], len, ranks_c, k, c_offset, out_c, cmp)
                },
            )
        },
    );
}

/// Selects `ranks` from a part of `len` values split off from `parent_len` values with
/// `parent_ranks` ranks.
#[allow(clippy::too_many_arguments)]
fn select_part<T, F>(
    segs: Vec<&mut [T]>,
    len: usize,
    parent_len: usize,
    ranks: &[usize],
    parent_ranks: usize,
    offset: usize,
    out: &mut [T],
    cmp: &F,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    // A split that shrinks neither the part nor its ranks by a constant factor came from an
    // unlucky sample; finishing that part sequentially keeps the worst case at O(n log k).
    if 4 * len <= 3 * parent_len || 4 * ranks.len() <= 3 * parent_ranks {
        select_many_rec(segs, len, ranks, offset, out, cmp)
    } else {
        select_many_gathered(segs, ranks, offset, out, cmp)
    }
}

/// [`select_many_rec`] sequentially, on a copy of `segs` unless there is only one.
fn select_many_gathered<T, F>(
    mut segs: Vec<&mut [T]>,
    ranks: &[usize],
    offset: usize,
    out: &mut [T],
    cmp: &F,
) where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    match segs.as_mut_slice() {
        _ if ranks.is_empty() => {},
        [seg] => select_many_seq(seg, ranks, offset, out, cmp),
        _ => select_many_seq(&mut segs.concat(), ranks, offset, out, cmp),
    }
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

/// Two values of the `len` that count in the concatenated `parts` whose ranks a sample estimates
/// to lie on either side of `rank`, close to it, and whether the sample holds either more than
/// once; or `None` if too few drawn values count to fill the sample.
fn sample_bracket<T, F>(
    parts: &[MaskedSlice<'_, T>],
    len: usize,
    rank: usize,
    seed: usize,
    cmp: &F,
) -> Option<(T, T, bool)>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    let starts: Vec<usize> = parts
        .iter()
        .scan(0, |end, p| {
            *end += p.values.len();
            Some(*end - p.values.len())
        })
        .collect();
    let raw_len = starts
        .last()
        .map_or(0, |s| s + parts[parts.len() - 1].values.len());
    // A fixed seed per part keeps which of several equal values lands at a rank deterministic.
    let mut rng = SmallRng::seed_from_u64(seed as u64);
    let mut sample: Vec<T> = Vec::with_capacity(SELECT_SAMPLE_LEN);
    for _ in 0..SELECT_MAX_DRAWS {
        if sample.len() == SELECT_SAMPLE_LEN {
            break;
        }
        let i = rng.random_range(0..raw_len);
        let p = starts.partition_point(|&s| s <= i) - 1;
        if parts[p].is_valid(i - starts[p]) {
            sample.push(parts[p].values[i - starts[p]]);
        }
    }
    if sample.len() < SELECT_SAMPLE_LEN {
        return None;
    }
    sample.sort_unstable_by(cmp);
    let pos = (rank as u64 * SELECT_SAMPLE_LEN as u64 / len as u64) as usize;
    let (lo_pos, hi_pos) = (
        pos.saturating_sub(SELECT_SLACK),
        (pos + SELECT_SLACK).min(SELECT_SAMPLE_LEN - 1),
    );
    let repeated = |i: usize| {
        [i.wrapping_sub(1), i + 1]
            .iter()
            .any(|&j| sample.get(j).is_some_and(|x| cmp(x, &sample[i]).is_eq()))
    };
    let repeats = repeated(lo_pos) || repeated(hi_pos);
    Some((sample[lo_pos], sample[hi_pos], repeats))
}

/// Whether `x` lies below the upper end of the bracket of `lo` and `hi`, which is `[lo, hi)`, or
/// `[lo, lo]` if `lo` equals `hi` (`hi_inclusive`).
///
/// Excluding `hi` keeps a bracket over two distinct values from holding all of them.
fn below_hi<T, F: Fn(&T, &T) -> Ordering>(x: &T, hi: &T, hi_inclusive: bool, cmp: &F) -> bool {
    if hi_inclusive {
        cmp(x, hi).is_le()
    } else {
        cmp(x, hi).is_lt()
    }
}

/// Partitions `v` into its values below the bracket of `lo` and `hi` (see [`below_hi`]), inside
/// it and above it, and returns their counts.
fn split_block<T, F>(v: &mut [T], lo: &T, hi: &T, cmp: &F) -> [usize; 3]
where
    F: Fn(&T, &T) -> Ordering,
{
    let hi_inclusive = cmp(lo, hi).is_eq();
    let below = partition_block(v, &|x| cmp(x, lo).is_lt());
    let inside = partition_block(&mut v[below..], &|x| below_hi(x, hi, hi_inclusive, cmp));
    [below, inside, v.len() - below - inside]
}

/// Copies the values that count in `block` and lie between `lo` and `hi`, and returns them with
/// the counts of those below `lo`, equal to `lo`, between, equal to `hi` (unless it equals `lo`),
/// and of all that count.
///
/// Only with `exact` are values equal to `lo` or `hi` counted apart instead of being between,
/// which costs more work per value; it is required if `lo` equals `hi`.
fn copy_between<T, F>(
    block: &MaskedSlice<'_, T>,
    lo: &T,
    hi: &T,
    exact: bool,
    scratch: &mut Vec<T>,
    cmp: &F,
) -> (Vec<T>, [usize; 5])
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    let counts = if exact {
        copy_classified(block, scratch, |x| {
            let (to_lo, to_hi) = (cmp(x, lo), cmp(x, hi));
            let eq_lo = to_lo.is_eq();
            [
                to_lo.is_lt(),
                eq_lo,
                to_lo.is_gt() & to_hi.is_lt(),
                to_hi.is_eq() & !eq_lo,
            ]
        })
    } else {
        debug_assert!(cmp(lo, hi).is_lt());
        copy_classified(block, scratch, |x| {
            let lt = cmp(x, lo).is_lt();
            [lt, false, !lt & cmp(x, hi).is_lt(), false]
        })
    };
    (scratch.clone(), counts)
}

/// Copies the values that count in `block` and whose third `classify` flag is set into `scratch`,
/// and returns how often each flag was set among them, and how many count.
fn copy_classified<T: Copy>(
    block: &MaskedSlice<'_, T>,
    scratch: &mut Vec<T>,
    classify: impl Fn(&T) -> [bool; 4],
) -> [usize; 5] {
    match block.validity {
        None => copy_where(block.values, scratch, |_, x| {
            let [a, b, c, d] = classify(x);
            (c, [a, b, c, d, true])
        }),
        Some(_) => copy_where(block.values, scratch, |i, x| {
            let valid = block.is_valid(i);
            let [a, b, c, d] = classify(x).map(|f| valid & f);
            (c, [a, b, c, d, valid])
        }),
    }
}

/// A copy of the values that count in `block`.
fn copy_valid<T: Copy>(block: &MaskedSlice<'_, T>) -> Vec<T> {
    let mut valid = Vec::new();
    match block.validity {
        None => valid.extend_from_slice(block.values),
        Some(_) => {
            copy_where(block.values, &mut valid, |i, _| (block.is_valid(i), []));
        },
    }
    valid
}

/// Copies the values of `v` for which `class(i, x)` says they're kept into `scratch`, and returns
/// how often each of the flags it also returns was set.
fn copy_where<T: Copy, const N: usize>(
    v: &[T],
    scratch: &mut Vec<T>,
    class: impl Fn(usize, &T) -> (bool, [bool; N]),
) -> [usize; N] {
    scratch.clear();
    scratch.reserve(v.len());
    let dst = scratch.spare_capacity_mut().as_mut_ptr();
    let (mut kept, mut counts) = (0, [0; N]);
    for (i, x) in v.iter().enumerate() {
        let (keep, flags) = class(i, x);
        // Writing every value and keeping only some avoids branching on them.
        // SAFETY: `kept <= i < v.len()`, within the reserved capacity.
        unsafe { (*dst.add(kept)).write(*x) };
        kept += keep as usize;
        for (count, flag) in counts.iter_mut().zip(flags) {
            *count += flag as usize;
        }
    }
    // SAFETY: the loop wrote the first `kept` values.
    unsafe { scratch.set_len(kept) };
    counts
}

/// Pushes the non-empty parts of `seg`, split into consecutive runs of `sizes`, onto `parts`.
fn push_split<'a, T>(parts: &mut [Vec<&'a mut [T]>; 3], seg: &'a mut [T], sizes: [usize; 3]) {
    let (a, rest) = seg.split_at_mut(sizes[0]);
    let (b, c) = rest.split_at_mut(sizes[1]);
    for (part, s) in parts.iter_mut().zip([a, b, c]) {
        if !s.is_empty() {
            part.push(s);
        }
    }
}

/// Partitions `v` so that the values for which `pred` holds come first, without branching on
/// `pred`, and returns their count.
fn partition_block<T, P: Fn(&T) -> bool>(v: &mut [T], pred: &P) -> usize {
    let mut split = 0;
    for i in 0..v.len() {
        // `v[..split]` holds matches and `v[split..i]` non-matches; swapping unconditionally keeps
        // both true whichever `v[i]` is, so the loop never branches on `pred`.
        let matches = pred(&v[i]);
        v.swap(split, i);
        split += matches as usize;
    }
    split
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(v: &[i64], ranks: &[usize]) {
        let mut sorted = v.to_vec();
        sorted.sort();
        let expected: Vec<i64> = ranks.iter().map(|&r| sorted[r]).collect();

        let mut out = vec![0; ranks.len()];
        let third = v.len() / 3;
        let chunks =
            [&v[..third], &v[third..2 * third], &v[2 * third..]].map(|values| MaskedSlice {
                values,
                validity: None,
            });
        select_many_by(&chunks, v.len(), ranks, &mut out, Ord::cmp);
        assert_eq!(out, expected);

        let mut reordered = v.to_vec();
        out.fill(0);
        select_many_unstable_by(&mut reordered, ranks, &mut out, Ord::cmp);
        assert_eq!(out, expected);
        reordered.sort();
        assert_eq!(reordered, sorted);
    }

    /// [`select_many_by`] on `v` with the values masked out where `mask` is false.
    fn check_masked(v: &[i64], mask: &[bool], ranks: &[usize]) -> SelectPath {
        let mut valid: Vec<i64> = v
            .iter()
            .zip(mask)
            .filter(|(_, m)| **m)
            .map(|(x, _)| *x)
            .collect();
        valid.sort();
        let ranks: Vec<usize> = ranks.iter().copied().filter(|&r| r < valid.len()).collect();
        let expected: Vec<i64> = ranks.iter().map(|&r| valid[r]).collect();
        // Start the bitmap at bit 3, so reads must handle a mask that starts mid-byte.
        let mut bytes = vec![0u8; (mask.len() + 3).div_ceil(8)];
        for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
            bytes[(i + 3) / 8] |= 1 << ((i + 3) % 8);
        }
        let half = v.len() / 2;
        let chunks = [
            MaskedSlice {
                values: &v[..half],
                validity: Some((&bytes[..], 3)),
            },
            MaskedSlice {
                values: &v[half..],
                validity: Some((&bytes[..], 3 + half)),
            },
        ];
        let mut out = vec![0; ranks.len()];
        let path = select_many_by_path(&chunks, valid.len(), &ranks, &mut out, Ord::cmp);
        assert_eq!(out, expected);
        path
    }

    #[test]
    fn test_select_many() {
        let mut rng = SmallRng::seed_from_u64(0);
        for n in [1, 2, 3, 10, 100, 1000, 300_000] {
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
                let all: Vec<usize> = if n <= 1000 { (0..n).collect() } else { vec![] };
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
                // Mostly valid values are masked as they are read, mostly invalid ones filtered
                // first.
                for valid_share in [0.7, 0.3] {
                    let mask: Vec<bool> = (0..n).map(|_| rng.random_bool(valid_share)).collect();
                    let n_valid = mask.iter().filter(|m| **m).count();
                    let (mid, far) = (
                        [n_valid / 2, n_valid / 2 + 1],
                        [n_valid / 10, n_valid - n_valid / 10],
                    );
                    for ranks in [&[0][..], &mid, &far, &some] {
                        if ranks.is_sorted_by(|a, b| a < b) {
                            check_masked(v, &mask, ranks);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_select_many_sample_positions_null() {
        // Nulls at every position the first sample would draw leave it no valid value.
        let n = 300_000;
        let mut mask = vec![true; n];
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..SELECT_MAX_DRAWS {
            mask[rng.random_range(0..n)] = false;
        }
        let v: Vec<i64> = (0..n as i64).rev().collect();
        let n_valid = mask.iter().filter(|m| **m).count();
        let path = check_masked(&v, &mask, &[0, n_valid / 2, n_valid - 1]);
        assert_eq!(path, SelectPath::Unsampled);
    }

    #[test]
    fn test_select_many_paths() {
        let mut rng = SmallRng::seed_from_u64(0);
        let n = 300_000;
        let mask = vec![true; n];
        let distinct: Vec<i64> = (0..n)
            .map(|_| rng.random_range(i64::MIN..i64::MAX))
            .collect();
        let two_values: Vec<i64> = (0..n).map(|_| rng.random_range(0..2)).collect();
        for (v, ranks, path) in [
            // Ranks near the middle one lie in its bracket, also when the bounds repeat.
            (&distinct, vec![n / 2, n / 2 + 1], SelectPath::Bracket),
            (&two_values, vec![n / 2], SelectPath::Bracket),
            // Ranks far from the middle one fall outside its bracket.
            (
                &distinct,
                vec![n / 10, n / 2, n - n / 10],
                SelectPath::Split,
            ),
            (
                &distinct[..1000].to_vec(),
                vec![500],
                SelectPath::Sequential,
            ),
        ] {
            assert_eq!(check_masked(v, &mask[..v.len()], &ranks), path);
        }
    }
}
