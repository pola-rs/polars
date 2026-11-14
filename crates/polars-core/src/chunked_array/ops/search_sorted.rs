use std::fmt::Debug;

use polars_arrow::bitmap::utils::SlicesIterator;
use polars_utils::binary_search::ExponentialSearch;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use crate::runtime::RAYON;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum SearchSortedSide {
    #[default]
    Any,
    Left,
    Right,
}

impl SearchSortedSide {
    pub fn flip(self) -> Self {
        use SearchSortedSide::*;
        match self {
            Any => Any,
            Left => Right,
            Right => Left,
        }
    }
}

/// Computes the first point on [lo, hi) where f is true, assuming it is first
/// always false and then always true. It is assumed f(hi) is true.
/// midpoint is a function that returns some lo < i < hi if one exists, else lo.
fn lower_bound<I, F, M>(mut lo: I, mut hi: I, midpoint: M, f: F) -> I
where
    I: PartialEq + Eq,
    M: Fn(&I, &I) -> I,
    F: Fn(&I) -> bool,
{
    loop {
        let m = midpoint(&lo, &hi);
        if m == lo {
            return if f(&lo) { lo } else { hi };
        }

        if f(&m) {
            hi = m;
        } else {
            lo = m;
        }
    }
}

/// Search through a series of chunks for the first position where f(x) is true,
/// assuming it is first always false and then always true.
///
/// It repeats this for each value in search_values. If the search value is null null_idx is
/// returned.
///
/// Assumes the chunks are non-empty.
pub fn lower_bound_chunks<'a, T, F>(
    chunks: &[&'a T::Array],
    search_values: impl Iterator<Item = Option<T::Physical<'a>>>,
    null_idx: IdxSize,
    f: F,
) -> Vec<IdxSize>
where
    T: PolarsDataType,
    F: Fn(&'a T::Array, usize, &T::Physical<'a>) -> bool,
{
    if chunks.is_empty() {
        return search_values.map(|_| 0).collect();
    }

    // Fast-path: only a single chunk.
    if chunks.len() == 1 {
        let chunk = &chunks[0];
        return search_values
            .map(|ov| {
                if let Some(v) = ov {
                    lower_bound(0, chunk.len(), |l, r| (l + r) / 2, |m| f(chunk, *m, &v)) as IdxSize
                } else {
                    null_idx
                }
            })
            .collect();
    }

    // Multiple chunks, precompute prefix sum of lengths so we can look up
    // in O(1) the global position of chunk i.
    let mut sz = 0;
    let mut chunk_len_prefix_sum = Vec::with_capacity(chunks.len() + 1);
    for c in chunks {
        chunk_len_prefix_sum.push(sz);
        sz += c.len();
    }
    chunk_len_prefix_sum.push(sz);

    // For each search value do a binary search on (chunk_idx, idx_in_chunk) pairs.
    search_values
        .map(|ov| {
            let Some(v) = ov else {
                return null_idx;
            };
            let left = (0, 0);
            let right = (chunks.len(), 0);
            let midpoint = |l: &(usize, usize), r: &(usize, usize)| {
                if l.0 == r.0 {
                    // Within same chunk.
                    (l.0, (l.1 + r.1) / 2)
                } else if l.0 + 1 == r.0 {
                    // Two adjacent chunks, might have to be l or r.
                    let left_len = chunks[l.0].len() - l.1;

                    let logical_mid = (left_len + r.1) / 2;
                    if logical_mid < left_len {
                        (l.0, l.1 + logical_mid)
                    } else {
                        (r.0, logical_mid - left_len)
                    }
                } else {
                    // Has a chunk in between.
                    ((l.0 + r.0) / 2, 0)
                }
            };

            let bound = lower_bound(left, right, midpoint, |m| {
                f(unsafe { chunks.get_unchecked(m.0) }, m.1, &v)
            });

            (chunk_len_prefix_sum[bound.0] + bound.1) as IdxSize
        })
        .collect()
}

#[allow(clippy::collapsible_else_if)]
pub fn binary_search_ca<'a, T>(
    ca: &'a ChunkedArray<T>,
    search_values: impl Iterator<Item = Option<T::Physical<'a>>>,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalOrd + Debug + Copy,
{
    let chunks: Vec<_> = ca.downcast_iter().filter(|c| c.len() > 0).collect();
    let has_nulls = ca.null_count() > 0;
    let nulls_last = has_nulls && chunks[0].get(0).is_some();
    let null_idx = if nulls_last {
        if side == SearchSortedSide::Right {
            ca.len()
        } else {
            ca.len() - ca.null_count()
        }
    } else {
        if side == SearchSortedSide::Right {
            ca.null_count()
        } else {
            0
        }
    } as IdxSize;

    // A value v is past the needle sv when, ascending, v > sv (right side) or !(sv > v) (left
    // side). Descending swaps v and sv.
    let right = side == SearchSortedSide::Right;
    let swap = descending == right;
    lower_bound_chunks::<T, _>(&chunks, search_values, null_idx, |chunk, i, sv| {
        match unsafe { chunk.get_unchecked(i) } {
            Some(v) => {
                let (a, b) = if swap { (sv, &v) } else { (&v, sv) };
                a.tot_gt(b) == right
            },
            None => nulls_last,
        }
    })
}

/// [`binary_search_ca`] for numeric `needles`, faster if `ca` is one chunk without nulls.
pub fn search_sorted_numeric<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    needles: &ChunkedArray<T>,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize> {
    match ca.cont_slice() {
        Ok(haystack) => search_sorted_slice(haystack, needles, side, descending),
        Err(_) => binary_search_ca(ca, needles.iter(), side, descending),
    }
}

/// [`binary_search_ca`] for a `haystack` held in one slice without nulls, searching the needles
/// with branchless binary searches, or from the previous needle's position within blocks of needles
/// that are sorted like the haystack, in parallel for many needles.
fn search_sorted_slice<T: PolarsNumericType>(
    haystack: &[T::Native],
    needles: &ChunkedArray<T>,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize> {
    const BLOCK: usize = 1 << 12;
    const MIN_PAR_LEN: usize = 1 << 16;

    let right = side == SearchSortedSide::Right;
    let before = |x: &T::Native, v: &T::Native| match (descending, right) {
        (false, false) => x.tot_lt(v),
        (false, true) => x.tot_le(v),
        (true, false) => x.tot_gt(v),
        (true, true) => x.tot_ge(v),
    };
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let position = |v: &T::Native| -> usize {
        // The float total order adds a NaN test to every comparison. For a needle that isn't NaN
        // the IEEE comparisons below agree with it, also against a NaN in the haystack, so only
        // NaN needles pay for it.
        if v.partial_cmp(v).is_some() {
            match (descending, right) {
                (false, false) => haystack.partition_point(|x| x < v),
                (false, true) => haystack.partition_point(|x| x <= v),
                (true, false) => haystack.partition_point(|x| !(x <= v)),
                (true, true) => haystack.partition_point(|x| !(x < v)),
            }
        } else {
            haystack.partition_point(|x| before(x, v))
        }
    };
    let ordered_like_haystack = |a: &T::Native, b: &T::Native| {
        if descending { a.tot_ge(b) } else { a.tot_le(b) }
    };
    let fill = |(out, vals): (&mut [IdxSize], &[T::Native])| {
        if vals.is_sorted_by(ordered_like_haystack) {
            // Each needle lands at or after the previous one, usually close by, where a search
            // touches few values and its branches are predictable.
            let mut pos = 0;
            for (o, v) in out.iter_mut().zip(vals) {
                if haystack.get(pos).is_some_and(|x| before(x, v)) {
                    pos += 1 + (&haystack[pos + 1..]).partition_point_exponential(|x| before(x, v));
                }
                *o = pos as IdxSize;
            }
        } else {
            for (o, v) in out.iter_mut().zip(vals) {
                *o = position(v) as IdxSize;
            }
        }
    };

    let mut out = vec![0 as IdxSize; needles.len()];
    let mut offset = 0;
    for arr in needles.downcast_iter() {
        let part = &mut out[offset..offset + arr.len()];
        offset += arr.len();
        // Order is detected per block, so sorted runs inside unsorted needles also take the
        // forward search.
        if arr.len() >= MIN_PAR_LEN {
            RAYON.install(|| {
                part.par_chunks_mut(BLOCK)
                    .zip(arr.values().par_chunks(BLOCK))
                    .for_each(fill)
            });
        } else {
            part.chunks_mut(BLOCK)
                .zip(arr.values().chunks(BLOCK))
                .for_each(fill);
        }
        // As in `binary_search_ca`, a null needle goes first when the haystack has no nulls.
        if let Some(validity) = arr.validity().filter(|v| v.unset_bits() > 0) {
            let mut valid_end = 0;
            for (start, len) in SlicesIterator::new(validity) {
                part[valid_end..start].fill(0);
                valid_end = start + len;
            }
            part[valid_end..].fill(0);
        }
    }
    out
}
