use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SearchSortedSide {
    #[default]
    Any,
    Left,
    Right,
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

    if !descending {
        if !has_nulls {
            if side == SearchSortedSide::Right {
                lower_bound_chunks::<T, _>(
                    &chunks,
                    search_values,
                    null_idx,
                    |chunk, i, sv| unsafe { chunk.value_unchecked(i).tot_gt(sv) },
                )
            } else {
                lower_bound_chunks::<T, _>(
                    &chunks,
                    search_values,
                    null_idx,
                    |chunk, i, sv| unsafe { chunk.value_unchecked(i).tot_ge(sv) },
                )
            }
        } else {
            if side == SearchSortedSide::Right {
                lower_bound_chunks::<T, _>(&chunks, search_values, null_idx, |chunk, i, sv| {
                    if let Some(v) = unsafe { chunk.get_unchecked(i) } {
                        v.tot_gt(sv)
                    } else {
                        nulls_last
                    }
                })
            } else {
                lower_bound_chunks::<T, _>(&chunks, search_values, null_idx, |chunk, i, sv| {
                    if let Some(v) = unsafe { chunk.get_unchecked(i) } {
                        v.tot_ge(sv)
                    } else {
                        nulls_last
                    }
                })
            }
        }
    } else {
        if !has_nulls {
            if side == SearchSortedSide::Right {
                lower_bound_chunks::<T, _>(
                    &chunks,
                    search_values,
                    null_idx,
                    |chunk, i, sv| unsafe { chunk.value_unchecked(i).tot_lt(sv) },
                )
            } else {
                lower_bound_chunks::<T, _>(
                    &chunks,
                    search_values,
                    null_idx,
                    |chunk, i, sv| unsafe { chunk.value_unchecked(i).tot_le(sv) },
                )
            }
        } else {
            if side == SearchSortedSide::Right {
                lower_bound_chunks::<T, _>(&chunks, search_values, null_idx, |chunk, i, sv| {
                    if let Some(v) = unsafe { chunk.get_unchecked(i) } {
                        v.tot_lt(sv)
                    } else {
                        nulls_last
                    }
                })
            } else {
                lower_bound_chunks::<T, _>(&chunks, search_values, null_idx, |chunk, i, sv| {
                    if let Some(v) = unsafe { chunk.get_unchecked(i) } {
                        v.tot_le(sv)
                    } else {
                        nulls_last
                    }
                })
            }
        }
    }
}
