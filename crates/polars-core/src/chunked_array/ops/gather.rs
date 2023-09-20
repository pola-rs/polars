use arrow::array::Array;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::chunked_array::ops::{ChunkTake, ChunkTakeUnchecked};
use crate::chunked_array::ChunkedArray;
use crate::datatypes::{IdxCa, PolarsDataType, StaticArray};
use crate::prelude::*;

const BINARY_SEARCH_LIMIT: usize = 8;

impl<T: PolarsDataType, I: AsRef<[IdxSize]> + ?Sized> ChunkTake<I> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTakeUnchecked<I>,
{
    /// Gather values from ChunkedArray by index.
    fn take(&self, indices: &I) -> PolarsResult<Self> {
        let len = self.len();
        let all_valid = indices.as_ref().iter().all(|i| (*i as usize) < len);
        polars_ensure!(all_valid, ComputeError: "invalid index in gather");

        // SAFETY: we just checked the indices are valid.
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

impl<T: PolarsDataType> ChunkTake<IdxCa> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTakeUnchecked<IdxCa>,
{
    /// Gather values from ChunkedArray by index.
    fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        let len = self.len();
        let all_valid = indices.downcast_iter().all(|a| {
            if a.null_count() == 0 {
                a.values_iter().all(|i| (*i as usize) < len)
            } else {
                a.iter().flatten().all(|i| (*i as usize) < len)
            }
        });
        polars_ensure!(all_valid, ComputeError: "take indices are out of bounds");

        // SAFETY: we just checked the indices are valid.
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

/// Computes cumulative lengths for efficient branchless binary search
/// lookup. The first element is always 0, and the last length of arrs
/// is always ignored (as we already checked that all indices are
/// in-bounds we don't need to check against the last length).
fn cumulative_lengths<A: StaticArray>(arrs: &[&A]) -> [IdxSize; BINARY_SEARCH_LIMIT] {
    assert!(arrs.len() <= BINARY_SEARCH_LIMIT);
    let mut ret = [IdxSize::MAX; BINARY_SEARCH_LIMIT];
    ret[0] = 0;
    for i in 1..arrs.len() {
        ret[i] = ret[i - 1] + arrs[i - 1].len() as IdxSize;
    }
    ret
}

#[rustfmt::skip]
#[inline]
fn resolve_chunked_idx(idx: IdxSize, cumlens: &[IdxSize; BINARY_SEARCH_LIMIT]) -> (usize, usize) {
    // Branchless bitwise binary search.
    let mut chunk_idx = 0;
    chunk_idx += if idx >= cumlens[chunk_idx + 0b100] { 0b0100 } else { 0 };
    chunk_idx += if idx >= cumlens[chunk_idx + 0b010] { 0b0010 } else { 0 };
    chunk_idx += if idx >= cumlens[chunk_idx + 0b001] { 0b0001 } else { 0 };
    (chunk_idx, (idx - cumlens[chunk_idx]) as usize)
}

#[inline]
unsafe fn target_value_unchecked<'a, A: StaticArray>(
    targets: &[&'a A],
    cumlens: &[IdxSize; BINARY_SEARCH_LIMIT],
    idx: IdxSize,
) -> A::ValueT<'a> {
    let (chunk_idx, arr_idx) = resolve_chunked_idx(idx, cumlens);
    let arr = targets.get_unchecked(chunk_idx);
    arr.value_unchecked(arr_idx)
}

#[inline]
unsafe fn target_get_unchecked<'a, A: StaticArray>(
    targets: &[&'a A],
    cumlens: &[IdxSize; BINARY_SEARCH_LIMIT],
    idx: IdxSize,
) -> Option<A::ValueT<'a>> {
    let (chunk_idx, arr_idx) = resolve_chunked_idx(idx, cumlens);
    let arr = targets.get_unchecked(chunk_idx);
    arr.get_unchecked(arr_idx)
}

unsafe fn gather_idx_array_unchecked<A: StaticArray>(
    dtype: DataType,
    targets: &[&A],
    has_nulls: bool,
    indices: &[IdxSize],
) -> A {
    let it = indices.iter().copied();
    if targets.len() == 1 {
        let target = targets.first().unwrap();
        if has_nulls {
            it.map(|i| target.get_unchecked(i as usize))
                .collect_arr_trusted_with_dtype(dtype)
        } else {
            target.gather_unchecked_trusted(indices.iter().map(|i| *i as usize), dtype)
        }
    } else {
        let cumlens = cumulative_lengths(targets);
        if has_nulls {
            it.map(|i| target_get_unchecked(targets, &cumlens, i))
                .collect_arr_trusted_with_dtype(dtype)
        } else {
            it.map(|i| target_value_unchecked(targets, &cumlens, i))
                .collect_arr_trusted_with_dtype(dtype)
        }
    }
}

impl<T: PolarsDataType, I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for ChunkedArray<T> {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let rechunked;
        let mut ca = self;
        if self.chunks().len() > BINARY_SEARCH_LIMIT {
            rechunked = self.rechunk();
            ca = &rechunked;
        }
        let targets: Vec<_> = ca.downcast_iter().collect();
        let arr = gather_idx_array_unchecked(
            ca.dtype().clone(),
            &targets,
            ca.null_count() > 0,
            indices.as_ref(),
        );
        ChunkedArray::from_chunk_iter_like(ca, [arr])
    }
}

impl<T: PolarsDataType> ChunkTakeUnchecked<IdxCa> for ChunkedArray<T> {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let rechunked;
        let mut ca = self;
        if self.chunks().len() > BINARY_SEARCH_LIMIT {
            rechunked = self.rechunk();
            ca = &rechunked;
        }
        let targets_have_nulls = ca.null_count() > 0;
        let targets: Vec<_> = ca.downcast_iter().collect();

        let chunks = indices.downcast_iter().map(|idx_arr| {
            let dtype = ca.dtype().clone();
            if idx_arr.null_count() == 0 {
                gather_idx_array_unchecked(dtype, &targets, targets_have_nulls, idx_arr.values())
            } else if targets.len() == 1 {
                let target = targets.first().unwrap();
                if targets_have_nulls {
                    idx_arr
                        .iter()
                        .map(|i| target.get_unchecked(*i? as usize))
                        .collect_arr_trusted_with_dtype(dtype)
                } else {
                    idx_arr
                        .iter()
                        .map(|i| Some(target.value_unchecked(*i? as usize)))
                        .collect_arr_trusted_with_dtype(dtype)
                }
            } else {
                let cumlens = cumulative_lengths(&targets);
                if targets_have_nulls {
                    idx_arr
                        .iter()
                        .map(|i| target_get_unchecked(&targets, &cumlens, *i?))
                        .collect_arr_trusted_with_dtype(dtype)
                } else {
                    idx_arr
                        .iter()
                        .map(|i| Some(target_value_unchecked(&targets, &cumlens, *i?)))
                        .collect_arr_trusted_with_dtype(dtype)
                }
            }
        });

        let mut out = ChunkedArray::from_chunk_iter_like(ca, chunks);

        use crate::series::IsSorted::*;
        let sorted_flag = match (ca.is_sorted_flag(), indices.is_sorted_flag()) {
            (_, Not) => Not,
            (Not, _) => Not,
            (Ascending, Ascending) => Ascending,
            (Ascending, Descending) => Descending,
            (Descending, Ascending) => Descending,
            (Descending, Descending) => Ascending,
        };
        out.set_sorted_flag(sorted_flag);
        out
    }
}
