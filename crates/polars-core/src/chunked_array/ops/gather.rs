use arrow::array::Array;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::chunked_array::ops::{ChunkTake, ChunkTakeUnchecked};
use crate::chunked_array::ChunkedArray;
use crate::datatypes::{IdxCa, PolarsDataType, StaticArray};
use crate::prelude::*;
use crate::utils::index_to_chunked_index;

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

unsafe fn target_value_unchecked<'a, A: StaticArray>(
    targets: &[&'a A],
    idx: IdxSize,
) -> A::ValueT<'a> {
    let (chunk_idx, arr_idx) =
        index_to_chunked_index(targets.iter().map(|a| a.len()), idx as usize);
    let arr = targets.get_unchecked(chunk_idx);
    arr.value_unchecked(arr_idx)
}

unsafe fn target_get_unchecked<'a, A: StaticArray>(
    targets: &[&'a A],
    idx: IdxSize,
) -> Option<A::ValueT<'a>> {
    let (chunk_idx, arr_idx) =
        index_to_chunked_index(targets.iter().map(|a| a.len()), idx as usize);
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
        let arr = targets.iter().next().unwrap();
        if has_nulls {
            it.map(|i| arr.get_unchecked(i as usize))
                .collect_arr_with_dtype(dtype)
        } else {
            it.map(|i| arr.value_unchecked(i as usize))
                .collect_arr_with_dtype(dtype)
        }
    } else if has_nulls {
        it.map(|i| target_get_unchecked(targets, i))
            .collect_arr_with_dtype(dtype)
    } else {
        it.map(|i| target_value_unchecked(targets, i))
            .collect_arr_with_dtype(dtype)
    }
}

impl<T: PolarsDataType, I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for ChunkedArray<T> {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let targets: Vec<_> = self.downcast_iter().collect();
        let arr = gather_idx_array_unchecked(
            self.dtype().clone(),
            &targets,
            self.null_count() > 0,
            indices.as_ref(),
        );
        ChunkedArray::from_chunk_iter_like(self, [arr])
    }
}

impl<T: PolarsDataType> ChunkTakeUnchecked<IdxCa> for ChunkedArray<T> {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let targets_have_nulls = self.null_count() > 0;
        let targets: Vec<_> = self.downcast_iter().collect();

        let chunks = indices.downcast_iter().map(|idx_arr| {
            if idx_arr.null_count() == 0 {
                gather_idx_array_unchecked(
                    self.dtype().clone(),
                    &targets,
                    targets_have_nulls,
                    idx_arr.values(),
                )
            } else if targets.len() == 1 {
                let target = targets.first().unwrap();
                if targets_have_nulls {
                    idx_arr
                        .iter()
                        .map(|i| target.get_unchecked(*i? as usize))
                        .collect_arr_with_dtype(self.dtype().clone())
                } else {
                    idx_arr
                        .iter()
                        .map(|i| Some(target.value_unchecked(*i? as usize)))
                        .collect_arr_with_dtype(self.dtype().clone())
                }
            } else if targets_have_nulls {
                idx_arr
                    .iter()
                    .map(|i| target_get_unchecked(&targets, *i?))
                    .collect_arr_with_dtype(self.dtype().clone())
            } else {
                idx_arr
                    .iter()
                    .map(|i| Some(target_value_unchecked(&targets, *i?)))
                    .collect_arr_with_dtype(self.dtype().clone())
            }
        });

        let mut out = ChunkedArray::from_chunk_iter_like(self, chunks);

        use crate::series::IsSorted::*;
        let sorted_flag = match (self.is_sorted_flag(), indices.is_sorted_flag()) {
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
