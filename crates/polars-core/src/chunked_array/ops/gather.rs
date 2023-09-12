use arrow::array::Array;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::chunked_array::ops::{ChunkGather, ChunkGatherUnchecked};
use crate::chunked_array::{ChunkedArray, ChunkedArrayLayout};
use crate::datatypes::{IdxCa, IdxType, PolarsDataType, PolarsNumericType, StaticArray};


impl<T: PolarsDataType> ChunkGather for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkGatherUnchecked,
{
    /// Gather values from ChunkedArray by index.
    fn gather(&self, indices: &IdxCa) -> PolarsResult<Self> {
        let len = self.len();
        let all_valid = indices.downcast_iter().all(|a| {
            if a.null_count() == 0 {
                a.values_iter().all(|i| (*i as usize) < len)
            } else {
                a.iter().filter_map(|x| x).all(|i| (*i as usize) < len)
            }
        });
        polars_ensure!(all_valid, ComputeError: "invalid index in gather");

        // SAFETY: we just checked the indices are valid.
        Ok(unsafe { self.gather_unchecked(indices) })
    }
}

impl<T: PolarsDataType> ChunkGatherUnchecked for ChunkedArray<T> {
    /// Gather values from ChunkedArray by index.
    unsafe fn gather_unchecked(&self, indices: &IdxCa) -> Self {
        todo!()
    }
}

