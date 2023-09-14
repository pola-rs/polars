use arrow::array::Array;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::chunked_array::ops::{ChunkGather, ChunkGatherUnchecked};
use crate::chunked_array::{ChunkedArray, ChunkedArrayLayout};
use crate::datatypes::{IdxCa, IdxType, PolarsDataType, PolarsNumericType, StaticArray};
use crate::prelude::*;

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
                a.iter().flatten().all(|i| (*i as usize) < len)
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
        let arrays: Vec<_> = self.downcast_iter().collect();
        if arrays.len() == 1 {
            // let arr = arrays.pop().unwrap();
            // self.apply_generic(op)
            // return match indices.layout() {
            //     ChunkedArrayLayout::SingleNoNull(idx_arr) => {
            //         idx_arr.values_iter().map(|&i| arr.get_value_unchecked(i)).collect_ca_like(self);
            //     },
            //     ChunkedArrayLayout::Single(_) => todo!(),
            //     ChunkedArrayLayout::MultiNoNull(_) => todo!(),
            //     ChunkedArrayLayout::Multi(_) => todo!(),
            // };
        }

        // indices
        //     .iter()
        //     .map(|&i| {


        //     })
            // .flat_map(|a| a.values_iter())
            // .collect_ca_like(self)
        todo!();
    }
}
