use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_arrow::array::{LargeUnionArray, LargeUnionArrayBuilder};
use polars_compute::gather::take_unchecked;
use polars_error::constants::LENGTH_LIMIT_MSG;

use crate::chunked_array::ops::append::new_chunks_owned;
use crate::prelude::*;
use crate::series::IsSorted;

impl UnionChunked {
    /// The variants of this union.
    pub fn union_fields(&self) -> &[Field] {
        let DataType::Union(fields) = self.dtype() else {
            unreachable!()
        };
        fields
    }

    /// An all-null union of the given `Union` dtype.
    pub fn full_null_with_dtype(name: PlSmallStr, length: usize, dtype: &DataType) -> Self {
        let arr = LargeUnionArray::new_null(dtype.to_physical_arrow(CompatLevel::newest()), length);
        // SAFETY: the chunk has the physical arrow dtype of `dtype`.
        unsafe { Self::from_chunks_and_dtype_unchecked(name, vec![arr.boxed()], dtype.clone()) }
    }

    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        self.append_owned(other.clone())
    }

    pub fn append_owned(&mut self, mut other: Self) -> PolarsResult<()> {
        polars_ensure!(self.dtype() == other.dtype(), append);
        let len = self.len();
        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;
        self.set_sorted_flag(IsSorted::Not);
        new_chunks_owned(&mut self.chunks, std::mem::take(&mut other.chunks), len);
        Ok(())
    }

    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        self.append(other)?;
        self.rechunk_mut();
        Ok(())
    }
}

impl ChunkTakeUnchecked<IdxCa> for UnionChunked {
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let a = self.rechunk();
        let index = indices.rechunk();

        let chunks = a
            .downcast_iter()
            .zip(index.downcast_iter())
            .map(|(arr, idx)| unsafe { take_unchecked(arr, idx) })
            .collect::<Vec<_>>();
        unsafe { self.copy_with_chunks(chunks) }
    }
}

impl<I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for UnionChunked {
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let idx = IdxCa::mmap_slice(PlSmallStr::EMPTY, indices.as_ref());
        unsafe { self.take_unchecked(&idx) }
    }
}

impl ChunkExpandAtIndex<UnionType> for UnionChunked {
    fn new_from_index(&self, index: usize, length: usize) -> UnionChunked {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let chunk = self.downcast_get(chunk_idx).unwrap();
        let mut builder = LargeUnionArrayBuilder::new(chunk.dtype().clone());
        builder.subslice_extend_each_repeated(chunk, idx, 1, length, ShareStrategy::Always);
        // SAFETY: the chunk has the dtype of self.
        unsafe { self.copy_with_chunks(vec![builder.freeze().boxed()]) }
    }
}

impl ChunkShift<UnionType> for UnionChunked {
    fn shift(&self, periods: i64) -> UnionChunked {
        let len = self.len() as i64;
        let periods = periods.clamp(-len, len);
        let fill_length = periods.unsigned_abs() as usize;
        let mut slice = self.slice((-periods).max(0), self.len() - fill_length);
        let mut fill = Self::full_null_with_dtype(self.name().clone(), fill_length, self.dtype());

        if periods < 0 {
            slice.append_owned(fill).unwrap();
            slice
        } else {
            fill.append_owned(slice).unwrap();
            fill
        }
    }
}
