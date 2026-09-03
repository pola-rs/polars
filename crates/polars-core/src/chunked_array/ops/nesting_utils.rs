use polars_array::arrow::bridge::chunk_to_arrow;
use polars_array::arrow::export;
use polars_array::{PlArray, PlFixedSizeListArray, PlListArray, PlStructArray, StaticArray};
use polars_compute::find_validity_mismatch::find_validity_mismatch;
use polars_utils::IdxSize;

use super::ListChunked;
use crate::chunked_array::flags::StatisticsFlags;
use crate::prelude::{ChunkedArray, FalseT, PolarsDataType, ToArrow};
use crate::series::Series;
use crate::series::implementations::null::NullChunked;
use crate::utils::align_chunks_binary_ca_series;

/// Utility methods for dealing with nested chunked arrays.
pub trait ChunkNestingUtils: Sized {
    /// Propagate nulls of nested datatype to all levels of nesting.
    fn propagate_nulls(&self) -> Option<Self>;

    /// Trim all lists of unused start and end elements recursively.
    fn trim_lists_to_normalized_offsets(&self) -> Option<Self>;

    /// Find the indices of the values where the validity mismatches.
    ///
    /// This is done recursively.
    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>);
}

impl ChunkNestingUtils for ListChunked {
    fn propagate_nulls(&self) -> Option<Self> {
        use polars_compute::propagate_nulls::propagate_nulls_list;

        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return None;
        }

        if !self.inner_dtype().is_nested() && !self.has_nulls() {
            self.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(propagated_chunk) = propagate_nulls_list(&chunk_to_arrow(chunk)) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(PlListArray::from_arrow(&propagated_chunk).into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match propagate_nulls_list(&chunk_to_arrow(chunk)) {
                    None => chunk.to_boxed(),
                    Some(chunk) => PlListArray::from_arrow(&chunk).into_boxed(),
                }
            }));

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return Some(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
        None
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        use polars_compute::trim_lists_to_normalized_offsets::trim_lists_to_normalized_offsets_list;

        let flags = self.get_flags();

        if flags.has_trimmed_lists_to_normalized_offsets() {
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(trimmed) = trim_lists_to_normalized_offsets_list(&chunk_to_arrow(chunk)) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(PlListArray::from_arrow(&trimmed).into_boxed());
                break;
            }
        }

        // If we found a chunk that needs compacting, create a new ArrayChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match trim_lists_to_normalized_offsets_list(&chunk_to_arrow(chunk)) {
                    Some(chunk) => PlListArray::from_arrow(&chunk).into_boxed(),
                    None => chunk.to_boxed(),
                }
            }));

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };

            ca.set_flags(flags | StatisticsFlags::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS);
            return Some(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS);
        None
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        let (slf, other) = align_chunks_binary_ca_series(self, other);
        let mut offset: IdxSize = 0;
        for (l, r) in slf.downcast_iter().zip(other.chunks()) {
            let start_length = idxs.len();
            // The kernel is the Arrow one, so both chunks cross over — see `polars_array::arrow::bridge`.
            find_validity_mismatch(&*export::to_arrow(l), &*export::to_arrow(&**r), idxs);
            for idx in idxs[start_length..].iter_mut() {
                *idx += offset;
            }
            offset += l.len() as IdxSize;
        }
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkNestingUtils for super::ArrayChunked {
    fn propagate_nulls(&self) -> Option<Self> {
        use polars_compute::propagate_nulls::propagate_nulls_fsl;

        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return None;
        }

        if !self.inner_dtype().is_nested() && !self.has_nulls() {
            self.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(propagated_chunk) = propagate_nulls_fsl(&chunk_to_arrow(chunk)) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(PlFixedSizeListArray::from_arrow(&propagated_chunk).into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match propagate_nulls_fsl(&chunk_to_arrow(chunk)) {
                    None => chunk.to_boxed(),
                    Some(chunk) => PlFixedSizeListArray::from_arrow(&chunk).into_boxed(),
                }
            }));

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return Some(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
        None
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        use polars_compute::trim_lists_to_normalized_offsets::trim_lists_to_normalized_offsets_fsl;

        let flags = self.get_flags();

        if flags.has_trimmed_lists_to_normalized_offsets()
            || !self.inner_dtype().contains_list_recursive()
        {
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(trimmed) = trim_lists_to_normalized_offsets_fsl(&chunk_to_arrow(chunk)) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(PlFixedSizeListArray::from_arrow(&trimmed).into_boxed());
                break;
            }
        }

        // If we found a chunk that needs compacting, create a new ArrayChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match trim_lists_to_normalized_offsets_fsl(&chunk_to_arrow(chunk)) {
                    Some(chunk) => PlFixedSizeListArray::from_arrow(&chunk).into_boxed(),
                    None => chunk.to_boxed(),
                }
            }));

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS);
            return Some(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS);
        None
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        let (slf, other) = align_chunks_binary_ca_series(self, other);
        let mut offset: IdxSize = 0;
        for (l, r) in slf.downcast_iter().zip(other.chunks()) {
            let start_length = idxs.len();
            // The kernel is the Arrow one, so both chunks cross over — see `polars_array::arrow::bridge`.
            find_validity_mismatch(&*export::to_arrow(l), &*export::to_arrow(&**r), idxs);
            for idx in idxs[start_length..].iter_mut() {
                *idx += offset;
            }
            offset += l.len() as IdxSize;
        }
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkNestingUtils for super::StructChunked {
    fn propagate_nulls(&self) -> Option<Self> {
        use polars_compute::propagate_nulls::propagate_nulls_struct;

        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return None;
        }

        if self.struct_fields().iter().all(|f| !f.dtype().is_nested()) && !self.has_nulls() {
            self.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(propagated_chunk) = propagate_nulls_struct(&chunk_to_arrow(chunk)) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(PlStructArray::from_arrow(&propagated_chunk).into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match propagate_nulls_struct(&chunk_to_arrow(chunk)) {
                    None => chunk.to_boxed(),
                    Some(chunk) => PlStructArray::from_arrow(&chunk).into_boxed(),
                }
            }));

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return Some(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
        None
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        use polars_compute::trim_lists_to_normalized_offsets::trim_lists_to_normalized_offsets_struct;

        let flags = self.get_flags();

        if flags.has_trimmed_lists_to_normalized_offsets()
            || !self
                .struct_fields()
                .iter()
                .any(|f| f.dtype().contains_list_recursive())
        {
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(trimmed) = trim_lists_to_normalized_offsets_struct(&chunk_to_arrow(chunk)) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(PlStructArray::from_arrow(&trimmed).into_boxed());
                break;
            }
        }

        // If we found a chunk that needs compacting, create a new ArrayChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match trim_lists_to_normalized_offsets_struct(&chunk_to_arrow(chunk)) {
                    Some(chunk) => PlStructArray::from_arrow(&chunk).into_boxed(),
                    None => chunk.to_boxed(),
                }
            }));

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS);
            return Some(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS);
        None
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        let (slf, other) = align_chunks_binary_ca_series(self, other);
        let mut offset: IdxSize = 0;
        for (l, r) in slf.downcast_iter().zip(other.chunks()) {
            let start_length = idxs.len();
            // The kernel is the Arrow one, so both chunks cross over — see `polars_array::arrow::bridge`.
            find_validity_mismatch(&*export::to_arrow(l), &*export::to_arrow(&**r), idxs);
            for idx in idxs[start_length..].iter_mut() {
                *idx += offset;
            }
            offset += l.len() as IdxSize;
        }
    }
}

impl<T: PolarsDataType<IsNested = FalseT>> ChunkNestingUtils for ChunkedArray<T> {
    fn propagate_nulls(&self) -> Option<Self> {
        None
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        None
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        let slf_nc = self.null_count();
        let other_nc = other.null_count();

        // Fast path for non-nested datatypes.
        if slf_nc == other_nc && (slf_nc == 0 || slf_nc == self.len()) {
            return;
        }

        let (slf, other) = align_chunks_binary_ca_series(self, other);
        let mut offset: IdxSize = 0;
        for (l, r) in slf.downcast_iter().zip(other.chunks()) {
            let start_length = idxs.len();
            // The kernel is the Arrow one, so both chunks cross over — see `polars_array::arrow::bridge`.
            find_validity_mismatch(&*export::to_arrow(l), &*export::to_arrow(&**r), idxs);
            for idx in idxs[start_length..].iter_mut() {
                *idx += offset;
            }
            offset += l.len() as IdxSize;
        }
    }
}

impl ChunkNestingUtils for NullChunked {
    fn propagate_nulls(&self) -> Option<Self> {
        None
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        None
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        let other_nc = other.null_count();

        // Fast path for non-nested datatypes.
        if other_nc == self.len() {
            return;
        }

        match other.rechunk_validity() {
            None => idxs.extend(0..self.len() as IdxSize),
            Some(v) => idxs.extend(v.true_idx_iter().map(|v| v as IdxSize)),
        }
    }
}
