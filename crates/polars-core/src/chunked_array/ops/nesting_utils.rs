use polars_array::{PlArray, StaticArray};
use polars_compute::find_validity_mismatch::find_validity_mismatch;
use polars_utils::IdxSize;

use super::ListChunked;
use crate::chunked_array::flags::StatisticsFlags;
use crate::prelude::{ChunkedArray, FalseT, PolarsDataType};
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
        use polars_compute::propagate_nulls::{propagate_nulls_list, propagate_nulls_list_shallow};

        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return None;
        }

        if !self.inner_dtype().is_nested() && !self.has_nulls() {
            self.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return None;
        }

        let map_aware = self.inner_dtype().contains_map();
        let propagate_chunk = |chunk| {
            if map_aware {
                propagate_nulls_list_shallow(chunk)
            } else {
                propagate_nulls_list(chunk)
            }
        };

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(propagated_chunk) = propagate_chunk(chunk) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(propagated_chunk.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        let mut out = if chunks.is_empty() {
            None
        } else {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(
                |chunk| match propagate_chunk(chunk) {
                    None => chunk.to_boxed(),
                    Some(chunk) => chunk.into_boxed(),
                },
            ));

            // SAFETY: The length and null_count should remain the same.
            Some(unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            })
        };

        if map_aware {
            out = propagate_values_nulls(out, self);
        }

        finish_propagate_nulls(out, self, flags)
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        use polars_compute::trim_lists_to_normalized_offsets::trim_lists_to_normalized_offsets_list;

        let flags = self.get_flags();

        if flags.has_trimmed_lists_to_normalized_offsets() {
            return None;
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(trimmed) = trim_lists_to_normalized_offsets_list(chunk) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(trimmed.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs compacting, create a new ArrayChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match trim_lists_to_normalized_offsets_list(chunk) {
                    Some(chunk) => chunk.into_boxed(),
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
            find_validity_mismatch(l, &**r, idxs);
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
        use polars_compute::propagate_nulls::{propagate_nulls_fsl, propagate_nulls_fsl_shallow};

        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return None;
        }

        if !self.inner_dtype().is_nested() && !self.has_nulls() {
            self.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return None;
        }

        let map_aware = self.inner_dtype().contains_map();
        let propagate_chunk = |chunk| {
            if map_aware {
                propagate_nulls_fsl_shallow(chunk)
            } else {
                propagate_nulls_fsl(chunk)
            }
        };

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(propagated_chunk) = propagate_chunk(chunk) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(propagated_chunk.into_boxed());
                break;
            }
        }

        let mut out = if chunks.is_empty() {
            None
        } else {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(
                |chunk| match propagate_chunk(chunk) {
                    None => chunk.to_boxed(),
                    Some(chunk) => chunk.into_boxed(),
                },
            ));

            // SAFETY: The length and null_count should remain the same.
            Some(unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            })
        };

        if map_aware {
            out = propagate_values_nulls(out, self);
        }

        finish_propagate_nulls(out, self, flags)
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
            if let Some(trimmed) = trim_lists_to_normalized_offsets_fsl(chunk) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(trimmed.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs compacting, create a new ArrayChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match trim_lists_to_normalized_offsets_fsl(chunk) {
                    Some(chunk) => chunk.into_boxed(),
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
            find_validity_mismatch(l, &**r, idxs);
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
        use polars_compute::propagate_nulls::{
            propagate_nulls_struct, propagate_nulls_struct_shallow,
        };

        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return None;
        }

        if self.struct_fields().iter().all(|f| !f.dtype().is_nested()) && !self.has_nulls() {
            self.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return None;
        }

        let map_aware = self.dtype().contains_map();
        let propagate_chunk = |chunk| {
            if map_aware {
                propagate_nulls_struct_shallow(chunk)
            } else {
                propagate_nulls_struct(chunk)
            }
        };

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Some(propagated_chunk) = propagate_chunk(chunk) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(propagated_chunk.into_boxed());
                break;
            }
        }

        let mut out = if chunks.is_empty() {
            None
        } else {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(
                |chunk| match propagate_chunk(chunk) {
                    None => chunk.to_boxed(),
                    Some(chunk) => chunk.into_boxed(),
                },
            ));

            // SAFETY: The length and null_count should remain the same.
            Some(unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            })
        };

        if map_aware {
            out = propagate_values_nulls(out, self);
        }

        finish_propagate_nulls(out, self, flags)
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
            if let Some(trimmed) = trim_lists_to_normalized_offsets_struct(chunk) {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(trimmed.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs compacting, create a new ArrayChunked
        if !chunks.is_empty() {
            chunks.extend(self.downcast_iter().skip(chunks.len()).map(|chunk| {
                match trim_lists_to_normalized_offsets_struct(chunk) {
                    Some(chunk) => chunk.into_boxed(),
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
            find_validity_mismatch(l, &**r, idxs);
            for idx in idxs[start_length..].iter_mut() {
                *idx += offset;
            }
            offset += l.len() as IdxSize;
        }
    }
}

/// Propagate through [`Series`] to respect logical dtypes. Arrow recursion treats Map
/// storage as a list and would null entries retained by null Map rows.
trait PropagateValuesNulls: Sized {
    fn values_propagate_nulls(&self) -> Option<Self>;
}

impl PropagateValuesNulls for ListChunked {
    fn values_propagate_nulls(&self) -> Option<Self> {
        let values = self.get_inner().propagate_nulls()?;
        Some(self.with_inner_values(&values))
    }
}

#[cfg(feature = "dtype-array")]
impl PropagateValuesNulls for super::ArrayChunked {
    fn values_propagate_nulls(&self) -> Option<Self> {
        let values = self.get_inner().propagate_nulls()?;
        Some(self.with_inner_values(&values))
    }
}

#[cfg(feature = "dtype-struct")]
impl PropagateValuesNulls for super::StructChunked {
    fn values_propagate_nulls(&self) -> Option<Self> {
        let fields = self.fields_as_series();
        let mut new_fields = Vec::with_capacity(fields.len());
        let mut changed = false;
        for field in &fields {
            let new_field = field.propagate_nulls();
            changed |= new_field.is_some();
            new_fields.push(new_field);
        }

        if !changed {
            return None;
        }

        // Field names, lengths, and outer validity are unchanged, so this cannot fail.
        let mut new_fields = new_fields.into_iter();
        let out = self
            .try_apply_fields(|field| {
                Ok(new_fields.next().unwrap().unwrap_or_else(|| field.clone()))
            })
            .expect("propagating nulls keeps the struct fields");
        Some(out)
    }
}

/// Propagate child nulls in `out`, or `orig` if unchanged.
fn propagate_values_nulls<T: PropagateValuesNulls>(out: Option<T>, orig: &T) -> Option<T> {
    out.as_ref()
        .unwrap_or(orig)
        .values_propagate_nulls()
        .or(out)
}

/// Mark `out` or `orig` as having propagated nulls.
fn finish_propagate_nulls<T: PolarsDataType>(
    out: Option<ChunkedArray<T>>,
    orig: &ChunkedArray<T>,
    flags: StatisticsFlags,
) -> Option<ChunkedArray<T>> {
    match out {
        Some(mut ca) => {
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            Some(ca)
        },
        None => {
            orig.flags
                .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            None
        },
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
            find_validity_mismatch(l, &**r, idxs);
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
            // A mask that repeats a single bit says the same of every element: the answer is
            // either every index or none of them, and no bits are walked to find that out.
            Some(v) => match v.scalar_value() {
                Some(true) => idxs.extend(0..v.len() as IdxSize),
                Some(false) => {},
                None => {
                    idxs.extend((v.flat_bitmap().unwrap().true_idx_iter()).map(|v| v as IdxSize))
                },
            },
        }
    }
}
