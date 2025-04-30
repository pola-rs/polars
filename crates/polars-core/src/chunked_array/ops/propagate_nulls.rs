use std::borrow::Cow;

use arrow::array::{Array, IntoBoxedArray};

use super::ListChunked;
use crate::chunked_array::flags::StatisticsFlags;

/// Propagate nulls to nested values.
pub trait ChunkPropagateNulls: Clone {
    /// Propagate nulls of nested datatype to all levels of nesting.
    fn propagate_nulls(&self) -> Cow<Self>;
}

impl ChunkPropagateNulls for ListChunked {
    fn propagate_nulls(&self) -> Cow<Self> {
        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return Cow::Borrowed(self);
        }

        // Fast explode also means there are no nulls to propagate to.
        if flags.has_trimmed_masked() {
            return Cow::Borrowed(self);
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Cow::Owned(propagated_chunk) = chunk.propagate_nulls() {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(propagated_chunk.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        if !chunks.is_empty() {
            chunks.extend(
                self.downcast_iter()
                    .skip(chunks.len())
                    .map(|chunk| chunk.propagate_nulls().to_boxed()),
            );

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return Cow::Owned(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
        Cow::Borrowed(self)
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkPropagateNulls for super::ArrayChunked {
    fn propagate_nulls(&self) -> Cow<Self> {
        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return Cow::Borrowed(self);
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Cow::Owned(propagated_chunk) = chunk.propagate_nulls() {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(propagated_chunk.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        if !chunks.is_empty() {
            chunks.extend(
                self.downcast_iter()
                    .skip(chunks.len())
                    .map(|chunk| chunk.propagate_nulls().to_boxed()),
            );

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return Cow::Owned(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
        Cow::Borrowed(self)
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkPropagateNulls for super::StructChunked {
    fn propagate_nulls(&self) -> Cow<Self> {
        let flags = self.get_flags();

        if flags.has_propagated_nulls() {
            return Cow::Borrowed(self);
        }

        let mut chunks = Vec::new();
        for (i, chunk) in self.downcast_iter().enumerate() {
            if let Cow::Owned(propagated_chunk) = chunk.propagate_nulls() {
                chunks.reserve(self.chunks.len());
                chunks.extend(self.chunks[..i].iter().cloned());
                chunks.push(propagated_chunk.into_boxed());
                break;
            }
        }

        // If we found a chunk that needs propagating, create a new ListChunked
        if !chunks.is_empty() {
            chunks.extend(
                self.downcast_iter()
                    .skip(chunks.len())
                    .map(|chunk| chunk.propagate_nulls().to_boxed()),
            );

            // SAFETY: The length and null_count should remain the same.
            let mut ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            ca.set_flags(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
            return Cow::Owned(ca);
        }

        self.flags
            .set(flags | StatisticsFlags::HAS_PROPAGATED_NULLS);
        Cow::Borrowed(self)
    }
}
