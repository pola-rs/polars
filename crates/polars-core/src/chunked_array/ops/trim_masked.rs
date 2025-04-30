use std::borrow::Cow;

use arrow::array::{Array, IntoBoxedArray};

use super::ListChunked;
use crate::chunked_array::flags::StatisticsFlags;

/// Trim out masked elements for lists.
pub trait ChunkTrimMasked: Clone {
    /// Remove masked list elements.
    fn trim_masked(&self) -> Cow<Self>;
}

impl ChunkTrimMasked for ListChunked {
    fn trim_masked(&self) -> Cow<Self> {
        let flags = self.get_flags();

        if flags.has_trimmed_masked()
            || (flags.can_fast_explode_list() && !self.inner_dtype().contains_list_recursive())
        {
            return Cow::Borrowed(self);
        }

        let chunks = self
            .downcast_iter()
            .map(|arr| arr.trim_to_normalized_offsets_recursive().into_boxed())
            .collect();

        // SAFETY: The length and null_count should remain the same.
        let mut ca = unsafe {
            Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
        };
        ca.set_flags(flags | StatisticsFlags::HAS_TRIMMED_MASKED);
        return Cow::Owned(ca);
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkTrimMasked for super::ArrayChunked {
    fn trim_masked(&self) -> Cow<Self> {
        let flags = self.get_flags();

        if flags.has_trimmed_masked() || !self.inner_dtype().contains_list_recursive() {
            return Cow::Borrowed(self);
        }

        let chunks = self
            .downcast_iter()
            .map(|arr| arr.trim_masked().into_boxed())
            .collect();

        self.flags.set(flags | StatisticsFlags::HAS_TRIMMED_MASKED);
        Cow::Borrowed(self)
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkTrimMasked for super::StructChunked {
    fn trim_masked(&self) -> Cow<Self> {
        let flags = self.get_flags();

        if flags.has_trimmed_masked() || !self.struct_fields().iter().any(|f| f.dtype().is_nested())
        {
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
            let ca = unsafe {
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
            };
            return Cow::Owned(ca);
        }

        self.flags.set(flags | StatisticsFlags::HAS_TRIMMED_MASKED);
        Cow::Borrowed(self)
    }
}
