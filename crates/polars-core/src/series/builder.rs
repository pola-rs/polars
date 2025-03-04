use arrow::array::builder::{make_builder, ArrayBuilder, ShareStrategy};
use polars_utils::IdxSize;

use crate::prelude::*;
use crate::utils::Container;

/// A type-erased wrapper around ArrayBuilder.
pub struct SeriesBuilder {
    dtype: DataType,
    builder: Box<dyn ArrayBuilder>,
}

impl SeriesBuilder {
    pub fn new(dtype: DataType) -> Self {
        let builder = make_builder(&dtype.to_physical().to_arrow(CompatLevel::newest()));
        Self { dtype, builder }
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        self.builder.reserve(additional);
    }

    pub fn freeze(self, name: PlSmallStr) -> Series {
        unsafe {
            Series::from_chunks_and_dtype_unchecked(name, vec![self.builder.freeze()], &self.dtype)
        }
    }

    pub fn len(&self) -> usize {
        self.builder.len()
    }

    pub fn is_empty(&self) -> bool {
        self.builder.len() == 0
    }

    /// Extends this builder with the contents of the given series. May panic if
    /// other does not match the dtype of this builder.
    #[inline(always)]
    pub fn extend(&mut self, other: &Series, share: ShareStrategy) {
        self.subslice_extend(other, 0, other.len(), share);
    }

    /// Extends this builder with the contents of the given series subslice.
    /// May panic if other does not match the dtype of this builder.
    pub fn subslice_extend(
        &mut self,
        other: &Series,
        mut start: usize,
        mut length: usize,
        share: ShareStrategy,
    ) {
        if length == 0 || other.len() == 0 {
            return;
        }

        for chunk in other.chunks() {
            if start < chunk.len() {
                let length_in_chunk = length.min(chunk.len() - start);
                self.builder
                    .subslice_extend(&**chunk, start, length_in_chunk, share);

                start = 0;
                length -= length_in_chunk;
                if length == 0 {
                    break;
                }
            } else {
                start -= chunk.len();
            }
        }
    }

    pub fn subslice_extend_repeated(
        &mut self,
        other: &Series,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        if length == 0 || other.len() == 0 {
            return;
        }

        let chunks = other.chunks();
        if chunks.len() == 1 {
            self.builder
                .subslice_extend_repeated(&*chunks[0], start, length, repeats, share);
        } else {
            for _ in 0..repeats {
                self.subslice_extend(other, start, length, share);
            }
        }
    }

    /// Extends this builder with the contents of the given series at the given
    /// indices. That is, `other[idxs[i]]` is appended to this builder in order,
    /// for each i=0..idxs.len(). May panic if other does not match the dtype
    /// of this builder, or if the other series is not rechunked.
    ///
    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather_extend(&mut self, other: &Series, idxs: &[IdxSize], share: ShareStrategy) {
        let chunks = other.chunks();
        assert!(chunks.len() == 1);
        self.builder.gather_extend(&*chunks[0], idxs, share);
    }

    pub fn opt_gather_extend(&mut self, other: &Series, idxs: &[IdxSize], share: ShareStrategy) {
        let chunks = other.chunks();
        assert!(chunks.len() == 1);
        self.builder.opt_gather_extend(&*chunks[0], idxs, share);
    }
}
