use arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use polars_utils::IdxSize;

#[cfg(feature = "object")]
use crate::chunked_array::object::registry::get_object_builder;
use crate::prelude::*;
use crate::utils::Container;

#[cfg(feature = "dtype-categorical")]
#[inline(always)]
fn fill_rev_map(dtype: &DataType, rev_map_merger: &mut Option<Box<GlobalRevMapMerger>>) {
    if let DataType::Categorical(Some(rev_map), _) = dtype {
        assert!(
            rev_map.is_active_global(),
            "{}",
            polars_err!(string_cache_mismatch)
        );
        if let Some(merger) = rev_map_merger {
            merger.merge_map(rev_map).unwrap();
        } else {
            *rev_map_merger = Some(Box::new(GlobalRevMapMerger::new(rev_map.clone())));
        }
    }
}

/// A type-erased wrapper around ArrayBuilder.
pub struct SeriesBuilder {
    dtype: DataType,
    builder: Box<dyn ArrayBuilder>,
    #[cfg(feature = "dtype-categorical")]
    rev_map_merger: Option<Box<GlobalRevMapMerger>>,
}

impl SeriesBuilder {
    pub fn new(dtype: DataType) -> Self {
        // FIXME: get rid of this hack.
        #[cfg(feature = "object")]
        if matches!(dtype, DataType::Object(_)) {
            let builder = get_object_builder(PlSmallStr::EMPTY, 0).as_array_builder();
            return Self {
                dtype,
                builder,
                #[cfg(feature = "dtype-categorical")]
                rev_map_merger: None,
            };
        }

        let builder = make_builder(&dtype.to_physical().to_arrow(CompatLevel::newest()));
        Self {
            dtype,
            builder,
            #[cfg(feature = "dtype-categorical")]
            rev_map_merger: None,
        }
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        self.builder.reserve(additional);
    }

    fn freeze_dtype(&mut self) -> DataType {
        #[cfg(feature = "dtype-categorical")]
        if let Some(rev_map_merger) = self.rev_map_merger.take() {
            let DataType::Categorical(_, order) = self.dtype else {
                unreachable!()
            };
            return DataType::Categorical(Some(rev_map_merger.finish()), order);
        }

        self.dtype.clone()
    }

    pub fn freeze(mut self, name: PlSmallStr) -> Series {
        unsafe {
            let dtype = self.freeze_dtype();
            Series::from_chunks_and_dtype_unchecked(name, vec![self.builder.freeze()], &dtype)
        }
    }

    pub fn freeze_reset(&mut self, name: PlSmallStr) -> Series {
        unsafe {
            Series::from_chunks_and_dtype_unchecked(
                name,
                vec![self.builder.freeze_reset()],
                &self.freeze_dtype(),
            )
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
        #[cfg(feature = "dtype-categorical")]
        {
            fill_rev_map(other.dtype(), &mut self.rev_map_merger);
        }

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
        #[cfg(feature = "dtype-categorical")]
        {
            fill_rev_map(other.dtype(), &mut self.rev_map_merger);
        }

        if length == 0 || other.is_empty() {
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
        #[cfg(feature = "dtype-categorical")]
        {
            fill_rev_map(other.dtype(), &mut self.rev_map_merger);
        }

        if length == 0 || other.is_empty() {
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

    pub fn subslice_extend_each_repeated(
        &mut self,
        other: &Series,
        mut start: usize,
        mut length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        #[cfg(feature = "dtype-categorical")]
        {
            fill_rev_map(other.dtype(), &mut self.rev_map_merger);
        }

        if length == 0 || repeats == 0 || other.is_empty() {
            return;
        }

        for chunk in other.chunks() {
            if start < chunk.len() {
                let length_in_chunk = length.min(chunk.len() - start);
                self.builder.subslice_extend_each_repeated(
                    &**chunk,
                    start,
                    length_in_chunk,
                    repeats,
                    share,
                );

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

    /// Extends this builder with the contents of the given series at the given
    /// indices. That is, `other[idxs[i]]` is appended to this builder in order,
    /// for each i=0..idxs.len(). May panic if other does not match the dtype
    /// of this builder, or if the other series is not rechunked.
    ///
    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather_extend(&mut self, other: &Series, idxs: &[IdxSize], share: ShareStrategy) {
        #[cfg(feature = "dtype-categorical")]
        {
            fill_rev_map(other.dtype(), &mut self.rev_map_merger);
        }

        let chunks = other.chunks();
        assert!(chunks.len() == 1);
        self.builder.gather_extend(&*chunks[0], idxs, share);
    }

    pub fn opt_gather_extend(&mut self, other: &Series, idxs: &[IdxSize], share: ShareStrategy) {
        #[cfg(feature = "dtype-categorical")]
        {
            fill_rev_map(other.dtype(), &mut self.rev_map_merger);
        }

        let chunks = other.chunks();
        assert!(chunks.len() == 1);
        self.builder.opt_gather_extend(&*chunks[0], idxs, share);
    }
}
