use polars_utils::IdxSize;

use super::FixedSizeListArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy, StaticArrayBuilder};
use crate::bitmap::OptBitmapBuilder;
use crate::datatypes::ArrowDataType;

pub struct FixedSizeListArrayBuilder<B: ArrayBuilder> {
    dtype: ArrowDataType,
    size: usize,
    length: usize,
    inner_builder: B,
    validity: OptBitmapBuilder,
}
impl<B: ArrayBuilder> FixedSizeListArrayBuilder<B> {
    pub fn new(dtype: ArrowDataType, inner_builder: B) -> Self {
        Self {
            size: FixedSizeListArray::get_child_and_size(&dtype).1,
            dtype,
            length: 0,
            inner_builder,
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl<B: ArrayBuilder> StaticArrayBuilder for FixedSizeListArrayBuilder<B> {
    type Array = FixedSizeListArray;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.inner_builder.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> FixedSizeListArray {
        let values = self.inner_builder.freeze();
        let validity = self.validity.into_opt_validity();
        FixedSizeListArray::new(self.dtype, self.length, values, validity)
    }

    fn freeze_reset(&mut self) -> Self::Array {
        let values = self.inner_builder.freeze_reset();
        let validity = core::mem::take(&mut self.validity).into_opt_validity();
        let out = FixedSizeListArray::new(self.dtype.clone(), self.length, values, validity);
        self.length = 0;
        out
    }

    fn len(&self) -> usize {
        self.length
    }

    fn extend_nulls(&mut self, length: usize) {
        self.inner_builder.extend_nulls(length * self.size);
        self.validity.extend_constant(length, false);
        self.length += length;
    }

    fn subslice_extend(
        &mut self,
        other: &FixedSizeListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.inner_builder.subslice_extend(
            &**other.values(),
            start * self.size,
            length * self.size,
            share,
        );
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
        self.length += length.min(other.len().saturating_sub(start));
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &FixedSizeListArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let other_values = &**other.values();
        self.inner_builder.reserve(repeats * length * self.size);
        for outer_idx in start..start + length {
            self.inner_builder.subslice_extend_repeated(
                other_values,
                outer_idx * self.size,
                self.size,
                repeats,
                share,
            );
        }
        self.validity
            .subslice_extend_each_repeated_from_opt_validity(
                other.validity(),
                start,
                length,
                repeats,
            );
    }

    unsafe fn gather_extend(
        &mut self,
        other: &FixedSizeListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        let other_values = &**other.values();
        self.inner_builder.reserve(idxs.len() * self.size);

        // Group consecutive indices into larger copies.
        let mut group_start = 0;
        while group_start < idxs.len() {
            let start_idx = idxs[group_start] as usize;
            let mut group_len = 1;
            while group_start + group_len < idxs.len()
                && idxs[group_start + group_len] as usize == start_idx + group_len
            {
                group_len += 1;
            }
            self.inner_builder.subslice_extend(
                other_values,
                start_idx * self.size,
                group_len * self.size,
                share,
            );
            group_start += group_len;
        }

        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
        self.length += idxs.len();
    }

    fn opt_gather_extend(
        &mut self,
        other: &FixedSizeListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        let other_values = &**other.values();
        self.inner_builder.reserve(idxs.len() * self.size);

        // Group consecutive indices into larger copies.
        let mut group_start = 0;
        while group_start < idxs.len() {
            let start_idx = idxs[group_start] as usize;
            let mut group_len = 1;
            let in_bounds = start_idx < other.len();

            if in_bounds {
                while group_start + group_len < idxs.len()
                    && idxs[group_start + group_len] as usize == start_idx + group_len
                    && start_idx + group_len < other.len()
                {
                    group_len += 1;
                }

                self.inner_builder.subslice_extend(
                    other_values,
                    start_idx * self.size,
                    group_len * self.size,
                    share,
                );
            } else {
                while group_start + group_len < idxs.len()
                    && idxs[group_start + group_len] as usize >= other.len()
                {
                    group_len += 1;
                }

                self.inner_builder.extend_nulls(group_len * self.size);
            }
            group_start += group_len;
        }

        self.validity
            .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
        self.length += idxs.len();
    }
}
