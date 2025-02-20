use polars_utils::IdxSize;

use super::FixedSizeListArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy};
use crate::array::Array;
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

impl<B: ArrayBuilder> ArrayBuilder for FixedSizeListArrayBuilder<B> {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.inner_builder.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> Box<dyn Array> {
        let values = self.inner_builder.freeze();
        let validity = self.validity.into_opt_validity();
        Box::new(FixedSizeListArray::new(
            self.dtype,
            self.length,
            values,
            validity,
        ))
    }

    fn subslice_extend(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        let other: &FixedSizeListArray = other.as_any().downcast_ref().unwrap();
        self.inner_builder.subslice_extend(
            &**other.values(),
            start * self.size,
            length * self.size,
            share,
        );
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
        self.length += length;
    }

    unsafe fn gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy) {
        let other: &FixedSizeListArray = other.as_any().downcast_ref().unwrap();
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
}
