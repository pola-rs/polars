use polars_utils::IdxSize;

use super::ListArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy};
use crate::array::Array;
use crate::bitmap::{BitmapBuilder, OptBitmapBuilder};
use crate::datatypes::ArrowDataType;
use crate::offset::{Offsets, OffsetsBuffer};
use crate::types::Offset;

pub struct ListArrayBuilder<O: Offset, B: ArrayBuilder> {
    dtype: ArrowDataType,
    offsets: Offsets<O>,
    inner_builder: B,
    validity: OptBitmapBuilder,
}

impl<O: Offset, B: ArrayBuilder> ListArrayBuilder<O, B> {
    pub fn new(dtype: ArrowDataType, inner_builder: B) -> Self {
        Self {
            dtype,
            inner_builder,
            offsets: Offsets::new(),
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl<O: Offset, B: ArrayBuilder> ArrayBuilder for ListArrayBuilder<O, B> {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
        // No inner reserve, we have no idea how large it needs to be.
    }

    fn freeze(self) -> Box<dyn Array> {
        let offsets = OffsetsBuffer::from(self.offsets);
        let values = self.inner_builder.freeze();
        let validity = self.validity.into_opt_validity();
        Box::new(ListArray::new(self.dtype, offsets, values, validity))
    }

    fn subslice_extend(&mut self, other: &dyn Array, start: usize, length: usize, share: ShareStrategy) {
        let other: &ListArray<O> = other.as_any().downcast_ref().unwrap();
        let start_offset = other.offsets()[start].to_usize();
        let stop_offset = other.offsets()[start + length].to_usize();
        self.offsets.try_extend_from_slice(other.offsets(), start, length).unwrap();
        self.inner_builder.subslice_extend(&**other.values(), start_offset, stop_offset - start_offset, share);
        self.validity.subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    unsafe fn gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy) {
        let other: &ListArray<O> = other.as_any().downcast_ref().unwrap();
        let other_values = &**other.values();
        let other_offsets = other.offsets();
        
        // Pre-compute proper length for reserve.
        let total_len: usize = idxs.iter().map(|i| {
            let start = other_offsets.get_unchecked(*i as usize).to_usize();
            let stop = other_offsets.get_unchecked(*i as usize + 1).to_usize();
            stop - start
        }).sum();
        self.inner_builder.reserve(total_len);
        
        // Group consecutive indices into larger copies.
        let mut group_start = 0;
        while group_start < idxs.len() {
            let start_idx = idxs[group_start] as usize;
            let mut group_len = 1;
            while group_start + group_len < idxs.len() && idxs[group_start + group_len] as usize == start_idx + group_len {
                group_len += 1;
            }
            
            let start_offset = other_offsets.get_unchecked(start_idx).to_usize();
            let stop_offset = other_offsets.get_unchecked(start_idx + group_len).to_usize();
            self.offsets.try_extend_from_slice(other_offsets, group_start, group_len).unwrap();
            self.inner_builder
                .subslice_extend(other_values, start_offset, stop_offset - start_offset, share);
            group_start += group_len;
        }

        self.validity.gather_extend_from_opt_validity(other.validity(), idxs);
    }
}
