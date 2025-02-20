use polars_utils::IdxSize;

use super::FixedSizeBinaryArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy};
use crate::array::Array;
use crate::bitmap::OptBitmapBuilder;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

pub struct FixedSizeBinaryArrayBuilder {
    dtype: ArrowDataType,
    values: Vec<u8>,
    validity: OptBitmapBuilder,
}

impl FixedSizeBinaryArrayBuilder {
    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            dtype,
            values: Vec::new(),
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl ArrayBuilder for FixedSizeBinaryArrayBuilder {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        let bytes = additional * FixedSizeBinaryArray::get_size(&self.dtype);
        self.values.reserve(bytes);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> Box<dyn Array> {
        let values = Buffer::from(self.values);
        let validity = self.validity.into_opt_validity();
        Box::new(FixedSizeBinaryArray::new(self.dtype, values, validity))
    }

    fn subslice_extend(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        let other: &FixedSizeBinaryArray = other.as_any().downcast_ref().unwrap();
        let other_slice = other.values().as_slice();
        let size = FixedSizeBinaryArray::get_size(&self.dtype);
        self.values
            .extend_from_slice(&other_slice[start * size..(start + length) * size]);
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    unsafe fn gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], _share: ShareStrategy) {
        let other: &FixedSizeBinaryArray = other.as_any().downcast_ref().unwrap();
        let other_slice = other.values().as_slice();
        let size = FixedSizeBinaryArray::get_size(&self.dtype);
        self.values.reserve(idxs.len() * size);
        for idx in idxs {
            let idx = *idx as usize;
            let subslice = other_slice.get_unchecked(idx * size..(idx + 1) * size);
            self.values.extend_from_slice(subslice);
        }
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }
}
