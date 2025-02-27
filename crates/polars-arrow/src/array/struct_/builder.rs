use polars_utils::IdxSize;

use super::StructArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy, StaticArrayBuilder};
use crate::array::Array;
use crate::bitmap::OptBitmapBuilder;
use crate::datatypes::ArrowDataType;

pub struct StructArrayBuilder {
    dtype: ArrowDataType,
    length: usize,
    inner_builders: Vec<Box<dyn ArrayBuilder>>,
    validity: OptBitmapBuilder,
}

impl StructArrayBuilder {
    pub fn new(dtype: ArrowDataType, inner_builders: Vec<Box<dyn ArrayBuilder>>) -> Self {
        Self {
            dtype,
            length: 0,
            inner_builders,
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl StaticArrayBuilder for StructArrayBuilder {
    type Array = StructArray;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        for builder in &mut self.inner_builders {
            builder.reserve(additional);
        }
        self.validity.reserve(additional);
    }

    fn freeze(self) -> StructArray {
        let values = self
            .inner_builders
            .into_iter()
            .map(|b| b.freeze())
            .collect();
        let validity = self.validity.into_opt_validity();
        StructArray::new(self.dtype, self.length, values, validity)
    }

    fn subslice_extend(
        &mut self,
        other: &StructArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        for (builder, other_values) in self.inner_builders.iter_mut().zip(other.values()) {
            builder.subslice_extend(&**other_values, start, length, share);
        }
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
        self.length += length;
    }

    unsafe fn gather_extend(
        &mut self,
        other: &StructArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        for (builder, other_values) in self.inner_builders.iter_mut().zip(other.values()) {
            builder.gather_extend(&**other_values, idxs, share);
        }
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
        self.length += idxs.len();
    }
}
