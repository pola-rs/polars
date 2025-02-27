use polars_utils::vec::PushUnchecked;
use polars_utils::IdxSize;

use super::PrimitiveArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy, StaticArrayBuilder};
use crate::array::Array;
use crate::bitmap::OptBitmapBuilder;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::types::NativeType;

pub struct PrimitiveArrayBuilder<T> {
    dtype: ArrowDataType,
    values: Vec<T>,
    validity: OptBitmapBuilder,
}

impl<T: NativeType> PrimitiveArrayBuilder<T> {
    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            dtype,
            values: Vec::new(),
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl<T: NativeType> StaticArrayBuilder for PrimitiveArrayBuilder<T> {
    type Array = PrimitiveArray<T>;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> PrimitiveArray<T> {
        let values = Buffer::from(self.values);
        let validity = self.validity.into_opt_validity();
        PrimitiveArray::new(self.dtype, values, validity)
    }

    fn subslice_extend(
        &mut self,
        other: &PrimitiveArray<T>,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        let other: &PrimitiveArray<T> = other.as_any().downcast_ref().unwrap();
        self.values
            .extend_from_slice(&other.values()[start..start + length]);
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    unsafe fn gather_extend(
        &mut self,
        other: &PrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        let other: &PrimitiveArray<T> = other.as_any().downcast_ref().unwrap();
        self.values.reserve(idxs.len());
        for idx in idxs {
            self.values
                .push_unchecked(other.value_unchecked(*idx as usize));
        }
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }
}
