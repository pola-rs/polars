use polars_utils::IdxSize;

use super::BooleanArray;
use crate::array::builder::{ShareStrategy, StaticArrayBuilder};
use crate::bitmap::{BitmapBuilder, OptBitmapBuilder};
use crate::datatypes::ArrowDataType;

pub struct BooleanArrayBuilder {
    dtype: ArrowDataType,
    values: BitmapBuilder,
    validity: OptBitmapBuilder,
}

impl BooleanArrayBuilder {
    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            dtype,
            values: BitmapBuilder::new(),
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl StaticArrayBuilder for BooleanArrayBuilder {
    type Array = BooleanArray;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> BooleanArray {
        let values = self.values.freeze();
        let validity = self.validity.into_opt_validity();
        BooleanArray::try_new(self.dtype, values, validity).unwrap()
    }

    fn subslice_extend(
        &mut self,
        other: &BooleanArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        self.values
            .subslice_extend_from_bitmap(other.values(), start, length);
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    unsafe fn gather_extend(&mut self, other: &BooleanArray, idxs: &[IdxSize], _share: ShareStrategy) {
        self.values.reserve(idxs.len());
        for idx in idxs {
            self.values
                .push_unchecked(other.value_unchecked(*idx as usize));
        }
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }
}
