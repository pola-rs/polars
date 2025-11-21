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

    fn freeze_reset(&mut self) -> Self::Array {
        let values = core::mem::take(&mut self.values).freeze();
        let validity = core::mem::take(&mut self.validity).into_opt_validity();
        BooleanArray::try_new(self.dtype.clone(), values, validity).unwrap()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn extend_nulls(&mut self, length: usize) {
        self.values.extend_constant(length, false);
        self.validity.extend_constant(length, false);
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

    fn subslice_extend_each_repeated(
        &mut self,
        other: &BooleanArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        self.values.subslice_extend_each_repeated_from_bitmap(
            other.values(),
            start,
            length,
            repeats,
        );
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
        other: &BooleanArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.values.reserve(idxs.len());
        for idx in idxs {
            self.values
                .push_unchecked(other.value_unchecked(*idx as usize));
        }
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }

    fn opt_gather_extend(&mut self, other: &BooleanArray, idxs: &[IdxSize], _share: ShareStrategy) {
        self.values.reserve(idxs.len());
        unsafe {
            for idx in idxs {
                let val = if (*idx as usize) < other.len() {
                    // We don't use get here as that double-checks the validity
                    // which we don't care about here.
                    other.value_unchecked(*idx as usize)
                } else {
                    false
                };
                self.values.push_unchecked(val);
            }
        }
        self.validity
            .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
    }
}
