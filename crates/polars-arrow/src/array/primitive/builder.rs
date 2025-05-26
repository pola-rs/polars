use polars_utils::IdxSize;
use polars_utils::vec::PushUnchecked;

use super::PrimitiveArray;
use crate::array::builder::{ShareStrategy, StaticArrayBuilder};
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

    fn freeze_reset(&mut self) -> Self::Array {
        let values = Buffer::from(core::mem::take(&mut self.values));
        let validity = core::mem::take(&mut self.validity).into_opt_validity();
        PrimitiveArray::new(self.dtype.clone(), values, validity)
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn extend_nulls(&mut self, length: usize) {
        self.values.resize(self.values.len() + length, T::zeroed());
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PrimitiveArray<T>,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        self.values
            .extend_from_slice(&other.values()[start..start + length]);
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PrimitiveArray<T>,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        self.values.reserve(length * repeats);

        for value in other.values()[start..start + length].iter() {
            unsafe {
                for _ in 0..repeats {
                    self.values.push_unchecked(*value);
                }
            }
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
        other: &PrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        // TODO: SIMD gather kernels?
        let other_values_slice = other.values().as_slice();
        self.values.extend(
            idxs.iter()
                .map(|idx| *other_values_slice.get_unchecked(*idx as usize)),
        );
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }

    fn opt_gather_extend(
        &mut self,
        other: &PrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.values.reserve(idxs.len());
        unsafe {
            for idx in idxs {
                let val = if (*idx as usize) < other.len() {
                    other.value_unchecked(*idx as usize)
                } else {
                    T::zeroed()
                };
                self.values.push_unchecked(val);
            }
        }
        self.validity
            .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
    }
}
