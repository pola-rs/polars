use polars_utils::IdxSize;

use super::FixedSizeBinaryArray;
use crate::array::builder::{ShareStrategy, StaticArrayBuilder};
use crate::bitmap::OptBitmapBuilder;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::pushable::Pushable;

pub struct FixedSizeBinaryArrayBuilder {
    dtype: ArrowDataType,
    size: usize,
    length: usize,
    values: Vec<u8>,
    validity: OptBitmapBuilder,
}

impl FixedSizeBinaryArrayBuilder {
    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            size: FixedSizeBinaryArray::get_size(&dtype),
            length: 0,
            dtype,
            values: Vec::new(),
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl StaticArrayBuilder for FixedSizeBinaryArrayBuilder {
    type Array = FixedSizeBinaryArray;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        let bytes = additional * self.size;
        self.values.reserve(bytes);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> FixedSizeBinaryArray {
        // TODO: FixedSizeBinaryArray should track its own length to be correct
        // for size-0 inner.
        let values = Buffer::from(self.values);
        let validity = self.validity.into_opt_validity();
        FixedSizeBinaryArray::new(self.dtype, values, validity)
    }

    fn freeze_reset(&mut self) -> Self::Array {
        // TODO: FixedSizeBinaryArray should track its own length to be correct
        // for size-0 inner.
        let values = Buffer::from(core::mem::take(&mut self.values));
        let validity = core::mem::take(&mut self.validity).into_opt_validity();
        let out = FixedSizeBinaryArray::new(self.dtype.clone(), values, validity);
        self.length = 0;
        out
    }

    fn len(&self) -> usize {
        self.length
    }

    fn extend_nulls(&mut self, length: usize) {
        self.values.extend_constant(length * self.size, 0);
        self.validity.extend_constant(length, false);
        self.length += length;
    }

    fn subslice_extend(
        &mut self,
        other: &FixedSizeBinaryArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        let other_slice = other.values().as_slice();
        self.values
            .extend_from_slice(&other_slice[start * self.size..(start + length) * self.size]);
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
        self.length += length.min(other.len().saturating_sub(start));
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &FixedSizeBinaryArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        let other_slice = other.values().as_slice();
        for outer_idx in start..start + length {
            for _ in 0..repeats {
                self.values.extend_from_slice(
                    &other_slice[outer_idx * self.size..(outer_idx + 1) * self.size],
                );
            }
        }

        self.validity
            .subslice_extend_each_repeated_from_opt_validity(
                other.validity(),
                start,
                length,
                repeats,
            );
        self.length += repeats * length.min(other.len().saturating_sub(start));
    }

    unsafe fn gather_extend(
        &mut self,
        other: &FixedSizeBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        let other_slice = other.values().as_slice();
        self.values.reserve(idxs.len() * self.size);
        for idx in idxs {
            let idx = *idx as usize;
            let subslice = other_slice.get_unchecked(idx * self.size..(idx + 1) * self.size);
            self.values.extend_from_slice(subslice);
        }
        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
        self.length += idxs.len();
    }

    fn opt_gather_extend(
        &mut self,
        other: &FixedSizeBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        let other_slice = other.values().as_slice();
        self.values.reserve(idxs.len() * self.size);
        for idx in idxs {
            let idx = *idx as usize;
            if let Some(subslice) = other_slice.get(idx * self.size..(idx + 1) * self.size) {
                self.values.extend_from_slice(subslice);
            } else {
                self.values.extend_constant(self.size, 0);
            }
        }
        self.validity
            .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
        self.length += idxs.len();
    }
}
