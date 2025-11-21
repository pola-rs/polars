use polars_utils::IdxSize;

use crate::array::BinaryArray;
use crate::array::builder::{ShareStrategy, StaticArrayBuilder};
use crate::bitmap::OptBitmapBuilder;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets, OffsetsBuffer};

pub struct BinaryArrayBuilder<O: Offset> {
    dtype: ArrowDataType,
    offsets: Offsets<O>,
    values: Vec<u8>,
    validity: OptBitmapBuilder,
}

impl<O: Offset> BinaryArrayBuilder<O> {
    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            dtype,
            offsets: Offsets::new(),
            values: Vec::new(),
            validity: OptBitmapBuilder::default(),
        }
    }
}

impl<O: Offset> StaticArrayBuilder for BinaryArrayBuilder<O> {
    type Array = BinaryArray<O>;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
        // No values reserve, we have no idea how large it needs to be.
    }

    fn freeze(self) -> BinaryArray<O> {
        let offsets = OffsetsBuffer::from(self.offsets);
        let values = Buffer::from(self.values);
        let validity = self.validity.into_opt_validity();
        BinaryArray::new(self.dtype, offsets, values, validity)
    }

    fn freeze_reset(&mut self) -> Self::Array {
        let offsets = OffsetsBuffer::from(core::mem::take(&mut self.offsets));
        let values = Buffer::from(core::mem::take(&mut self.values));
        let validity = core::mem::take(&mut self.validity).into_opt_validity();
        BinaryArray::new(self.dtype.clone(), offsets, values, validity)
    }

    fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    fn extend_nulls(&mut self, length: usize) {
        self.offsets.extend_constant(length);
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &BinaryArray<O>,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        let start_offset = other.offsets()[start].to_usize();
        let stop_offset = other.offsets()[start + length].to_usize();
        self.offsets
            .try_extend_from_slice(other.offsets(), start, length)
            .unwrap();
        self.values
            .extend_from_slice(&other.values()[start_offset..stop_offset]);
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &BinaryArray<O>,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        let other_offsets = other.offsets();
        let other_values = &**other.values();

        let start_offset = other.offsets()[start].to_usize();
        let stop_offset = other.offsets()[start + length].to_usize();
        self.offsets.reserve(length * repeats);
        self.values.reserve((stop_offset - start_offset) * repeats);
        for offset_idx in start..start + length {
            let substring_start = other_offsets[offset_idx].to_usize();
            let substring_stop = other_offsets[offset_idx + 1].to_usize();
            for _ in 0..repeats {
                self.offsets
                    .try_push(substring_stop - substring_start)
                    .unwrap();
                self.values
                    .extend_from_slice(&other_values[substring_start..substring_stop]);
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
        other: &BinaryArray<O>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        let other_values = &**other.values();
        let other_offsets = other.offsets();

        // Pre-compute proper length for reserve.
        let total_len: usize = idxs
            .iter()
            .map(|i| {
                let start_offset = other_offsets.get_unchecked(*i as usize).to_usize();
                let stop_offset = other_offsets.get_unchecked(*i as usize + 1).to_usize();
                stop_offset - start_offset
            })
            .sum();
        self.values.reserve(total_len);

        for idx in idxs {
            let start_offset = other_offsets.get_unchecked(*idx as usize).to_usize();
            let stop_offset = other_offsets.get_unchecked(*idx as usize + 1).to_usize();
            self.values
                .extend_from_slice(other_values.get_unchecked(start_offset..stop_offset));
        }

        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }

    fn opt_gather_extend(
        &mut self,
        other: &BinaryArray<O>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        let other_values = &**other.values();
        let other_offsets = other.offsets();

        unsafe {
            // Pre-compute proper length for reserve.
            let total_len: usize = idxs
                .iter()
                .map(|idx| {
                    if (*idx as usize) < other.len() {
                        let start_offset = other_offsets.get_unchecked(*idx as usize).to_usize();
                        let stop_offset = other_offsets.get_unchecked(*idx as usize + 1).to_usize();
                        stop_offset - start_offset
                    } else {
                        0
                    }
                })
                .sum();
            self.values.reserve(total_len);

            for idx in idxs {
                let start_offset = other_offsets.get_unchecked(*idx as usize).to_usize();
                let stop_offset = other_offsets.get_unchecked(*idx as usize + 1).to_usize();
                self.values
                    .extend_from_slice(other_values.get_unchecked(start_offset..stop_offset));
            }

            self.validity
                .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
        }
    }
}
