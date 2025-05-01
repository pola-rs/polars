use polars_utils::IdxSize;

use super::ListArray;
use crate::array::builder::{ArrayBuilder, ShareStrategy, StaticArrayBuilder};
use crate::bitmap::OptBitmapBuilder;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets, OffsetsBuffer};

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

impl<O: Offset, B: ArrayBuilder> StaticArrayBuilder for ListArrayBuilder<O, B> {
    type Array = ListArray<O>;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
        // No inner reserve, we have no idea how large it needs to be.
    }

    fn freeze(self) -> ListArray<O> {
        let offsets = OffsetsBuffer::from(self.offsets);
        let values = self.inner_builder.freeze();
        let validity = self.validity.into_opt_validity();
        ListArray::new(self.dtype, offsets, values, validity)
    }

    fn freeze_reset(&mut self) -> Self::Array {
        let offsets = OffsetsBuffer::from(core::mem::take(&mut self.offsets));
        let values = self.inner_builder.freeze_reset();
        let validity = core::mem::take(&mut self.validity).into_opt_validity();
        ListArray::new(self.dtype.clone(), offsets, values, validity)
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
        other: &ListArray<O>,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        let start_offset = other.offsets()[start].to_usize();
        let stop_offset = other.offsets()[start + length].to_usize();
        self.offsets
            .try_extend_from_slice(other.offsets(), start, length)
            .unwrap();
        self.inner_builder.subslice_extend(
            &**other.values(),
            start_offset,
            stop_offset - start_offset,
            share,
        );
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &ListArray<O>,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let other_offsets = other.offsets();
        let other_values = &**other.values();

        let start_offset = other.offsets()[start].to_usize();
        let stop_offset = other.offsets()[start + length].to_usize();
        self.offsets.reserve(length * repeats);
        self.inner_builder
            .reserve((stop_offset - start_offset) * repeats);
        for offset_idx in start..start + length {
            let sublist_start = other_offsets[offset_idx].to_usize();
            let sublist_stop = other_offsets[offset_idx + 1].to_usize();
            for _ in 0..repeats {
                self.offsets.try_push(sublist_stop - sublist_start).unwrap();
            }
            self.inner_builder.subslice_extend_repeated(
                other_values,
                sublist_start,
                sublist_stop - sublist_start,
                repeats,
                share,
            );
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
        other: &ListArray<O>,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        let other_values = &**other.values();
        let other_offsets = other.offsets();

        // Pre-compute proper length for reserve.
        let total_len: usize = idxs
            .iter()
            .map(|i| {
                let start = other_offsets.get_unchecked(*i as usize).to_usize();
                let stop = other_offsets.get_unchecked(*i as usize + 1).to_usize();
                stop - start
            })
            .sum();
        self.inner_builder.reserve(total_len);

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

            let start_offset = other_offsets.get_unchecked(start_idx).to_usize();
            let stop_offset = other_offsets
                .get_unchecked(start_idx + group_len)
                .to_usize();
            self.offsets
                .try_extend_from_slice(other_offsets, start_idx, group_len)
                .unwrap();
            self.inner_builder.subslice_extend(
                other_values,
                start_offset,
                stop_offset - start_offset,
                share,
            );
            group_start += group_len;
        }

        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }

    fn opt_gather_extend(&mut self, other: &ListArray<O>, idxs: &[IdxSize], share: ShareStrategy) {
        let other_values = &**other.values();
        let other_offsets = other.offsets();

        unsafe {
            // Pre-compute proper length for reserve.
            let total_len: usize = idxs
                .iter()
                .map(|idx| {
                    if (*idx as usize) < other.len() {
                        let start = other_offsets.get_unchecked(*idx as usize).to_usize();
                        let stop = other_offsets.get_unchecked(*idx as usize + 1).to_usize();
                        stop - start
                    } else {
                        0
                    }
                })
                .sum();
            self.inner_builder.reserve(total_len);

            // Group consecutive indices into larger copies.
            let mut group_start = 0;
            while group_start < idxs.len() {
                let start_idx = idxs[group_start] as usize;
                let mut group_len = 1;
                let in_bounds = start_idx < other.len();

                if in_bounds {
                    while group_start + group_len < idxs.len()
                        && idxs[group_start + group_len] as usize == start_idx + group_len
                        && start_idx + group_len < other.len()
                    {
                        group_len += 1;
                    }

                    let start_offset = other_offsets.get_unchecked(start_idx).to_usize();
                    let stop_offset = other_offsets
                        .get_unchecked(start_idx + group_len)
                        .to_usize();
                    self.offsets
                        .try_extend_from_slice(other_offsets, start_idx, group_len)
                        .unwrap();
                    self.inner_builder.subslice_extend(
                        other_values,
                        start_offset,
                        stop_offset - start_offset,
                        share,
                    );
                } else {
                    while group_start + group_len < idxs.len()
                        && idxs[group_start + group_len] as usize >= other.len()
                    {
                        group_len += 1;
                    }
                    self.offsets.extend_constant(group_len);
                }
                group_start += group_len;
            }

            self.validity
                .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
        }
    }
}
