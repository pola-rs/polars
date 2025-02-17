use arrow::array::{GenericBinaryArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::buffer::Buffer;
use arrow::offset::{Offset, Offsets, OffsetsBuffer};
use polars_utils::vec::{CapacityByFactor, PushUnchecked};

use super::Index;

fn create_offsets<I: Iterator<Item = usize>, O: Offset>(
    lengths: I,
    idx_len: usize,
) -> OffsetsBuffer<O> {
    let mut length_so_far = O::default();
    let mut offsets = Vec::with_capacity(idx_len + 1);
    offsets.push(length_so_far);

    for len in lengths {
        unsafe {
            length_so_far += O::from_usize(len).unwrap_unchecked();
            offsets.push_unchecked(length_so_far)
        };
    }
    unsafe { Offsets::new_unchecked(offsets).into() }
}

pub(super) unsafe fn take_values<O: Offset>(
    length: O,
    starts: &[O],
    offsets: &OffsetsBuffer<O>,
    values: &[u8],
) -> Buffer<u8> {
    let new_len = length.to_usize();
    let mut buffer = Vec::with_capacity(new_len);
    starts
        .iter()
        .map(|start| start.to_usize())
        .zip(offsets.lengths())
        .for_each(|(start, length)| {
            let end = start + length;
            buffer.extend_from_slice(values.get_unchecked(start..end));
        });
    buffer.into()
}

// take implementation when neither values nor indices contain nulls
pub(super) unsafe fn take_no_validity_unchecked<O: Offset, I: Index>(
    offsets: &OffsetsBuffer<O>,
    values: &[u8],
    indices: &[I],
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let values_len = offsets.last().to_usize();
    let fraction_estimate = indices.len() as f64 / offsets.len() as f64 + 0.3;
    let mut buffer = Vec::<u8>::with_capacity_by_factor(values_len, fraction_estimate);

    let lengths = indices.iter().map(|index| index.to_usize()).map(|index| {
        let (start, end) = offsets.start_end_unchecked(index);
        buffer.extend_from_slice(values.get_unchecked(start..end));
        end - start
    });
    let offsets = create_offsets(lengths, indices.len());

    (offsets, buffer.into(), None)
}

// take implementation when only values contain nulls
pub(super) unsafe fn take_values_validity<O: Offset, I: Index, A: GenericBinaryArray<O>>(
    values: &A,
    indices: &[I],
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let validity_values = values.validity().unwrap();
    let validity = indices
        .iter()
        .map(|index| validity_values.get_bit_unchecked(index.to_usize()));
    let validity = Bitmap::from_trusted_len_iter(validity);

    let mut total_length = O::default();

    let offsets = values.offsets();
    let values_values = values.values();

    let mut starts = Vec::<O>::with_capacity(indices.len());
    let lengths = indices.iter().map(|index| {
        let index = index.to_usize();
        let start = *offsets.get_unchecked(index);
        let length = *offsets.get_unchecked(index + 1) - start;
        total_length += length;
        starts.push_unchecked(start);
        length.to_usize()
    });
    let offsets = create_offsets(lengths, indices.len());
    let buffer = take_values(total_length, starts.as_slice(), &offsets, values_values);

    (offsets, buffer, validity.into())
}

// take implementation when only indices contain nulls
pub(super) unsafe fn take_indices_validity<O: Offset, I: Index>(
    offsets: &OffsetsBuffer<O>,
    values: &[u8],
    indices: &PrimitiveArray<I>,
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let mut total_length = O::default();

    let offsets = offsets.buffer();

    let mut starts = Vec::<O>::with_capacity(indices.len());
    let lengths = indices.values().iter().map(|index| {
        let index = index.to_usize();
        let length;
        match offsets.get(index + 1) {
            Some(&next) => {
                let start = *offsets.get_unchecked(index);
                length = next - start;
                total_length += length;
                starts.push_unchecked(start);
            },
            None => {
                length = O::zero();
                starts.push_unchecked(O::default());
            },
        };
        length.to_usize()
    });
    let offsets = create_offsets(lengths, indices.len());

    let buffer = take_values(total_length, &starts, &offsets, values);

    (offsets, buffer, indices.validity().cloned())
}

// take implementation when both indices and values contain nulls
pub(super) unsafe fn take_values_indices_validity<O: Offset, I: Index, A: GenericBinaryArray<O>>(
    values: &A,
    indices: &PrimitiveArray<I>,
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let mut total_length = O::default();
    let mut validity = BitmapBuilder::with_capacity(indices.len());

    let values_validity = values.validity().unwrap();
    let offsets = values.offsets();
    let values_values = values.values();

    let mut starts = Vec::<O>::with_capacity(indices.len());
    let lengths = indices.iter().map(|index| {
        let length;
        match index {
            Some(index) => {
                let index = index.to_usize();
                if values_validity.get_bit(index) {
                    validity.push(true);
                    length = *offsets.get_unchecked(index + 1) - *offsets.get_unchecked(index);
                    starts.push_unchecked(*offsets.get_unchecked(index));
                } else {
                    validity.push(false);
                    length = O::zero();
                    starts.push_unchecked(O::default());
                }
            },
            None => {
                validity.push(false);
                length = O::zero();
                starts.push_unchecked(O::default());
            },
        };
        total_length += length;
        length.to_usize()
    });
    let offsets = create_offsets(lengths, indices.len());

    let buffer = take_values(total_length, &starts, &offsets, values_values);

    (offsets, buffer, validity.into_opt_validity())
}
