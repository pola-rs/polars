use crate::{
    array::{GenericBinaryArray, PrimitiveArray},
    bitmap::{Bitmap, MutableBitmap},
    buffer::Buffer,
    offset::{Offset, Offsets, OffsetsBuffer},
};

use super::Index;

pub fn take_values<O: Offset>(
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
            buffer.extend_from_slice(&values[start..end]);
        });
    buffer.into()
}

// take implementation when neither values nor indices contain nulls
pub fn take_no_validity<O: Offset, I: Index>(
    offsets: &OffsetsBuffer<O>,
    values: &[u8],
    indices: &[I],
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let mut buffer = Vec::<u8>::new();
    let lengths = indices.iter().map(|index| index.to_usize()).map(|index| {
        let (start, end) = offsets.start_end(index);
        // todo: remove this bound check
        buffer.extend_from_slice(&values[start..end]);
        end - start
    });
    let offsets = Offsets::try_from_lengths(lengths).expect("");

    (offsets.into(), buffer.into(), None)
}

// take implementation when only values contain nulls
pub fn take_values_validity<O: Offset, I: Index, A: GenericBinaryArray<O>>(
    values: &A,
    indices: &[I],
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let validity_values = values.validity().unwrap();
    let validity = indices
        .iter()
        .map(|index| validity_values.get_bit(index.to_usize()));
    let validity = Bitmap::from_trusted_len_iter(validity);

    let mut length = O::default();

    let offsets = values.offsets();
    let values_values = values.values();

    let mut starts = Vec::<O>::with_capacity(indices.len());
    let offsets = indices.iter().map(|index| {
        let index = index.to_usize();
        let start = offsets[index];
        length += offsets[index + 1] - start;
        starts.push(start);
        length
    });
    let offsets = std::iter::once(O::default())
        .chain(offsets)
        .collect::<Vec<_>>();
    // Safety: by construction offsets are monotonically increasing
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();

    let buffer = take_values(length, starts.as_slice(), &offsets, values_values);

    (offsets, buffer, validity.into())
}

// take implementation when only indices contain nulls
pub fn take_indices_validity<O: Offset, I: Index>(
    offsets: &OffsetsBuffer<O>,
    values: &[u8],
    indices: &PrimitiveArray<I>,
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let mut length = O::default();

    let offsets = offsets.buffer();

    let mut starts = Vec::<O>::with_capacity(indices.len());
    let offsets = indices.values().iter().map(|index| {
        let index = index.to_usize();
        match offsets.get(index + 1) {
            Some(&next) => {
                let start = offsets[index];
                length += next - start;
                starts.push(start);
            }
            None => starts.push(O::default()),
        };
        length
    });
    let offsets = std::iter::once(O::default())
        .chain(offsets)
        .collect::<Vec<_>>();
    // Safety: by construction offsets are monotonically increasing
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();

    let buffer = take_values(length, &starts, &offsets, values);

    (offsets, buffer, indices.validity().cloned())
}

// take implementation when both indices and values contain nulls
pub fn take_values_indices_validity<O: Offset, I: Index, A: GenericBinaryArray<O>>(
    values: &A,
    indices: &PrimitiveArray<I>,
) -> (OffsetsBuffer<O>, Buffer<u8>, Option<Bitmap>) {
    let mut length = O::default();
    let mut validity = MutableBitmap::with_capacity(indices.len());

    let values_validity = values.validity().unwrap();
    let offsets = values.offsets();
    let values_values = values.values();

    let mut starts = Vec::<O>::with_capacity(indices.len());
    let offsets = indices.iter().map(|index| {
        match index {
            Some(index) => {
                let index = index.to_usize();
                if values_validity.get_bit(index) {
                    validity.push(true);
                    length += offsets[index + 1] - offsets[index];
                    starts.push(offsets[index]);
                } else {
                    validity.push(false);
                    starts.push(O::default());
                }
            }
            None => {
                validity.push(false);
                starts.push(O::default());
            }
        };
        length
    });
    let offsets = std::iter::once(O::default())
        .chain(offsets)
        .collect::<Vec<_>>();
    // Safety: by construction offsets are monotonically increasing
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();

    let buffer = take_values(length, &starts, &offsets, values_values);

    (offsets, buffer, validity.into())
}
