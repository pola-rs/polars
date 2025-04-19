use super::super::delta_bitpacked;
use crate::parquet::encoding::delta_length_byte_array;

/// Encodes an iterator of according to DELTA_BYTE_ARRAY
pub fn encode<'a, I: ExactSizeIterator<Item = &'a [u8]> + Clone>(
    iterator: I,
    buffer: &mut Vec<u8>,
) {
    let mut previous = b"".as_ref();

    let mut sum_lengths = 0;
    let prefixes = iterator
        .clone()
        .map(|item| {
            let prefix_length = item
                .iter()
                .zip(previous.iter())
                .enumerate()
                // find first difference
                .find_map(|(length, (lhs, rhs))| (lhs != rhs).then_some(length))
                .unwrap_or(previous.len());
            previous = item;

            sum_lengths += item.len() - prefix_length;
            prefix_length as i64
        })
        .collect::<Vec<_>>();
    delta_bitpacked::encode(prefixes.iter().copied(), buffer, 1);

    let remaining = iterator
        .zip(prefixes)
        .map(|(item, prefix)| &item[prefix as usize..]);

    delta_length_byte_array::encode(remaining, buffer);
}
