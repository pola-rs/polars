use super::super::{bitpacked, uleb128, zigzag_leb128};
use crate::parquet::encoding::ceil8;

/// Encodes an iterator of `i64` according to parquet's `DELTA_BINARY_PACKED`.
/// # Implementation
/// * This function does not allocate on the heap.
/// * The number of mini-blocks is always 1. This may change in the future.
pub fn encode<I: ExactSizeIterator<Item = i64>>(
    mut iterator: I,
    buffer: &mut Vec<u8>,
    num_miniblocks_per_block: usize,
) {
    const BLOCK_SIZE: usize = 256;
    assert!([1, 2, 4].contains(&num_miniblocks_per_block));
    let values_per_miniblock = BLOCK_SIZE / num_miniblocks_per_block;

    let mut container = [0u8; 10];
    let encoded_len = uleb128::encode(BLOCK_SIZE as u64, &mut container);
    buffer.extend_from_slice(&container[..encoded_len]);

    let encoded_len = uleb128::encode(num_miniblocks_per_block as u64, &mut container);
    buffer.extend_from_slice(&container[..encoded_len]);

    let length = iterator.len();
    let encoded_len = uleb128::encode(length as u64, &mut container);
    buffer.extend_from_slice(&container[..encoded_len]);

    let mut values = [0i64; BLOCK_SIZE];
    let mut deltas = [0u64; BLOCK_SIZE];
    let mut num_bits = [0u8; 4];

    let first_value = iterator.next().unwrap_or_default();
    let (container, encoded_len) = zigzag_leb128::encode(first_value);
    buffer.extend_from_slice(&container[..encoded_len]);

    let mut prev = first_value;
    let mut length = iterator.len();
    while length != 0 {
        let mut min_delta = i64::MAX;
        let mut max_delta = i64::MIN;
        for (i, integer) in iterator.by_ref().enumerate().take(BLOCK_SIZE) {
            if i % values_per_miniblock == 0 {
                min_delta = i64::MAX;
                max_delta = i64::MIN
            }

            let delta = integer.wrapping_sub(prev);
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);

            let miniblock_idx = i / values_per_miniblock;
            num_bits[miniblock_idx] = (64 - max_delta.abs_diff(min_delta).leading_zeros()) as u8;
            values[i] = delta;
            prev = integer;
        }
        let consumed = std::cmp::min(length - iterator.len(), BLOCK_SIZE);
        length = iterator.len();
        let values = &values[..consumed];

        values.iter().zip(deltas.iter_mut()).for_each(|(v, delta)| {
            *delta = v.wrapping_sub(min_delta) as u64;
        });

        // <min delta> <list of bitwidths of miniblocks> <miniblocks>
        let (container, encoded_len) = zigzag_leb128::encode(min_delta);
        buffer.extend_from_slice(&container[..encoded_len]);

        // one miniblock => 1 byte
        let mut values_remaining = consumed;
        buffer.extend_from_slice(&num_bits[..num_miniblocks_per_block]);
        for i in 0..num_miniblocks_per_block {
            if values_remaining == 0 {
                break;
            }

            values_remaining = values_remaining.saturating_sub(values_per_miniblock);
            write_miniblock(
                buffer,
                num_bits[i],
                &deltas[i * values_per_miniblock..(i + 1) * values_per_miniblock],
            );
        }
    }
}

fn write_miniblock(buffer: &mut Vec<u8>, num_bits: u8, deltas: &[u64]) {
    let num_bits = num_bits as usize;
    if num_bits > 0 {
        let start = buffer.len();

        // bitpack encode all (deltas.len = 128 which is a multiple of 32)
        let bytes_needed = start + ceil8(deltas.len() * num_bits);
        buffer.resize(bytes_needed, 0);
        bitpacked::encode(deltas, num_bits, &mut buffer[start..]);

        let bytes_needed = start + ceil8(deltas.len() * num_bits);
        buffer.truncate(bytes_needed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_delta() {
        // header: [128, 2, 1, 5, 2]:
        //  block size: 256    <=u> 128, 2
        //  mini-blocks: 1     <=u> 1
        //  elements: 5        <=u> 5
        //  first_value: 2     <=z> 1
        // block1: [2, 0, 0, 0, 0]
        //  min_delta: 1        <=z> 2
        //  bitwidth: 0
        let data = 1..=5;
        let expected = vec![128u8, 2, 1, 5, 2, 2, 0];

        let mut buffer = vec![];
        encode(data.collect::<Vec<_>>().into_iter(), &mut buffer, 1);
        assert_eq!(expected, buffer);
    }

    #[test]
    fn negative_min_delta() {
        // max - min = 1 - -4 = 5
        let data = vec![1, 2, 3, 4, 5, 1];
        // header: [128, 2, 4, 6, 2]
        //  block size: 256    <=u> 128, 2
        //  mini-blocks: 1     <=u> 1
        //  elements: 6        <=u> 5
        //  first_value: 2     <=z> 1
        // block1: [7, 3, 253, 255]
        //  min_delta: -4        <=z> 7
        //  bitwidth: 3
        //  values: [5, 5, 5, 5, 0] <=b> [
        //      0b01101101
        //      0b00001011
        // ]
        let mut expected = vec![128u8, 2, 1, 6, 2, 7, 3, 0b01101101, 0b00001011];
        expected.extend(std::iter::repeat(0).take(256 * 3 / 8 - 2)); // 128 values, 3 bits, 2 already used

        let mut buffer = vec![];
        encode(data.into_iter(), &mut buffer, 1);
        assert_eq!(expected, buffer);
    }
}
