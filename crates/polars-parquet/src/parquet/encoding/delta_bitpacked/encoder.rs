use crate::encoding::ceil8;

use super::super::bitpacked;
use super::super::uleb128;
use super::super::zigzag_leb128;

/// Encodes an iterator of `i64` according to parquet's `DELTA_BINARY_PACKED`.
/// # Implementation
/// * This function does not allocate on the heap.
/// * The number of mini-blocks is always 1. This may change in the future.
pub fn encode<I: Iterator<Item = i64>>(mut iterator: I, buffer: &mut Vec<u8>) {
    let block_size = 128;
    let mini_blocks = 1;

    let mut container = [0u8; 10];
    let encoded_len = uleb128::encode(block_size, &mut container);
    buffer.extend_from_slice(&container[..encoded_len]);

    let encoded_len = uleb128::encode(mini_blocks, &mut container);
    buffer.extend_from_slice(&container[..encoded_len]);

    let length = iterator.size_hint().1.unwrap();
    let encoded_len = uleb128::encode(length as u64, &mut container);
    buffer.extend_from_slice(&container[..encoded_len]);

    let mut values = [0i64; 128];
    let mut deltas = [0u64; 128];

    let first_value = iterator.next().unwrap_or_default();
    let (container, encoded_len) = zigzag_leb128::encode(first_value);
    buffer.extend_from_slice(&container[..encoded_len]);

    let mut prev = first_value;
    let mut length = iterator.size_hint().1.unwrap();
    while length != 0 {
        let mut min_delta = i64::MAX;
        let mut max_delta = i64::MIN;
        let mut num_bits = 0;
        for (i, integer) in (0..128).zip(&mut iterator) {
            let delta = integer - prev;
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);

            num_bits = 64 - (max_delta - min_delta).leading_zeros();
            values[i] = delta;
            prev = integer;
        }
        let consumed = std::cmp::min(length - iterator.size_hint().1.unwrap(), 128);
        length = iterator.size_hint().1.unwrap();
        let values = &values[..consumed];

        values.iter().zip(deltas.iter_mut()).for_each(|(v, delta)| {
            *delta = (v - min_delta) as u64;
        });

        // <min delta> <list of bitwidths of miniblocks> <miniblocks>
        let (container, encoded_len) = zigzag_leb128::encode(min_delta);
        buffer.extend_from_slice(&container[..encoded_len]);

        // one miniblock => 1 byte
        buffer.push(num_bits as u8);
        write_miniblock(buffer, num_bits as usize, deltas);
    }
}

fn write_miniblock(buffer: &mut Vec<u8>, num_bits: usize, deltas: [u64; 128]) {
    if num_bits > 0 {
        let start = buffer.len();

        // bitpack encode all (deltas.len = 128 which is a multiple of 32)
        let bytes_needed = start + ceil8(deltas.len() * num_bits);
        buffer.resize(bytes_needed, 0);
        bitpacked::encode(deltas.as_ref(), num_bits, &mut buffer[start..]);

        let bytes_needed = start + ceil8(deltas.len() * num_bits);
        buffer.truncate(bytes_needed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_delta() {
        // header: [128, 1, 1, 5, 2]:
        //  block size: 128    <=u> 128, 1
        //  mini-blocks: 1     <=u> 1
        //  elements: 5        <=u> 5
        //  first_value: 2     <=z> 1
        // block1: [2, 0, 0, 0, 0]
        //  min_delta: 1        <=z> 2
        //  bitwidth: 0
        let data = 1..=5;
        let expected = vec![128u8, 1, 1, 5, 2, 2, 0];

        let mut buffer = vec![];
        encode(data, &mut buffer);
        assert_eq!(expected, buffer);
    }

    #[test]
    fn negative_min_delta() {
        // max - min = 1 - -4 = 5
        let data = vec![1, 2, 3, 4, 5, 1];
        // header: [128, 1, 4, 6, 2]
        //  block size: 128    <=u> 128, 1
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
        let mut expected = vec![128u8, 1, 1, 6, 2, 7, 3, 0b01101101, 0b00001011];
        expected.extend(std::iter::repeat(0).take(128 * 3 / 8 - 2)); // 128 values, 3 bits, 2 already used

        let mut buffer = vec![];
        encode(data.into_iter(), &mut buffer);
        assert_eq!(expected, buffer);
    }
}
