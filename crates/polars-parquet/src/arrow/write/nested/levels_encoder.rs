//! Hybrid RLE encoding of a slice of levels.

use super::dremel::Level;
use crate::parquet::encoding::uleb128;

/// Values in a group of a bit-packed run.
const GROUP_LEN: usize = 8;

/// Maximum number of values in a bit-packed run. Some readers decode a whole run at once.
const MAX_BITPACKED_LEN: usize = 1024 * GROUP_LEN;

/// Writes `levels` with the hybrid RLE encoding using `num_bits` bits per value.
///
/// Runs of equal values are written as a RLE run if bit-packing them takes at least 3 bytes, which
/// is about the cost of a RLE run. All other values are bit-packed.
pub fn encode<T: Level>(buffer: &mut Vec<u8>, levels: &[T], num_bits: usize) {
    debug_assert!((1..=16).contains(&num_bits));
    let min_run_len = (24 / num_bits).max(GROUP_LEN);
    let mut literal_start = 0;
    let mut i = 0;
    while i < levels.len() {
        let value = levels[i];
        let run_len = run_length(&levels[i..], value);
        if run_len >= min_run_len {
            write_bitpacked(buffer, &levels[literal_start..i], num_bits);
            write_rle(buffer, run_len, value, num_bits);
            i += run_len;
            literal_start = i;
        } else {
            i = (i + GROUP_LEN).min(levels.len());
            if i - literal_start == MAX_BITPACKED_LEN {
                write_bitpacked(buffer, &levels[literal_start..i], num_bits);
                literal_start = i;
            }
        }
    }
    write_bitpacked(buffer, &levels[literal_start..], num_bits);
}

/// Returns the number of leading values equal to `value`.
#[inline]
fn run_length<T: Level>(values: &[T], value: T) -> usize {
    const CHUNK_LEN: usize = 32;
    let mut len = 0;
    for chunk in values.chunks_exact(CHUNK_LEN) {
        if !chunk.iter().fold(true, |all_eq, &v| all_eq & (v == value)) {
            break;
        }
        len += CHUNK_LEN;
    }
    len + values[len..]
        .iter()
        .position(|&v| v != value)
        .unwrap_or(values.len() - len)
}

fn write_header(buffer: &mut Vec<u8>, header: u64) {
    let mut container = [0; 10];
    let used = uleb128::encode(header, &mut container);
    buffer.extend_from_slice(&container[..used]);
}

fn write_rle<T: Level>(buffer: &mut Vec<u8>, run_len: usize, value: T, num_bits: usize) {
    write_header(buffer, (run_len as u64) << 1);
    let value: u64 = value.into();
    buffer.extend_from_slice(&value.to_le_bytes()[..num_bits.div_ceil(8)]);
}

/// Writes a bit-packed run. Only the last run of a page may have a length that is not a multiple
/// of [`GROUP_LEN`], it is padded with zeros.
fn write_bitpacked<T: Level>(buffer: &mut Vec<u8>, values: &[T], num_bits: usize) {
    if values.is_empty() {
        return;
    }
    let num_groups = values.len().div_ceil(GROUP_LEN);
    write_header(buffer, ((num_groups as u64) << 1) | 1);
    buffer.reserve(num_groups * num_bits);

    let groups = values.chunks_exact(GROUP_LEN);
    let remainder = groups.remainder();
    for group in groups {
        write_group(buffer, group.try_into().unwrap(), num_bits);
    }
    if !remainder.is_empty() {
        let mut group = [T::default(); GROUP_LEN];
        group[..remainder.len()].copy_from_slice(remainder);
        write_group(buffer, &group, num_bits);
    }
}

/// Bit-packs 8 values into `num_bits` bytes.
#[inline]
fn write_group<T: Level>(buffer: &mut Vec<u8>, group: &[T; GROUP_LEN], num_bits: usize) {
    if num_bits <= 8 {
        let mut packed = 0u64;
        for (i, &v) in group.iter().enumerate() {
            packed |= v.into() << (i * num_bits);
        }
        buffer.extend_from_slice(&packed.to_le_bytes()[..num_bits]);
    } else {
        let mut packed = 0u128;
        for (i, &v) in group.iter().enumerate() {
            packed |= (v.into() as u128) << (i * num_bits);
        }
        buffer.extend_from_slice(&packed.to_le_bytes()[..num_bits]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;

    fn roundtrip<T: Level>(levels: &[T], num_bits: usize) {
        let mut buffer = vec![];
        encode(&mut buffer, levels, num_bits);
        let decoded = HybridRleDecoder::new(&buffer, num_bits as u32, levels.len())
            .collect()
            .unwrap();
        let expected = levels.iter().map(|&l| l.into() as u32).collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn rle_and_bitpacked() {
        let mut buffer = vec![];
        encode::<u8>(&mut buffer, &[1; 24], 1);
        assert_eq!(buffer, vec![24 << 1, 1]);

        let mut buffer = vec![];
        encode::<u8>(&mut buffer, &[1; 23], 1);
        assert_eq!(buffer, vec![(3 << 1) | 1, 0xFF, 0xFF, 0x7F]);

        let mut buffer = vec![];
        encode::<u8>(&mut buffer, &[0, 1, 2, 1, 2, 1, 1, 0, 3], 2);
        assert_eq!(
            buffer,
            vec![(2 << 1) | 1, 0b01_10_01_00, 0b00_01_01_10, 0b11, 0]
        );
    }

    #[test]
    fn roundtrip_patterns() {
        let mut state = 7u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as usize
        };
        for len in [0, 1, 7, 8, 9, 15, 16, 17, 33, 100, 1000, 20000] {
            for num_bits in [1, 2, 3, 5, 8, 9, 13, 16] {
                let max = (1usize << num_bits) - 1;
                // Mix of runs of random length and random values.
                let mut levels = Vec::with_capacity(len);
                while levels.len() < len {
                    let value = next() % (max + 1);
                    let run = if next() % 2 == 0 { 1 } else { next() % 50 };
                    levels.extend(std::iter::repeat_n(value, run.min(len - levels.len())));
                }
                if num_bits <= 8 {
                    roundtrip(
                        &levels.iter().map(|&l| l as u8).collect::<Vec<_>>(),
                        num_bits,
                    );
                }
                roundtrip(
                    &levels.iter().map(|&l| l as u16).collect::<Vec<_>>(),
                    num_bits,
                );
            }
        }
    }
}
