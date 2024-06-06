use std::io::Write;

use super::bitpacked_encode;
use crate::parquet::encoding::{bitpacked, ceil8, uleb128};

// Arbitrary value that balances memory usage and storage overhead
const MAX_VALUES_PER_LITERAL_RUN: usize = (1 << 10) * 8;

pub trait Encoder<T: PartialEq + Default + Copy> {
    fn bitpacked_encode<W: Write, I: Iterator<Item = T>>(
        writer: &mut W,
        iterator: I,
        num_bits: usize,
    ) -> std::io::Result<()>;

    fn run_length_encode<W: Write>(
        writer: &mut W,
        run_length: usize,
        value: T,
        bit_width: u32,
    ) -> std::io::Result<()>;
}

const U32_BLOCK_LEN: usize = 32;

impl Encoder<u32> for u32 {
    fn bitpacked_encode<W: Write, I: Iterator<Item = u32>>(
        writer: &mut W,
        mut iterator: I,
        num_bits: usize,
    ) -> std::io::Result<()> {
        // the length of the iterator.
        let length = iterator.size_hint().1.unwrap();

        let mut header = ceil8(length) as u64;
        header <<= 1;
        header |= 1; // it is bitpacked => first bit is set
        let mut container = [0; 10];
        let used = uleb128::encode(header, &mut container);
        writer.write_all(&container[..used])?;

        let chunks = length / U32_BLOCK_LEN;
        let remainder = length - chunks * U32_BLOCK_LEN;
        let mut buffer = [0u32; U32_BLOCK_LEN];

        // simplified from ceil8(U32_BLOCK_LEN * num_bits) since U32_BLOCK_LEN = 32
        let compressed_chunk_size = 4 * num_bits;

        for _ in 0..chunks {
            iterator
                .by_ref()
                .take(U32_BLOCK_LEN)
                .zip(buffer.iter_mut())
                .for_each(|(item, buf)| *buf = item);

            let mut packed = [0u8; 4 * U32_BLOCK_LEN];
            bitpacked::encode_pack::<u32>(&buffer, num_bits, packed.as_mut());
            writer.write_all(&packed[..compressed_chunk_size])?;
        }

        if remainder != 0 {
            // Must be careful here to ensure we write a multiple of `num_bits`
            // (the bit width) to align with the spec. Some readers also rely on
            // this - see https://github.com/pola-rs/polars/pull/13883.

            // this is ceil8(remainder * num_bits), but we ensure the output is a
            // multiple of num_bits by rewriting it as ceil8(remainder) * num_bits
            let compressed_remainder_size = ceil8(remainder) * num_bits;
            iterator
                .by_ref()
                .take(remainder)
                .zip(buffer.iter_mut())
                .for_each(|(item, buf)| *buf = item);

            let mut packed = [0u8; 4 * U32_BLOCK_LEN];
            // No need to zero rest of buffer because remainder is either:
            // * Multiple of 8: We pad non-terminal literal runs to have a
            //   multiple of 8 values. Once compressed, the data will end on
            //   clean byte boundaries and packed[..compressed_remainder_size]
            //   will include only the remainder values and nothing extra.
            // * Final run: Extra values from buffer will be included in
            //   packed[..compressed_remainder_size] but ignored when decoding
            //   because they extend beyond known column length
            bitpacked::encode_pack(&buffer, num_bits, packed.as_mut());
            writer.write_all(&packed[..compressed_remainder_size])?;
        };
        Ok(())
    }

    fn run_length_encode<W: Write>(
        writer: &mut W,
        run_length: usize,
        value: u32,
        bit_width: u32,
    ) -> std::io::Result<()> {
        // write the length + indicator
        let mut header = run_length as u64;
        header <<= 1;
        let mut container = [0; 10];
        let used = uleb128::encode(header, &mut container);
        writer.write_all(&container[..used])?;

        let num_bytes = ceil8(bit_width as usize);
        let bytes = value.to_le_bytes();
        writer.write_all(&bytes[..num_bytes])?;
        Ok(())
    }
}

impl Encoder<bool> for bool {
    fn bitpacked_encode<W: Write, I: Iterator<Item = bool>>(
        writer: &mut W,
        iterator: I,
        _num_bits: usize,
    ) -> std::io::Result<()> {
        // the length of the iterator.
        let length = iterator.size_hint().1.unwrap();

        let mut header = ceil8(length) as u64;
        header <<= 1;
        header |= 1; // it is bitpacked => first bit is set
        let mut container = [0; 10];
        let used = uleb128::encode(header, &mut container);
        writer.write_all(&container[..used])?;
        bitpacked_encode(writer, iterator)?;
        Ok(())
    }

    fn run_length_encode<W: Write>(
        writer: &mut W,
        run_length: usize,
        value: bool,
        _bit_width: u32,
    ) -> std::io::Result<()> {
        // write the length + indicator
        let mut header = run_length as u64;
        header <<= 1;
        let mut container = [0; 10];
        let used = uleb128::encode(header, &mut container);
        writer.write_all(&container[..used])?;
        writer.write_all(&(value as u8).to_le_bytes())?;
        Ok(())
    }
}

#[allow(clippy::comparison_chain)]
pub fn encode<T: PartialEq + Default + Copy + Encoder<T>, W: Write, I: Iterator<Item = T>>(
    writer: &mut W,
    iterator: I,
    num_bits: u32,
) -> std::io::Result<()> {
    let mut consecutive_repeats: usize = 0;
    let mut previous_val = T::default();
    let mut buffered_bits = [previous_val; MAX_VALUES_PER_LITERAL_RUN];
    let mut buffer_idx = 0;
    let mut literal_run_idx = 0;
    for val in iterator {
        if val == previous_val {
            consecutive_repeats += 1;
            if consecutive_repeats >= 8 {
                // Run is long enough to RLE, no need to buffer values
                if consecutive_repeats > 8 {
                    continue;
                } else {
                    // When we encounter a run long enough to potentially RLE,
                    // we must first ensure that the buffered literal run has
                    // a multiple of 8 values for bit-packing. If not, we pad
                    // up by taking some of the consecutive repeats
                    let literal_padding = (8 - (literal_run_idx % 8)) % 8;
                    consecutive_repeats -= literal_padding;
                    literal_run_idx += literal_padding;
                }
            }
            // Too short to RLE, continue to buffer values
        } else if consecutive_repeats > 8 {
            // Value changed so start a new run but the current run is long
            // enough to RLE. First, bit-pack any buffered literal run. Then,
            // RLE current run and reset consecutive repeat counter and buffer.
            if literal_run_idx > 0 {
                debug_assert!(literal_run_idx % 8 == 0);
                T::bitpacked_encode(
                    writer,
                    buffered_bits.iter().take(literal_run_idx).copied(),
                    num_bits as usize,
                )?;
                literal_run_idx = 0;
            }
            T::run_length_encode(writer, consecutive_repeats, previous_val, num_bits)?;
            consecutive_repeats = 1;
            buffer_idx = 0;
        } else {
            // Value changed so start a new run but the current run is not long
            // enough to RLE. Consolidate all consecutive repeats into buffered
            // literal run.
            literal_run_idx = buffer_idx;
            consecutive_repeats = 1;
        }
        // If buffer is full, bit-pack as literal run and reset
        if buffer_idx == MAX_VALUES_PER_LITERAL_RUN {
            T::bitpacked_encode(writer, buffered_bits.iter().copied(), num_bits as usize)?;
            // If buffer fills up in the middle of a run, all but the last
            // repeat is consolidated into the literal run.
            debug_assert!(
                (consecutive_repeats < 8)
                    && (buffer_idx - literal_run_idx == consecutive_repeats - 1)
            );
            consecutive_repeats = 1;
            buffer_idx = 0;
            literal_run_idx = 0;
        }
        buffered_bits[buffer_idx] = val;
        previous_val = val;
        buffer_idx += 1;
    }
    // Final run not long enough to RLE, extend literal run.
    if consecutive_repeats <= 8 {
        literal_run_idx = buffer_idx;
    }
    // Bit-pack final buffered literal run, if any
    if literal_run_idx > 0 {
        T::bitpacked_encode(
            writer,
            buffered_bits.iter().take(literal_run_idx).copied(),
            num_bits as usize,
        )?;
    }
    // RLE final consecutive run if long enough
    if consecutive_repeats > 8 {
        T::run_length_encode(writer, consecutive_repeats, previous_val, num_bits)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::bitmap::BitmapIter;
    use super::*;

    #[test]
    fn bool_basics_1() -> std::io::Result<()> {
        let iter = BitmapIter::new(&[0b10011101u8, 0b10011101], 0, 14);

        let mut vec = vec![];

        encode::<bool, _, _>(&mut vec, iter, 1)?;

        assert_eq!(vec, vec![(2 << 1 | 1), 0b10011101u8, 0b00011101]);

        Ok(())
    }

    #[test]
    fn bool_from_iter() -> std::io::Result<()> {
        let mut vec = vec![];

        encode::<bool, _, _>(
            &mut vec,
            vec![true, true, true, true, true, true, true, true].into_iter(),
            1,
        )?;

        assert_eq!(vec, vec![(1 << 1 | 1), 0b11111111]);
        Ok(())
    }

    #[test]
    fn test_encode_u32() -> std::io::Result<()> {
        let mut vec = vec![];

        encode::<u32, _, _>(&mut vec, vec![0, 1, 2, 1, 2, 1, 1, 0, 3].into_iter(), 2)?;

        assert_eq!(
            vec,
            vec![
                (2 << 1 | 1),
                0b01_10_01_00,
                0b00_01_01_10,
                0b_00_00_00_11,
                0b0
            ]
        );
        Ok(())
    }

    #[test]
    fn test_encode_u32_large() -> std::io::Result<()> {
        let mut vec = vec![];

        let values = (0..128).map(|x| x % 4);

        encode::<u32, _, _>(&mut vec, values, 2)?;

        let length = 128;
        let expected = 0b11_10_01_00u8;

        let mut expected = vec![expected; length / 4];
        expected.insert(0, ((length / 8) as u8) << 1 | 1);

        assert_eq!(vec, expected);
        Ok(())
    }

    #[test]
    fn test_u32_other() -> std::io::Result<()> {
        let values = vec![3, 3, 0, 3, 2, 3, 3, 3, 3, 1, 3, 3, 3, 0, 3].into_iter();

        let mut vec = vec![];
        encode::<u32, _, _>(&mut vec, values, 2)?;

        let expected = vec![5, 207, 254, 247, 51];
        assert_eq!(expected, vec);
        Ok(())
    }
}
