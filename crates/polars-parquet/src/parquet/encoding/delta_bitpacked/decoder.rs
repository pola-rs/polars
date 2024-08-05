use super::super::{bitpacked, uleb128, zigzag_leb128};
use crate::parquet::error::{ParquetError, ParquetResult};

/// An [`Iterator`] of [`i64`]
#[derive(Debug)]
struct Block<'a> {
    /// this is the minimum delta that must be added to every value.
    min_delta: i64,
    bitwidths: &'a [u8],
    current_miniblock: bitpacked::DecoderIter<'a, u64>,
    remainder: &'a [u8],
}

impl<'a> Default for Block<'a> {
    fn default() -> Self {
        Self {
            min_delta: 0,
            bitwidths: &[],
            current_miniblock: bitpacked::DecoderIter::new(&[], 0, 0).unwrap(),
            remainder: &[],
        }
    }
}

fn collect_miniblock(
    target: &'_ mut Vec<i64>,
    min_delta: i64,
    bitwidth: u8,
    values: &'_ [u8],
    values_per_miniblock: usize,
    last_value: &mut i64,
) {
    let bitwidth = bitwidth as usize;

    debug_assert!(bitwidth <= 64);
    debug_assert!((bitwidth * values_per_miniblock).div_ceil(8) == values.len());

    let start_length = target.len();

    // SAFETY: u64 and i64 have the same size, alignment and they are both Copy.
    let target = unsafe { std::mem::transmute::<&mut Vec<i64>, &mut Vec<u64>>(target) };
    bitpacked::Decoder::new(values, bitwidth, values_per_miniblock).collect_into(target);
    let target_length = target.len();

    debug_assert_eq!(target_length - start_length, values_per_miniblock);

    // @TODO: This should include the last value
    // Offset everything by the delta value.
    &mut target[target_length - values_per_miniblock..]
        .iter_mut()
        .for_each(|v| {
            *last_value = *last_value + (*v as i64) + min_delta;
            *v = *last_value as u64;
        });
}

fn sum_miniblock(
    min_delta: i64,
    bitwidth: u8,
    values: &[u8],
    values_per_miniblock: usize,
    last_value: &mut i64,
) -> i64 {
    let bitwidth = bitwidth as usize;

    debug_assert!(bitwidth <= 64);
    debug_assert!((bitwidth * values_per_miniblock).div_ceil(8) == values.len());

    let mut iter = bitpacked::Decoder::<u64>::new(values, bitwidth, values_per_miniblock);

    // @TODO: This should include the last value
    let mut accum = 0i64;
    for chunk in iter.chunked() {
        accum += chunk
            .into_iter()
            .map(|v| {
                *last_value = *last_value + (v as i64) + min_delta;
                *last_value
            })
            .sum::<i64>();
    }

    if let Some((chunk, length)) = iter.chunked().next_inexact() {
        accum += chunk[..length]
            .into_iter()
            .map(|&v| {
                *last_value = *last_value + (v as i64) + min_delta;
                *last_value
            })
            .sum::<i64>();
    }

    accum
}

fn collect_block(
    target: &'_ mut Vec<i64>,
    num_miniblocks: usize,
    values_per_miniblock: usize,
    mut values: &'_ [u8],
    last_value: &mut i64,
) {
    let (min_delta, consumed) = zigzag_leb128::decode(values);
    values = &values[consumed..];
    let bitwidths;
    (bitwidths, values) = values.split_at(num_miniblocks);

    target.reserve(num_miniblocks * values_per_miniblock);
    for &bitwidth in bitwidths {
        let miniblock;
        (miniblock, values) =
            values.split_at((bitwidth as usize * values_per_miniblock).div_ceil(8));
        collect_miniblock(
            target,
            min_delta,
            bitwidth,
            miniblock,
            values_per_miniblock,
            last_value,
        );
    }
}

fn sum_block(
    num_miniblocks: usize,
    values_per_miniblock: usize,
    mut values: &'_ [u8],

    last_value: &mut i64,
) -> i64 {
    let (min_delta, consumed) = zigzag_leb128::decode(values);
    values = &values[consumed..];
    let bitwidths;
    (bitwidths, values) = values.split_at(num_miniblocks);

    let mut accum = 0i64;
    for &bitwidth in bitwidths {
        let miniblock;
        (miniblock, values) =
            values.split_at((bitwidth as usize * values_per_miniblock).div_ceil(8));
        accum += sum_miniblock(
            min_delta,
            bitwidth,
            miniblock,
            values_per_miniblock,
            last_value,
        );
    }
    accum
}

impl<'a> Block<'a> {
    fn new(
        mut values: &'a [u8],
        num_miniblocks: usize,
        values_per_miniblock: usize,
        length: usize,
    ) -> ParquetResult<Self> {
        let length = std::cmp::min(length, num_miniblocks * values_per_miniblock);
        let actual_num_miniblocks =
            usize::min(num_miniblocks, length.div_ceil(values_per_miniblock));

        if actual_num_miniblocks == 0 {
            return Ok(Self::default());
        }

        let (min_delta, consumed) = zigzag_leb128::decode(values);
        values = &values[consumed..];
        let (bitwidths, remainder) = values.split_at(num_miniblocks);

        let bitwidths = &bitwidths[..actual_num_miniblocks];
        let first_bitwidth = bitwidths[0] as usize;

        let num_bytes = (first_bitwidth * values_per_miniblock).div_ceil(8);
        let (bytes, remainder) = remainder.split_at(num_bytes);
        let current_miniblock =
            bitpacked::DecoderIter::new(bytes, first_bitwidth, values_per_miniblock)?;

        Ok(Block {
            min_delta,
            bitwidths,
            current_miniblock,
            remainder,
        })
    }
}

/// Decoder of parquets' `DELTA_BINARY_PACKED`. Implements `Iterator<Item = i64>`.
/// # Implementation
/// This struct does not allocate on the heap.
#[derive(Debug)]
pub struct Decoder<'a> {
    num_miniblocks_per_block: usize,
    values_per_block: usize,

    values_remaining: usize,

    first_value: i64,
    values: &'a [u8],
    current_block: Block<'a>,
    // the total number of bytes consumed up to a given point, excluding the bytes on the current_block
    consumed_bytes: usize,
}

impl<'a> Decoder<'a> {
    pub fn try_new(mut values: &'a [u8]) -> ParquetResult<(Self, &'a [u8])> {
        let header_err = || ParquetError::oos("Insufficient bytes for Delta encoding header");

        let mut consumed_bytes = 0;

        // header:
        // <block size in values> <number of miniblocks in a block> <total value count> <first value>

        let (values_per_block, consumed) = uleb128::decode(values);
        let values_per_block = values_per_block as usize;
        consumed_bytes += consumed;
        values = values.get(consumed..).ok_or_else(header_err)?;

        assert_eq!(values_per_block % 128, 0);

        let (num_miniblocks_per_block, consumed) = uleb128::decode(values);
        let num_miniblocks_per_block = num_miniblocks_per_block as usize;
        consumed_bytes += consumed;
        values = values.get(consumed..).ok_or_else(header_err)?;

        let (total_count, consumed) = uleb128::decode(values);
        let total_count = total_count as usize;
        consumed_bytes += consumed;
        values = values.get(consumed..).ok_or_else(header_err)?;

        let (first_value, consumed) = zigzag_leb128::decode(values);
        consumed_bytes += consumed;
        values = values.get(consumed..).ok_or_else(header_err)?;

        assert_eq!(values_per_block % num_miniblocks_per_block, 0);
        assert_eq!((values_per_block / num_miniblocks_per_block) % 32, 0);

        let values_per_miniblock = values_per_block / num_miniblocks_per_block;
        assert_eq!(values_per_miniblock % 8, 0);

        // If we only have one value (first_value), there are no blocks.
        let current_block = if total_count > 1 {
            Some(Block::try_new(
                values,
                num_miniblocks_per_block,
                values_per_miniblock,
                total_count - 1,
            )?)
        } else {
            None
        };

        // We skip over all the values to determine where the slice stops.
        let remainder = if total_count > 0 {
            let mut rem = values;
            let mut num_values_read = total_count - 1;
            while num_values_read > 0 {
                // If the number of values is does not need all the miniblocks anymore, we need to
                // ignore the later miniblocks and regard them as having bitwidth = 0.
                //
                // Quoted from the specification:
                //
                // > If, in the last block, less than <number of miniblocks in a block> miniblocks
                // > are needed to store the values, the bytes storing the bit widths of the
                // > unneeded miniblocks are still present, their value should be zero, but readers
                // > must accept arbitrary values as well. There are no additional padding bytes for
                // > the miniblock bodies though, as if their bit widths were 0 (regardless of the
                // > actual byte values). The reader knows when to stop reading by keeping track of
                // > the number of values read.
                let num_remaining_mini_blocks = usize::min(
                    num_miniblocks_per_block,
                    num_values_read.div_ceil(values_per_miniblock),
                );

                // block:
                // <min delta> <list of bitwidths of miniblocks> <miniblocks>

                let (_, consumed) = zigzag_leb128::decode(values);
                rem = rem.get(consumed..).ok_or(ParquetError::oos(
                    "No min-delta value in delta encoding miniblock",
                ))?;

                if rem[..num_remaining_mini_blocks]
                    .iter()
                    .copied()
                    .any(|bitwidth| bitwidth > 64)
                {
                    return Err(ParquetError::oos(
                        "Delta encoding miniblock with bitwidth higher than maximum 64 bits",
                    ));
                }

                let num_bitpacking_bytes = rem[..num_remaining_mini_blocks]
                    .iter()
                    .copied()
                    .map(|bitwidth| (bitwidth as usize * values_per_miniblock).div_ceil(8))
                    .sum::<usize>();

                rem = rem
                    .get(num_miniblocks_per_block + num_bitpacking_bytes..)
                    .ok_or(ParquetError::oos(
                        "Not enough bytes for all bitpacked values in delta encoding",
                    ))?;

                num_values_read = num_values_read.saturating_sub(values_per_block as usize);
            }
            rem
        } else {
            values
        };

        Ok((
            Self {
                num_miniblocks_per_block,
                values_per_block,
                values_remaining: total_count,
                first_value,
                values: &values[..values.len() - remainder.len()],
                current_block,
                consumed_bytes,
            },
            remainder,
        ))
    }

    pub fn skip_in_place(&mut self, n: usize) {}

    /// Returns the total number of bytes consumed up to this point by [`Decoder`].
    pub fn consumed_bytes(&self) -> usize {
        self.consumed_bytes + self.current_block.as_ref().map_or(0, |b| b.consumed_bytes)
    }

    fn load_delta(&mut self) -> Result<i64, ParquetError> {
        // At this point we must have at least one block and value available
        let current_block = self.current_block.as_mut().unwrap();
        if let Some(x) = current_block.next() {
            x
        } else {
            // load next block
            self.values = &self.values[current_block.consumed_bytes..];
            self.consumed_bytes += current_block.consumed_bytes;

            let next_block = Block::try_new(
                self.values,
                self.num_miniblocks_per_block,
                self.values_per_block / self.num_miniblocks_per_block,
                self.values_remaining,
            );
            match next_block {
                Ok(mut next_block) => {
                    let delta = next_block
                        .next()
                        .ok_or_else(|| ParquetError::oos("Missing block"))?;
                    self.current_block = Some(next_block);
                    delta
                },
                Err(e) => Err(e),
            }
        }
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = Result<i64, ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.values_remaining == 0 {
            return None;
        }

        let result = Some(Ok(self.first_value));

        self.values_remaining -= 1;
        if self.values_remaining == 0 {
            // do not try to load another block
            return result;
        }

        let delta = match self.load_delta() {
            Ok(delta) => delta,
            Err(e) => return Some(Err(e)),
        };

        self.first_value += delta;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.values_remaining, Some(self.values_remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_value() {
        // Generated by parquet-rs
        //
        // header: [128, 1, 4, 1, 2]
        // block size: 128, 1
        // mini-blocks: 4
        // elements: 1
        // first_value: 2 <=z> 1
        let data = &[128, 1, 4, 1, 2];

        let (mut decoder, _) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(&r[..], &[1]);
        assert_eq!(decoder.consumed_bytes(), 5);
    }

    #[test]
    fn test_from_spec() {
        let expected = (1..=5).collect::<Vec<_>>();
        // VALIDATED FROM SPARK==3.1.1
        // header: [128, 1, 4, 5, 2]
        // block size: 128, 1
        // mini-blocks: 4
        // elements: 5
        // first_value: 2 <=z> 1
        // block1: [2, 0, 0, 0, 0]
        // min_delta: 2 <=z> 1
        // bit_width: 0
        let data = &[128, 1, 4, 5, 2, 2, 0, 0, 0, 0];

        let (mut decoder, _) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(expected, r);

        assert_eq!(decoder.consumed_bytes(), 10);
    }

    #[test]
    fn case2() {
        let expected = vec![1, 2, 3, 4, 5, 1];
        // VALIDATED FROM SPARK==3.1.1
        // header: [128, 1, 4, 6, 2]
        // block size: 128, 1 <=u> 128
        // mini-blocks: 4     <=u> 4
        // elements: 6        <=u> 6
        // first_value: 2     <=z> 1
        // block1: [7, 3, 0, 0, 0]
        // min_delta: 7       <=z> -4
        // bit_widths: [3, 0, 0, 0]
        // values: [
        //      0b01101101
        //      0b00001011
        //      ...
        // ]                  <=b> [3, 3, 3, 3, 0]
        let data = &[
            128, 1, 4, 6, 2, 7, 3, 0, 0, 0, 0b01101101, 0b00001011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // these should not be consumed
            1, 2, 3,
        ];

        let (mut decoder, remainder) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(expected, r);
        assert_eq!(decoder.consumed_bytes(), data.len() - 3);
        assert_eq!(remainder.len(), 3);
    }

    #[test]
    fn multiple_miniblocks() {
        #[rustfmt::skip]
        let data = &[
            // Header: [128, 1, 4, 65, 100]
            128, 1, // block size <=u> 128
            4,      // number of mini-blocks <=u> 4
            65,     // number of elements <=u> 65
            100,    // first_value <=z> 50

            // Block 1 header: [7, 3, 4, 0, 0]
            7,            // min_delta <=z> -4
            3, 4, 255, 0, // bit_widths (255 should not be used as only two miniblocks are needed)

            // 32 3-bit values of 0 for mini-block 1 (12 bytes)
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            // 32 4-bit values of 8 for mini-block 2 (16 bytes)
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
            0x88, 0x88,

            // these should not be consumed
            1, 2, 3,
        ];

        #[rustfmt::skip]
        let expected = [
            // First value
            50,

            // Mini-block 1: 32 deltas of -4
            46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2, -2, -6, -10, -14, -18, -22, -26, -30, -34,
            -38, -42, -46, -50, -54, -58, -62, -66, -70, -74, -78,

            // Mini-block 2: 32 deltas of 4
            -74, -70, -66, -62, -58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6,
            -2, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50,
        ];

        let (mut decoder, remainder) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(&expected[..], &r[..]);
        assert_eq!(decoder.consumed_bytes(), data.len() - 3);
        assert_eq!(remainder.len(), 3);
    }
}
