use super::super::{bitpacked, uleb128, zigzag_leb128};
use crate::parquet::encoding::bitpacked::{Unpackable, Unpacked};
use crate::parquet::error::{ParquetError, ParquetResult};

const MAX_BITWIDTH: u8 = 64;

pub trait DeltaGatherer {
    type Target;

    fn target_len(&self, target: &Self::Target) -> usize;
    fn target_reserve(&self, target: &mut Self::Target, n: usize);

    fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()>;
    fn gather_constant(
        &mut self,
        target: &mut Self::Target,
        v: i64,
        delta: i64,
        num_repeats: usize,
    ) -> ParquetResult<()> {
        for i in 0..num_repeats {
            self.gather_one(target, v + (i as i64) * delta)?;
        }
        Ok(())
    }
    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        for &v in slice {
            self.gather_one(target, v)?;
        }
        Ok(())
    }
    fn gather_chunk(&mut self, target: &mut Self::Target, chunk: &[i64; 64]) -> ParquetResult<()> {
        self.gather_slice(target, chunk)
    }
}

struct SkipGatherer;

impl DeltaGatherer for SkipGatherer {
    type Target = usize;

    fn target_len(&self, target: &Self::Target) -> usize {
        *target
    }
    fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

    fn gather_one(&mut self, target: &mut Self::Target, _v: i64) -> ParquetResult<()> {
        *target += 1;
        Ok(())
    }
    fn gather_chunk(&mut self, target: &mut Self::Target, chunk: &[i64; 64]) -> ParquetResult<()> {
        *target += chunk.len();
        Ok(())
    }
    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        *target += slice.len();
        Ok(())
    }
}

/// An [`Iterator`] of [`i64`]
#[derive(Debug, Clone)]
struct Block<'a> {
    /// this is the minimum delta that must be added to every value.
    min_delta: i64,
    bitwidths: &'a [u8],
    current_miniblock: bitpacked::DecoderIter<'a, u64>,
    remainder: &'a [u8],
    num_values_remaining: usize,
}

impl<'a> Default for Block<'a> {
    fn default() -> Self {
        Self {
            min_delta: 0,
            bitwidths: &[],
            current_miniblock: bitpacked::DecoderIter::new(&[], 0, 0).unwrap(),
            remainder: &[],
            num_values_remaining: 0,
        }
    }
}

impl<'a> bitpacked::DecoderIter<'a, u64> {
    fn gather_n_into<G: DeltaGatherer>(
        &mut self,
        target: &mut G::Target,
        min_delta: i64,
        next_value: &mut i64,
        mut n: usize,
        gatherer: &mut G,
    ) -> ParquetResult<()> {
        debug_assert!(n > 0);
        debug_assert!(self.len() >= n);

        if self.decoder.num_bits() == 0 {
            let num_repeats = usize::min(self.len(), n);
            let v = *next_value;
            gatherer.gather_constant(target, v, min_delta, num_repeats)?;
            *next_value = v.wrapping_add(min_delta * num_repeats as i64);
            self.decoder.length -= num_repeats;
            return Ok(());
        }

        if self.unpacked_start < self.unpacked_end {
            let length = usize::min(n, self.unpacked_end - self.unpacked_start);
            self.buffered[self.unpacked_start..self.unpacked_start + length]
                .iter_mut()
                .for_each(|v| {
                    let value = *next_value;
                    *next_value = next_value.wrapping_add(*v as i64).wrapping_add(min_delta);
                    *v = value as u64;
                });
            gatherer.gather_slice(
                target,
                bytemuck::cast_slice(
                    &self.buffered[self.unpacked_start..self.unpacked_start + length],
                ),
            )?;
            n -= length;
            self.unpacked_start += length;
        }

        if n == 0 {
            return Ok(());
        }

        const ITEMS_PER_PACK: usize = <<u64 as Unpackable>::Unpacked as Unpacked<u64>>::LENGTH;
        for _ in 0..n / ITEMS_PER_PACK {
            let mut chunk = self.decoder.chunked().next().unwrap();
            chunk.iter_mut().for_each(|v| {
                let value = *next_value;
                *next_value = next_value.wrapping_add(*v as i64).wrapping_add(min_delta);
                *v = value as u64;
            });
            gatherer.gather_chunk(target, bytemuck::cast_ref(&chunk))?;
            n -= ITEMS_PER_PACK;
        }

        if n == 0 {
            return Ok(());
        }

        let Some((chunk, len)) = self.decoder.chunked().next_inexact() else {
            debug_assert_eq!(n, 0);
            self.buffered = <u64 as Unpackable>::Unpacked::zero();
            self.unpacked_start = 0;
            self.unpacked_end = 0;
            return Ok(());
        };

        self.buffered = chunk;
        self.unpacked_start = 0;
        self.unpacked_end = len;

        if n > 0 {
            let length = usize::min(n, self.unpacked_end);
            self.buffered[..length].iter_mut().for_each(|v| {
                let value = *next_value;
                *next_value = next_value.wrapping_add(*v as i64).wrapping_add(min_delta);
                *v = value as u64;
            });
            gatherer.gather_slice(target, bytemuck::cast_slice(&self.buffered[..length]))?;
            self.unpacked_start = length;
        }

        Ok(())
    }
}

fn gather_bitpacked<G: DeltaGatherer>(
    target: &mut G::Target,
    min_delta: i64,
    next_value: &mut i64,
    mut decoder: bitpacked::Decoder<u64>,
    gatherer: &mut G,
) -> ParquetResult<()> {
    let mut chunked = decoder.chunked();
    for mut chunk in &mut chunked {
        for value in &mut chunk {
            let v = *value;
            *value = *next_value as u64;
            *next_value += (v as i64) + min_delta;
        }

        let chunk = bytemuck::cast_ref(&chunk);
        gatherer.gather_chunk(target, chunk)?;
    }

    if let Some((mut chunk, length)) = chunked.next_inexact() {
        let slice = &mut chunk[..length];

        for value in slice.iter_mut() {
            let v = *value;
            *value = *next_value as u64;
            *next_value += (v as i64) + min_delta;
        }

        let slice = bytemuck::cast_slice(slice);
        gatherer.gather_slice(target, slice)?;
    }

    Ok(())
}

fn gather_miniblock<G: DeltaGatherer>(
    target: &mut G::Target,
    min_delta: i64,
    bitwidth: u8,
    values: &[u8],
    values_per_miniblock: usize,
    next_value: &mut i64,
    gatherer: &mut G,
) -> ParquetResult<()> {
    let bitwidth = bitwidth as usize;

    debug_assert!(bitwidth <= 64);
    debug_assert_eq!((bitwidth * values_per_miniblock).div_ceil(8), values.len());

    let start_length = gatherer.target_len(target);
    gather_bitpacked(
        target,
        min_delta,
        next_value,
        bitpacked::Decoder::new(values, bitwidth, values_per_miniblock),
        gatherer,
    )?;
    let target_length = gatherer.target_len(target);

    debug_assert_eq!(target_length - start_length, values_per_miniblock);

    Ok(())
}

fn gather_block<'a, G: DeltaGatherer>(
    target: &mut G::Target,
    num_miniblocks: usize,
    values_per_miniblock: usize,
    mut values: &'a [u8],
    next_value: &mut i64,
    gatherer: &mut G,
) -> ParquetResult<&'a [u8]> {
    let (min_delta, consumed) = zigzag_leb128::decode(values);
    values = &values[consumed..];
    let bitwidths;
    (bitwidths, values) = values
        .split_at_checked(num_miniblocks)
        .ok_or(ParquetError::oos(
            "Not enough bitwidths available in delta encoding",
        ))?;

    gatherer.target_reserve(target, num_miniblocks * values_per_miniblock);
    for &bitwidth in bitwidths {
        let miniblock;
        (miniblock, values) = values
            .split_at_checked((bitwidth as usize * values_per_miniblock).div_ceil(8))
            .ok_or(ParquetError::oos(
                "Not enough bytes for miniblock in delta encoding",
            ))?;
        gather_miniblock(
            target,
            min_delta,
            bitwidth,
            miniblock,
            values_per_miniblock,
            next_value,
            gatherer,
        )?;
    }

    Ok(values)
}

impl<'a> Block<'a> {
    fn new(
        mut values: &'a [u8],
        num_miniblocks: usize,
        values_per_miniblock: usize,
        length: usize,
    ) -> ParquetResult<Self> {
        debug_assert!(!values.is_empty());

        let length = usize::min(length, num_miniblocks * values_per_miniblock);
        let actual_num_miniblocks =
            usize::min(num_miniblocks, length.div_ceil(values_per_miniblock));

        if actual_num_miniblocks == 0 {
            return Ok(Self::default());
        }

        // <min delta> <list of bitwidths of miniblocks> <miniblocks>

        let (min_delta, consumed) = zigzag_leb128::decode(values);

        values = &values[consumed..];
        let Some((bitwidths, remainder)) = values.split_at_checked(num_miniblocks) else {
            return Err(ParquetError::oos(
                "Not enough bitwidths available in delta encoding",
            ));
        };

        let bitwidths = bitwidths
            .get(..actual_num_miniblocks)
            .expect("actual_num_miniblocks <= num_miniblocks");
        // @NOTE: This never panics because the actual_num_miniblocks == 0 check above.
        let first_bitwidth = bitwidths[0];
        let bitwidths = &bitwidths[1..];

        if first_bitwidth > MAX_BITWIDTH {
            return Err(ParquetError::oos(format!(
                "Delta encoding bitwidth '{first_bitwidth}' is larger than maximum {MAX_BITWIDTH})"
            )));
        }

        let first_bitwidth = first_bitwidth as usize;

        let values_in_first_miniblock = usize::min(length, values_per_miniblock);
        let num_allocated_bytes = (first_bitwidth * values_per_miniblock).div_ceil(8);
        let num_actual_bytes = (first_bitwidth * values_in_first_miniblock).div_ceil(8);
        let Some((bytes, remainder)) = remainder.split_at_checked(num_allocated_bytes) else {
            return Err(ParquetError::oos(
                "Not enough bytes for miniblock in delta encoding",
            ));
        };
        let bytes = bytes
            .get(..num_actual_bytes)
            .expect("num_actual_bytes <= num_bytes");
        let current_miniblock =
            bitpacked::DecoderIter::new(bytes, first_bitwidth, values_in_first_miniblock)?;

        Ok(Block {
            min_delta,
            bitwidths,
            current_miniblock,
            remainder,
            num_values_remaining: length,
        })
    }

    pub fn gather_n_into<G: DeltaGatherer>(
        &mut self,
        target: &mut G::Target,
        n: usize,
        values_per_block: usize,
        values_per_miniblock: usize,
        next_value: &mut i64,
        gatherer: &mut G,
    ) -> ParquetResult<()> {
        debug_assert!(n <= values_per_block);
        debug_assert!(values_per_block >= values_per_miniblock);
        debug_assert_eq!(values_per_block % values_per_miniblock, 0);

        let mut n = usize::min(self.num_values_remaining, n);

        if n == 0 {
            return Ok(());
        }

        if n < self.current_miniblock.len() {
            self.current_miniblock.gather_n_into(
                target,
                self.min_delta,
                next_value,
                n,
                gatherer,
            )?;
            self.num_values_remaining -= n;
            return Ok(());
        }

        let length = self.current_miniblock.len();
        if length > 0 {
            self.current_miniblock.gather_n_into(
                target,
                self.min_delta,
                next_value,
                length,
                gatherer,
            )?;
            n -= length;
            self.num_values_remaining -= length;
        }

        while n >= values_per_miniblock {
            let bitwidth = self.bitwidths[0];
            self.bitwidths = &self.bitwidths[1..];

            let miniblock;
            (miniblock, self.remainder) = self
                .remainder
                .split_at((bitwidth as usize * values_per_miniblock).div_ceil(8));
            gather_miniblock(
                target,
                self.min_delta,
                bitwidth,
                miniblock,
                values_per_miniblock,
                next_value,
                gatherer,
            )?;
            n -= values_per_miniblock;
            self.num_values_remaining -= values_per_miniblock;
        }

        if n == 0 {
            return Ok(());
        }

        if !self.bitwidths.is_empty() {
            let bitwidth = self.bitwidths[0];
            self.bitwidths = &self.bitwidths[1..];

            if bitwidth > MAX_BITWIDTH {
                return Err(ParquetError::oos(format!(
                    "Delta encoding bitwidth '{bitwidth}' is larger than maximum {MAX_BITWIDTH})"
                )));
            }

            let miniblock;
            (miniblock, self.remainder) = self
                .remainder
                .split_at_checked((bitwidth as usize * values_per_miniblock).div_ceil(8))
                .ok_or(ParquetError::oos(
                    "Not enough space for delta encoded miniblock",
                ))?;
            let length = usize::min(values_per_miniblock, self.num_values_remaining);
            self.current_miniblock =
                bitpacked::DecoderIter::new(miniblock, bitwidth as usize, length)?;

            if n > 0 {
                self.current_miniblock.gather_n_into(
                    target,
                    self.min_delta,
                    next_value,
                    n,
                    gatherer,
                )?;
                self.num_values_remaining -= n;
            }
        }

        Ok(())
    }
}

/// Decoder of parquets' `DELTA_BINARY_PACKED`. Implements `Iterator<Item = i64>`.
/// # Implementation
/// This struct does not allocate on the heap.
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    num_miniblocks_per_block: usize,
    values_per_block: usize,

    values_remaining: usize,

    next_value: i64,
    values: &'a [u8],
    current_block: Block<'a>,
}

impl<'a> Decoder<'a> {
    pub fn try_new(mut values: &'a [u8]) -> ParquetResult<(Self, &'a [u8])> {
        let header_err = || ParquetError::oos("Insufficient bytes for Delta encoding header");

        // header:
        // <block size in values> <number of miniblocks in a block> <total value count> <first value>

        let (values_per_block, consumed) = uleb128::decode(values);
        let values_per_block = values_per_block as usize;
        values = values.get(consumed..).ok_or_else(header_err)?;

        assert_eq!(values_per_block % 128, 0);

        let (num_miniblocks_per_block, consumed) = uleb128::decode(values);
        let num_miniblocks_per_block = num_miniblocks_per_block as usize;
        values = values.get(consumed..).ok_or_else(header_err)?;

        let (total_count, consumed) = uleb128::decode(values);
        let total_count = total_count as usize;
        values = values.get(consumed..).ok_or_else(header_err)?;

        let (first_value, consumed) = zigzag_leb128::decode(values);
        values = values.get(consumed..).ok_or_else(header_err)?;

        assert_eq!(values_per_block % num_miniblocks_per_block, 0);
        assert_eq!((values_per_block / num_miniblocks_per_block) % 32, 0);

        let values_per_miniblock = values_per_block / num_miniblocks_per_block;
        assert_eq!(values_per_miniblock % 8, 0);

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

                if rem.len() < num_miniblocks_per_block {
                    return Err(ParquetError::oos(
                        "Not enough bitwidths available in delta encoding",
                    ));
                }
                if rem
                    .get(..num_remaining_mini_blocks)
                    .expect("num_remaining_mini_blocks <= num_miniblocks_per_block")
                    .iter()
                    .copied()
                    .any(|bitwidth| bitwidth > MAX_BITWIDTH)
                {
                    return Err(ParquetError::oos(format!(
                        "Delta encoding miniblock with bitwidth higher than maximum {MAX_BITWIDTH} bits",
                    )));
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

                num_values_read = num_values_read.saturating_sub(values_per_block);
            }
            rem
        } else {
            values
        };

        let values = &values[..values.len() - remainder.len()];

        // If we only have one value (first_value), there are no blocks.
        let current_block = if total_count > 1 {
            Block::new(
                values,
                num_miniblocks_per_block,
                values_per_miniblock,
                usize::min(values_per_block, total_count - 1),
            )?
        } else {
            Block::default()
        };

        Ok((
            Self {
                num_miniblocks_per_block,
                values_per_block,
                values_remaining: total_count,
                next_value: first_value,
                values,
                current_block,
            },
            remainder,
        ))
    }

    pub fn len(&self) -> usize {
        self.values_remaining
    }

    pub fn gather_n_into<G: DeltaGatherer>(
        &mut self,
        target: &mut G::Target,
        mut n: usize,
        gatherer: &mut G,
    ) -> ParquetResult<()> {
        if n == 0 || self.values_remaining == 0 {
            return Ok(());
        }

        if self.values_remaining == 1 {
            gatherer.gather_one(target, self.next_value)?;
            self.values_remaining = 0;
            return Ok(());
        }

        let values_per_miniblock = self.values_per_block / self.num_miniblocks_per_block;

        let start_num_values_remaining = self.current_block.num_values_remaining;
        if n <= self.current_block.num_values_remaining {
            self.current_block.gather_n_into(
                target,
                n,
                self.values_per_block,
                values_per_miniblock,
                &mut self.next_value,
                gatherer,
            )?;
            debug_assert_eq!(
                self.current_block.num_values_remaining,
                start_num_values_remaining - n
            );
            self.values = self.current_block.remainder;
            self.values_remaining = self.values_remaining.saturating_sub(n);
            return Ok(());
        }

        self.current_block.gather_n_into(
            target,
            self.current_block.num_values_remaining,
            self.values_per_block,
            values_per_miniblock,
            &mut self.next_value,
            gatherer,
        )?;
        debug_assert_eq!(self.current_block.num_values_remaining, 0);
        self.values = self.current_block.remainder;
        self.current_block = Block::default();
        self.values_remaining -= start_num_values_remaining;
        n -= start_num_values_remaining;

        while self.values_remaining >= self.values_per_block && n >= self.values_per_block {
            self.values = gather_block(
                target,
                self.num_miniblocks_per_block,
                values_per_miniblock,
                self.values,
                &mut self.next_value,
                gatherer,
            )?;
            n -= self.values_per_block;
            self.values_remaining -= self.values_per_block;
        }

        if self.values_remaining == 0 {
            return Ok(());
        }

        if self.values_remaining == 1 {
            gatherer.gather_one(target, self.next_value)?;
            self.values_remaining = 0;
            return Ok(());
        }

        let num_block_values = usize::min(self.values_remaining, self.values_per_block);
        self.current_block = Block::new(
            self.values,
            self.num_miniblocks_per_block,
            values_per_miniblock,
            num_block_values,
        )?;
        let num_gather_values = usize::min(num_block_values, n);
        self.current_block.gather_n_into(
            target,
            num_gather_values,
            self.values_per_block,
            values_per_miniblock,
            &mut self.next_value,
            gatherer,
        )?;
        self.values_remaining -= num_gather_values;

        Ok(())
    }

    pub fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        let mut gatherer = SkipGatherer;
        self.gather_n_into(&mut 0usize, n, &mut gatherer)
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = Result<i64, ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.values_remaining == 0 {
            return None;
        }

        let result = Some(Ok(self.next_value));
        if let Err(e) = self.skip_in_place(1) {
            return Some(Err(e));
        }
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

        let (mut decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(&r[..], &[1]);
        assert_eq!(data.len() - rem.len(), 5);
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

        let (mut decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(expected, r);

        assert_eq!(data.len() - rem.len(), 10);
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

        let (mut decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(expected, r);
        assert_eq!(data.len() - rem.len(), data.len() - 3);
        assert_eq!(rem.len(), 3);
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

        let (mut decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(&expected[..], &r[..]);
        assert_eq!(data.len() - rem.len(), data.len() - 3);
        assert_eq!(rem.len(), 3);
    }
}
