//! This module implements the `DELTA_BINARY_PACKED` encoding.
//!
//! For performance reasons this is done without iterators. Instead, we have `gather_n` functions
//! and a `DeltaGatherer` trait. These allow efficient decoding and mapping of the decoded values.
//!
//! Full information on the delta encoding can be found on the Apache Parquet Format repository.
//!
//! <https://github.com/apache/parquet-format/blob/e517ac4dbe08d518eb5c2e58576d4c711973db94/Encodings.md#delta-encoding-delta_binary_packed--5>
//!
//! Delta encoding compresses sequential integer values by encoding the first value and the
//! differences between consequentive values. This variant encodes the data into `Block`s and
//! `MiniBlock`s.
//!
//! - A `Block` contains a minimum delta, bitwidths and one or more miniblocks.
//! - A `MiniBlock` contains many deltas that are encoded in [`bitpacked`] encoding.
//!
//! The decoder keeps track of the last value and calculates a new value with the following
//! function.
//!
//! ```text
//! NextValue(Delta) = {
//!     Value = Decoder.LastValue + Delta + Block.MinDelta
//!     Decoder.LastValue = Value
//!     return Value
//! }
//! ```
//!
//! Note that all these additions need to be wrapping.

use super::super::{bitpacked, uleb128, zigzag_leb128};
use super::lin_natural_sum;
use crate::parquet::encoding::bitpacked::{Unpackable, Unpacked};
use crate::parquet::error::{ParquetError, ParquetResult};

const MAX_BITWIDTH: u8 = 64;

/// Decoder of parquets' `DELTA_BINARY_PACKED`.
#[derive(Debug)]
pub struct Decoder<'a> {
    num_miniblocks_per_block: usize,
    values_per_block: usize,

    values_remaining: usize,

    last_value: i64,

    values: &'a [u8],

    block: Block<'a>,
}

#[derive(Debug)]
struct Block<'a> {
    min_delta: i64,

    /// Bytes that give the `num_bits` for the [`bitpacked::Decoder`].
    ///
    /// Invariant: `bitwidth[i] <= MAX_BITWIDTH` for all `i`
    bitwidths: &'a [u8],
    values_remaining: usize,
    miniblock: MiniBlock<'a>,
}

#[derive(Debug)]
struct MiniBlock<'a> {
    decoder: bitpacked::Decoder<'a, u64>,
    buffered: <u64 as Unpackable>::Unpacked,
    unpacked_start: usize,
    unpacked_end: usize,
}

struct SkipGatherer;
pub(crate) struct SumGatherer(pub(crate) usize);

pub trait DeltaGatherer {
    type Target: std::fmt::Debug;

    fn target_len(&self, target: &Self::Target) -> usize;
    fn target_reserve(&self, target: &mut Self::Target, n: usize);

    /// Gather one element with value `v` into `target`.
    fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()>;

    /// Gather `num_repeats` elements into `target`.
    ///
    /// The first value is `v` and the `n`-th value is `v + (n-1)*delta`.
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
    /// Gather a `slice` of elements into `target`.
    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        for &v in slice {
            self.gather_one(target, v)?;
        }
        Ok(())
    }
    /// Gather a `chunk` of elements into `target`.
    fn gather_chunk(&mut self, target: &mut Self::Target, chunk: &[i64; 64]) -> ParquetResult<()> {
        self.gather_slice(target, chunk)
    }
}

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
    fn gather_constant(
        &mut self,
        target: &mut Self::Target,
        _v: i64,
        _delta: i64,
        num_repeats: usize,
    ) -> ParquetResult<()> {
        *target += num_repeats;
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

impl DeltaGatherer for SumGatherer {
    type Target = usize;

    fn target_len(&self, _target: &Self::Target) -> usize {
        self.0
    }
    fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

    fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
        if v < 0 {
            return Err(ParquetError::oos(format!(
                "Invalid delta encoding length {v}"
            )));
        }

        *target += v as usize;
        self.0 += 1;
        Ok(())
    }
    fn gather_constant(
        &mut self,
        target: &mut Self::Target,
        v: i64,
        delta: i64,
        num_repeats: usize,
    ) -> ParquetResult<()> {
        if v < 0 || (delta < 0 && num_repeats > 0 && (num_repeats - 1) as i64 * delta + v < 0) {
            return Err(ParquetError::oos("Invalid delta encoding length"));
        }

        *target += lin_natural_sum(v, delta, num_repeats) as usize;

        Ok(())
    }
    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        let min = slice.iter().copied().min().unwrap_or_default();
        if min < 0 {
            return Err(ParquetError::oos(format!(
                "Invalid delta encoding length {min}"
            )));
        }

        *target += slice.iter().copied().map(|v| v as usize).sum::<usize>();
        self.0 += slice.len();
        Ok(())
    }
    fn gather_chunk(&mut self, target: &mut Self::Target, chunk: &[i64; 64]) -> ParquetResult<()> {
        let min = chunk.iter().copied().min().unwrap_or_default();
        if min < 0 {
            return Err(ParquetError::oos(format!(
                "Invalid delta encoding length {min}"
            )));
        }
        *target += chunk.iter().copied().map(|v| v as usize).sum::<usize>();
        self.0 += chunk.len();
        Ok(())
    }
}

/// Gather the rest of the [`bitpacked::Decoder`] into `target`
fn gather_bitpacked<G: DeltaGatherer>(
    target: &mut G::Target,
    min_delta: i64,
    last_value: &mut i64,
    mut decoder: bitpacked::Decoder<u64>,
    gatherer: &mut G,
) -> ParquetResult<()> {
    let mut chunked = decoder.chunked();
    for mut chunk in &mut chunked {
        for value in &mut chunk {
            *last_value = last_value
                .wrapping_add(*value as i64)
                .wrapping_add(min_delta);
            *value = *last_value as u64;
        }

        let chunk = bytemuck::cast_ref(&chunk);
        gatherer.gather_chunk(target, chunk)?;
    }

    if let Some((mut chunk, length)) = chunked.next_inexact() {
        let slice = &mut chunk[..length];

        for value in slice.iter_mut() {
            *last_value = last_value
                .wrapping_add(*value as i64)
                .wrapping_add(min_delta);
            *value = *last_value as u64;
        }

        let slice = bytemuck::cast_slice(slice);
        gatherer.gather_slice(target, slice)?;
    }

    Ok(())
}

/// Gather an entire [`MiniBlock`] into `target`
fn gather_miniblock<G: DeltaGatherer>(
    target: &mut G::Target,
    min_delta: i64,
    bitwidth: u8,
    values: &[u8],
    values_per_miniblock: usize,
    last_value: &mut i64,
    gatherer: &mut G,
) -> ParquetResult<()> {
    let bitwidth = bitwidth as usize;

    if bitwidth == 0 {
        let v = last_value.wrapping_add(min_delta);
        gatherer.gather_constant(target, v, min_delta, values_per_miniblock)?;
        *last_value = last_value.wrapping_add(min_delta * values_per_miniblock as i64);
        return Ok(());
    }

    debug_assert!(bitwidth <= 64);
    debug_assert_eq!((bitwidth * values_per_miniblock).div_ceil(8), values.len());

    let start_length = gatherer.target_len(target);
    gather_bitpacked(
        target,
        min_delta,
        last_value,
        bitpacked::Decoder::new(values, bitwidth, values_per_miniblock),
        gatherer,
    )?;
    let target_length = gatherer.target_len(target);

    debug_assert_eq!(target_length - start_length, values_per_miniblock);

    Ok(())
}

/// Gather an entire [`Block`] into `target`
fn gather_block<'a, G: DeltaGatherer>(
    target: &mut G::Target,
    num_miniblocks: usize,
    values_per_miniblock: usize,
    mut values: &'a [u8],
    last_value: &mut i64,
    gatherer: &mut G,
) -> ParquetResult<&'a [u8]> {
    let (min_delta, consumed) = zigzag_leb128::decode(values);
    values = &values[consumed..];
    let bitwidths;
    (bitwidths, values) = values
        .split_at_checked(num_miniblocks)
        .ok_or_else(|| ParquetError::oos("Not enough bitwidths available in delta encoding"))?;

    gatherer.target_reserve(target, num_miniblocks * values_per_miniblock);
    for &bitwidth in bitwidths {
        let miniblock;
        (miniblock, values) = values
            .split_at_checked((bitwidth as usize * values_per_miniblock).div_ceil(8))
            .ok_or_else(|| ParquetError::oos("Not enough bytes for miniblock in delta encoding"))?;
        gather_miniblock(
            target,
            min_delta,
            bitwidth,
            miniblock,
            values_per_miniblock,
            last_value,
            gatherer,
        )?;
    }

    Ok(values)
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
        //
        // This also has the added benefit of error checking in advance, thus we can unwrap in
        // other places.

        let mut rem = values;
        if total_count > 1 {
            let mut num_values_left = total_count - 1;
            while num_values_left > 0 {
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
                    num_values_left.div_ceil(values_per_miniblock),
                );

                // block:
                // <min delta> <list of bitwidths of miniblocks> <miniblocks>

                let (_, consumed) = zigzag_leb128::decode(rem);
                rem = rem.get(consumed..).ok_or_else(|| {
                    ParquetError::oos("No min-delta value in delta encoding miniblock")
                })?;

                if rem.len() < num_miniblocks_per_block {
                    return Err(ParquetError::oos(
                        "Not enough bitwidths available in delta encoding",
                    ));
                }
                if let Some(err_bitwidth) = rem
                    .get(..num_remaining_mini_blocks)
                    .expect("num_remaining_mini_blocks <= num_miniblocks_per_block")
                    .iter()
                    .copied()
                    .find(|&bitwidth| bitwidth > MAX_BITWIDTH)
                {
                    return Err(ParquetError::oos(format!(
                        "Delta encoding miniblock with bitwidth {err_bitwidth} higher than maximum {MAX_BITWIDTH} bits",
                    )));
                }

                let num_bitpacking_bytes = rem[..num_remaining_mini_blocks]
                    .iter()
                    .copied()
                    .map(|bitwidth| (bitwidth as usize * values_per_miniblock).div_ceil(8))
                    .sum::<usize>();

                rem = rem
                    .get(num_miniblocks_per_block + num_bitpacking_bytes..)
                    .ok_or_else(|| {
                        ParquetError::oos(
                            "Not enough bytes for all bitpacked values in delta encoding",
                        )
                    })?;

                num_values_left = num_values_left.saturating_sub(values_per_block);
            }
        }

        let values = &values[..values.len() - rem.len()];

        let decoder = Self {
            num_miniblocks_per_block,
            values_per_block,
            values_remaining: total_count.saturating_sub(1),
            last_value: first_value,
            values,

            block: Block {
                // @NOTE:
                // We add one delta=0 into the buffered block which allows us to
                // prepend like the `first_value` is just any normal value.
                //
                // This is a bit of a hack, but makes the rest of the logic
                // **A LOT** simpler.
                values_remaining: usize::from(total_count > 0),
                min_delta: 0,
                bitwidths: &[],
                miniblock: MiniBlock {
                    decoder: bitpacked::Decoder::try_new_allow_zero(&[], 0, 1)?,
                    buffered: <u64 as Unpackable>::Unpacked::zero(),
                    unpacked_start: 0,
                    unpacked_end: 0,
                },
            },
        };

        Ok((decoder, rem))
    }

    /// Consume a new [`Block`] from `self.values`.
    fn consume_block(&mut self) {
        // @NOTE: All the panics here should be prevented in the `Decoder::try_new`.

        debug_assert!(!self.values.is_empty());

        let values_per_miniblock = self.values_per_miniblock();

        let length = usize::min(self.values_remaining, self.values_per_block);
        let actual_num_miniblocks = usize::min(
            self.num_miniblocks_per_block,
            length.div_ceil(values_per_miniblock),
        );

        debug_assert!(actual_num_miniblocks > 0);

        // <min delta> <list of bitwidths of miniblocks> <miniblocks>

        let (min_delta, consumed) = zigzag_leb128::decode(self.values);

        self.values = &self.values[consumed..];
        let (bitwidths, remainder) = self.values.split_at(self.num_miniblocks_per_block);

        let first_bitwidth = bitwidths[0];
        let bitwidths = &bitwidths[1..actual_num_miniblocks];
        debug_assert!(first_bitwidth <= MAX_BITWIDTH);
        let first_bitwidth = first_bitwidth as usize;

        let values_in_first_miniblock = usize::min(length, values_per_miniblock);
        let num_allocated_bytes = (first_bitwidth * values_per_miniblock).div_ceil(8);
        let num_actual_bytes = (first_bitwidth * values_in_first_miniblock).div_ceil(8);
        let (bytes, remainder) = remainder.split_at(num_allocated_bytes);
        let bytes = &bytes[..num_actual_bytes];

        let decoder =
            bitpacked::Decoder::new_allow_zero(bytes, first_bitwidth, values_in_first_miniblock);

        self.block = Block {
            min_delta,
            bitwidths,
            values_remaining: length,
            miniblock: MiniBlock {
                decoder,
                // We can leave this as it should not be read before it is updated
                buffered: self.block.miniblock.buffered,
                unpacked_start: 0,
                unpacked_end: 0,
            },
        };

        self.values_remaining -= length;
        self.values = remainder;
    }

    /// Gather `n` elements from the current [`MiniBlock`] to `target`
    fn gather_miniblock_n_into<G: DeltaGatherer>(
        &mut self,
        target: &mut G::Target,
        mut n: usize,
        gatherer: &mut G,
    ) -> ParquetResult<()> {
        debug_assert!(n > 0);
        debug_assert!(self.miniblock_len() >= n);

        // If the `num_bits == 0`, the delta is constant and equal to `min_delta`. The
        // `bitpacked::Decoder` basically only keeps track of the length.
        if self.block.miniblock.decoder.num_bits() == 0 {
            let num_repeats = usize::min(self.miniblock_len(), n);
            let v = self.last_value.wrapping_add(self.block.min_delta);
            gatherer.gather_constant(target, v, self.block.min_delta, num_repeats)?;
            self.last_value = self
                .last_value
                .wrapping_add(self.block.min_delta * num_repeats as i64);
            self.block.miniblock.decoder.length -= num_repeats;
            return Ok(());
        }

        if self.block.miniblock.unpacked_start < self.block.miniblock.unpacked_end {
            let length = usize::min(
                n,
                self.block.miniblock.unpacked_end - self.block.miniblock.unpacked_start,
            );
            self.block.miniblock.buffered
                [self.block.miniblock.unpacked_start..self.block.miniblock.unpacked_start + length]
                .iter_mut()
                .for_each(|v| {
                    self.last_value = self
                        .last_value
                        .wrapping_add(*v as i64)
                        .wrapping_add(self.block.min_delta);
                    *v = self.last_value as u64;
                });
            gatherer.gather_slice(
                target,
                bytemuck::cast_slice(
                    &self.block.miniblock.buffered[self.block.miniblock.unpacked_start
                        ..self.block.miniblock.unpacked_start + length],
                ),
            )?;
            n -= length;
            self.block.miniblock.unpacked_start += length;
        }

        if n == 0 {
            return Ok(());
        }

        const ITEMS_PER_PACK: usize = <<u64 as Unpackable>::Unpacked as Unpacked<u64>>::LENGTH;
        for _ in 0..n / ITEMS_PER_PACK {
            let mut chunk = self.block.miniblock.decoder.chunked().next().unwrap();
            chunk.iter_mut().for_each(|v| {
                self.last_value = self
                    .last_value
                    .wrapping_add(*v as i64)
                    .wrapping_add(self.block.min_delta);
                *v = self.last_value as u64;
            });
            gatherer.gather_chunk(target, bytemuck::cast_ref(&chunk))?;
            n -= ITEMS_PER_PACK;
        }

        if n == 0 {
            return Ok(());
        }

        let Some((chunk, len)) = self.block.miniblock.decoder.chunked().next_inexact() else {
            debug_assert_eq!(n, 0);
            self.block.miniblock.buffered = <u64 as Unpackable>::Unpacked::zero();
            self.block.miniblock.unpacked_start = 0;
            self.block.miniblock.unpacked_end = 0;
            return Ok(());
        };

        self.block.miniblock.buffered = chunk;
        self.block.miniblock.unpacked_start = 0;
        self.block.miniblock.unpacked_end = len;

        if n > 0 {
            let length = usize::min(n, self.block.miniblock.unpacked_end);
            self.block.miniblock.buffered[..length]
                .iter_mut()
                .for_each(|v| {
                    self.last_value = self
                        .last_value
                        .wrapping_add(*v as i64)
                        .wrapping_add(self.block.min_delta);
                    *v = self.last_value as u64;
                });
            gatherer.gather_slice(
                target,
                bytemuck::cast_slice(&self.block.miniblock.buffered[..length]),
            )?;
            self.block.miniblock.unpacked_start = length;
        }

        Ok(())
    }

    /// Gather `n` elements from the current [`Block`] to `target`
    fn gather_block_n_into<G: DeltaGatherer>(
        &mut self,
        target: &mut G::Target,
        n: usize,
        gatherer: &mut G,
    ) -> ParquetResult<()> {
        let values_per_miniblock = self.values_per_miniblock();

        debug_assert!(n <= self.values_per_block);
        debug_assert!(self.values_per_block >= values_per_miniblock);
        debug_assert_eq!(self.values_per_block % values_per_miniblock, 0);

        let mut n = usize::min(self.block.values_remaining, n);

        if n == 0 {
            return Ok(());
        }

        let miniblock_len = self.miniblock_len();
        if n < miniblock_len {
            self.gather_miniblock_n_into(target, n, gatherer)?;
            debug_assert_eq!(self.miniblock_len(), miniblock_len - n);
            self.block.values_remaining -= n;
            return Ok(());
        }

        if miniblock_len > 0 {
            self.gather_miniblock_n_into(target, miniblock_len, gatherer)?;
            n -= miniblock_len;
            self.block.values_remaining -= miniblock_len;
        }

        while n >= values_per_miniblock {
            let bitwidth = self.block.bitwidths[0];
            self.block.bitwidths = &self.block.bitwidths[1..];

            let miniblock;
            (miniblock, self.values) = self
                .values
                .split_at((bitwidth as usize * values_per_miniblock).div_ceil(8));
            gather_miniblock(
                target,
                self.block.min_delta,
                bitwidth,
                miniblock,
                values_per_miniblock,
                &mut self.last_value,
                gatherer,
            )?;
            n -= values_per_miniblock;
            self.block.values_remaining -= values_per_miniblock;
        }

        if n == 0 {
            return Ok(());
        }

        if !self.block.bitwidths.is_empty() {
            let bitwidth = self.block.bitwidths[0];
            self.block.bitwidths = &self.block.bitwidths[1..];

            if bitwidth > MAX_BITWIDTH {
                return Err(ParquetError::oos(format!(
                    "Delta encoding bitwidth '{bitwidth}' is larger than maximum {MAX_BITWIDTH})"
                )));
            }

            let length = usize::min(values_per_miniblock, self.block.values_remaining);

            let num_allocated_bytes = (bitwidth as usize * values_per_miniblock).div_ceil(8);
            let num_actual_bytes = (bitwidth as usize * length).div_ceil(8);

            let miniblock;
            (miniblock, self.values) =
                self.values
                    .split_at_checked(num_allocated_bytes)
                    .ok_or(ParquetError::oos(
                        "Not enough space for delta encoded miniblock",
                    ))?;

            let miniblock = &miniblock[..num_actual_bytes];

            let decoder =
                bitpacked::Decoder::try_new_allow_zero(miniblock, bitwidth as usize, length)?;
            self.block.miniblock = MiniBlock {
                decoder,
                buffered: self.block.miniblock.buffered,
                unpacked_start: 0,
                unpacked_end: 0,
            };

            if n > 0 {
                self.gather_miniblock_n_into(target, n, gatherer)?;
                self.block.values_remaining -= n;
            }
        }

        Ok(())
    }

    /// Gather `n` elements to `target`
    pub fn gather_n_into<G: DeltaGatherer>(
        &mut self,
        target: &mut G::Target,
        mut n: usize,
        gatherer: &mut G,
    ) -> ParquetResult<()> {
        n = usize::min(n, self.len());

        if n == 0 {
            return Ok(());
        }

        let values_per_miniblock = self.values_per_block / self.num_miniblocks_per_block;

        let start_num_values_remaining = self.block.values_remaining;
        if n <= self.block.values_remaining {
            self.gather_block_n_into(target, n, gatherer)?;
            debug_assert_eq!(self.block.values_remaining, start_num_values_remaining - n);
            return Ok(());
        }

        n -= self.block.values_remaining;
        self.gather_block_n_into(target, self.block.values_remaining, gatherer)?;
        debug_assert_eq!(self.block.values_remaining, 0);

        while usize::min(n, self.values_remaining) >= self.values_per_block {
            self.values = gather_block(
                target,
                self.num_miniblocks_per_block,
                values_per_miniblock,
                self.values,
                &mut self.last_value,
                gatherer,
            )?;
            n -= self.values_per_block;
            self.values_remaining -= self.values_per_block;
        }

        if n == 0 {
            return Ok(());
        }

        self.consume_block();
        self.gather_block_n_into(target, n, gatherer)?;

        Ok(())
    }

    pub fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        let mut gatherer = SkipGatherer;
        self.gather_n_into(&mut 0usize, n, &mut gatherer)
    }

    #[cfg(test)]
    pub(crate) fn collect_n<E: std::fmt::Debug + Extend<i64>>(
        &mut self,
        e: &mut E,
        n: usize,
    ) -> ParquetResult<()> {
        struct ExtendGatherer<'a, E: std::fmt::Debug + Extend<i64>>(
            std::marker::PhantomData<&'a E>,
        );

        impl<'a, E: std::fmt::Debug + Extend<i64>> DeltaGatherer for ExtendGatherer<'a, E> {
            type Target = (usize, &'a mut E);

            fn target_len(&self, target: &Self::Target) -> usize {
                target.0
            }

            fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

            fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
                target.1.extend(Some(v));
                target.0 += 1;
                Ok(())
            }
        }

        let mut gatherer = ExtendGatherer(std::marker::PhantomData);
        let mut target = (0, e);

        self.gather_n_into(&mut target, n, &mut gatherer)
    }

    #[cfg(test)]
    pub(crate) fn collect<E: std::fmt::Debug + Extend<i64> + Default>(
        mut self,
    ) -> ParquetResult<E> {
        let mut e = E::default();
        self.collect_n(&mut e, self.len())?;
        Ok(e)
    }

    pub fn len(&self) -> usize {
        self.values_remaining + self.block.values_remaining
    }

    fn values_per_miniblock(&self) -> usize {
        debug_assert_eq!(self.values_per_block % self.num_miniblocks_per_block, 0);
        self.values_per_block / self.num_miniblocks_per_block
    }

    fn miniblock_len(&self) -> usize {
        self.block.miniblock.unpacked_end - self.block.miniblock.unpacked_start
            + self.block.miniblock.decoder.len()
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

        let (decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.collect::<Vec<_>>().unwrap();

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

        let (decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.collect::<Vec<_>>().unwrap();

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

        let (decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.collect::<Vec<_>>().unwrap();

        assert_eq!(expected, r);
        assert_eq!(rem, &[1, 2, 3]);
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

        let (decoder, rem) = Decoder::try_new(data).unwrap();
        let r = decoder.collect::<Vec<_>>().unwrap();

        assert_eq!(&expected[..], &r[..]);
        assert_eq!(data.len() - rem.len(), data.len() - 3);
        assert_eq!(rem.len(), 3);
    }
}
