use super::Translator;
use crate::parquet::encoding::bitpacked::{self, Unpackable, Unpacked};
use crate::parquet::error::ParquetResult;

#[derive(Debug, Clone)]
pub struct BufferedBitpacked<'a> {
    pub unpacked: [u32; 32],
    pub unpacked_start: usize,
    pub unpacked_end: usize,

    pub decoder: bitpacked::Decoder<'a, u32>,
}

#[derive(Debug, Clone)]
pub struct BufferedRle {
    pub value: u32,
    pub length: usize,
}

/// A buffered set of items for the [`HybridRleDecoder`]. This can be iterated over and stopped at
/// any time.
#[derive(Debug, Clone)]
pub enum HybridRleBuffered<'a> {
    Bitpacked(BufferedBitpacked<'a>),
    Rle(BufferedRle),
}

impl Iterator for BufferedRle {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length > 0 {
            self.length -= 1;
            Some(self.value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl ExactSizeIterator for BufferedRle {}

impl<'a> Iterator for BufferedBitpacked<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.unpacked_start < self.unpacked_end {
            let value = self.unpacked[self.unpacked_start];
            self.unpacked_start += 1;
            return Some(value);
        }

        self.decoder
            .chunked()
            .next_inexact()
            .map(|(unpacked, unpacked_length)| {
                debug_assert!(unpacked_length > 0);
                let value = unpacked[0];
                self.unpacked = unpacked;
                self.unpacked_end = unpacked_length;
                self.unpacked_start = 1;
                value
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let unpacked_num_elements = self.unpacked_end - self.unpacked_start;
        let exact = unpacked_num_elements + self.decoder.len();
        (exact, Some(exact))
    }
}

impl<'a> ExactSizeIterator for BufferedBitpacked<'a> {}

impl<'a> Iterator for HybridRleBuffered<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            HybridRleBuffered::Bitpacked(b) => b.next(),
            HybridRleBuffered::Rle(b) => b.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            HybridRleBuffered::Bitpacked(b) => b.size_hint(),
            HybridRleBuffered::Rle(b) => b.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for HybridRleBuffered<'a> {}

impl<'a> BufferedBitpacked<'a> {
    fn translate_and_collect_limited_into<O: Clone + Default>(
        &mut self,
        target: &mut Vec<O>,
        limit: usize,
        translator: &impl Translator<O>,
    ) -> ParquetResult<usize> {
        let unpacked_num_elements = self.unpacked_end - self.unpacked_start;
        if limit <= unpacked_num_elements {
            translator.translate_slice(
                target,
                &self.unpacked[self.unpacked_start..self.unpacked_start + limit],
            )?;
            self.unpacked_start += limit;
            return Ok(limit);
        }

        translator.translate_slice(
            target,
            &self.unpacked[self.unpacked_start..self.unpacked_end],
        )?;
        self.unpacked_end = 0;
        self.unpacked_start = 0;
        let limit = limit - unpacked_num_elements;

        let decoder = self.decoder.take();
        let decoder_len = decoder.len();
        if limit >= decoder_len {
            translator.translate_bitpacked_all(target, decoder)?;
            Ok(unpacked_num_elements + decoder_len)
        } else {
            let buffered = translator.translate_bitpacked_limited(target, limit, decoder)?;
            *self = buffered;
            Ok(unpacked_num_elements + limit)
        }
    }

    pub fn translate_and_collect_into<O: Clone + Default>(
        self,
        target: &mut Vec<O>,
        translator: &impl Translator<O>,
    ) -> ParquetResult<usize> {
        let unpacked_num_elements = self.unpacked_end - self.unpacked_start;
        translator.translate_slice(
            target,
            &self.unpacked[self.unpacked_start..self.unpacked_end],
        )?;
        let decoder_len = self.decoder.len();
        translator.translate_bitpacked_all(target, self.decoder)?;
        Ok(unpacked_num_elements + decoder_len)
    }

    pub fn skip_in_place(&mut self, n: usize) -> usize {
        let unpacked_num_elements = self.unpacked_end - self.unpacked_start;

        if n < unpacked_num_elements {
            self.unpacked_start += n;
            return n;
        }

        let n = n - unpacked_num_elements;

        if self.decoder.len() > n {
            let num_chunks = n / <u32 as Unpackable>::Unpacked::LENGTH;
            let unpacked_offset = n % <u32 as Unpackable>::Unpacked::LENGTH;
            self.decoder.skip_chunks(num_chunks);
            let (unpacked, unpacked_length) = self.decoder.chunked().next_inexact().unwrap();

            self.unpacked = unpacked;
            self.unpacked_start = unpacked_offset;
            self.unpacked_end = unpacked_length;

            return unpacked_num_elements + n;
        }

        self.decoder.len() + unpacked_num_elements
    }
}

impl BufferedRle {
    pub fn translate_and_collect_limited_into<O: Clone + Default>(
        &mut self,
        target: &mut Vec<O>,
        limit: usize,
        translator: &impl Translator<O>,
    ) -> ParquetResult<usize> {
        let value = translator.translate(self.value)?;
        let num_elements = usize::min(self.length, limit);
        self.length -= num_elements;
        target.resize(target.len() + num_elements, value);
        Ok(num_elements)
    }

    pub fn translate_and_collect_into<O: Clone + Default>(
        self,
        target: &mut Vec<O>,
        translator: &impl Translator<O>,
    ) -> ParquetResult<usize> {
        let value = translator.translate(self.value)?;
        target.resize(target.len() + self.length, value);
        Ok(self.length)
    }

    pub fn skip_in_place(&mut self, n: usize) -> usize {
        let num_elements = usize::min(self.length, n);
        self.length -= num_elements;
        num_elements
    }
}

impl<'a> HybridRleBuffered<'a> {
    pub fn translate_and_collect_limited_into<O: Clone + Default>(
        &mut self,
        target: &mut Vec<O>,
        limit: usize,
        translator: &impl Translator<O>,
    ) -> ParquetResult<usize> {
        let start_target_length = target.len();
        let start_length = self.len();

        let num_processed = match self {
            HybridRleBuffered::Bitpacked(b) => {
                b.translate_and_collect_limited_into(target, limit, translator)
            },
            HybridRleBuffered::Rle(b) => {
                b.translate_and_collect_limited_into(target, limit, translator)
            },
        }?;

        debug_assert!(num_processed <= limit);
        debug_assert_eq!(num_processed, target.len() - start_target_length);
        debug_assert_eq!(num_processed, start_length - self.len());

        Ok(num_processed)
    }

    pub fn translate_and_collect_into<O: Clone + Default>(
        self,
        target: &mut Vec<O>,
        translator: &impl Translator<O>,
    ) -> ParquetResult<usize> {
        let start_target_length = target.len();
        let start_length = self.len();

        let num_processed = match self {
            HybridRleBuffered::Bitpacked(b) => b.translate_and_collect_into(target, translator),
            HybridRleBuffered::Rle(b) => b.translate_and_collect_into(target, translator),
        }?;

        debug_assert_eq!(num_processed, target.len() - start_target_length);
        debug_assert_eq!(num_processed, start_length);

        Ok(num_processed)
    }

    pub fn skip_in_place(&mut self, n: usize) -> usize {
        let start_length = self.len();

        let num_skipped = match self {
            HybridRleBuffered::Bitpacked(b) => b.skip_in_place(n),
            HybridRleBuffered::Rle(b) => b.skip_in_place(n),
        };

        debug_assert!(num_skipped <= n);
        debug_assert_eq!(num_skipped, start_length - self.len());

        num_skipped
    }
}
