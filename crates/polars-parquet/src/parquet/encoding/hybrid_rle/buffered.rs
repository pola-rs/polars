use super::gatherer::HybridRleGatherer;
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
    fn gather_limited_into<O: Clone, G: HybridRleGatherer<O>>(
        &mut self,
        target: &mut G::Target,
        limit: usize,
        gatherer: &G,
    ) -> ParquetResult<usize> {
        let unpacked_num_elements = self.unpacked_end - self.unpacked_start;
        if limit <= unpacked_num_elements {
            gatherer.gather_slice(
                target,
                &self.unpacked[self.unpacked_start..self.unpacked_start + limit],
            )?;
            self.unpacked_start += limit;
            return Ok(limit);
        }

        gatherer.gather_slice(
            target,
            &self.unpacked[self.unpacked_start..self.unpacked_end],
        )?;
        self.unpacked_end = 0;
        self.unpacked_start = 0;
        let limit = limit - unpacked_num_elements;

        let decoder = self.decoder.take();
        let decoder_len = decoder.len();
        if limit >= decoder_len {
            gatherer.gather_bitpacked_all(target, decoder)?;
            Ok(unpacked_num_elements + decoder_len)
        } else {
            let buffered = gatherer.gather_bitpacked_limited(target, decoder, limit)?;
            *self = buffered;
            Ok(unpacked_num_elements + limit)
        }
    }

    pub fn gather_into<O: Clone, G: HybridRleGatherer<O>>(
        self,
        target: &mut G::Target,
        gatherer: &G,
    ) -> ParquetResult<usize> {
        let unpacked_num_elements = self.unpacked_end - self.unpacked_start;
        gatherer.gather_slice(
            target,
            &self.unpacked[self.unpacked_start..self.unpacked_end],
        )?;
        let decoder_len = self.decoder.len();
        gatherer.gather_bitpacked_all(target, self.decoder)?;
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
            debug_assert!(unpacked_offset < unpacked_length);

            self.unpacked = unpacked;
            self.unpacked_start = unpacked_offset;
            self.unpacked_end = unpacked_length;

            return unpacked_num_elements + n;
        }

        // We skip the entire decoder. Essentially, just zero it out.
        let decoder = self.decoder.take();
        self.unpacked_start = 0;
        self.unpacked_end = 0;

        decoder.len() + unpacked_num_elements
    }
}

impl BufferedRle {
    pub fn gather_limited_into<O: Clone, G: HybridRleGatherer<O>>(
        &mut self,
        target: &mut G::Target,
        limit: usize,
        gatherer: &G,
    ) -> ParquetResult<usize> {
        let value = gatherer.hybridrle_to_target(self.value)?;
        let num_elements = usize::min(self.length, limit);
        self.length -= num_elements;
        gatherer.gather_repeated(target, value, num_elements)?;
        Ok(num_elements)
    }

    pub fn gather_into<O: Clone, A: HybridRleGatherer<O>>(
        self,
        target: &mut A::Target,
        applicator: &A,
    ) -> ParquetResult<usize> {
        let value = applicator.hybridrle_to_target(self.value)?;
        applicator.gather_repeated(target, value, self.length)?;
        Ok(self.length)
    }

    pub fn skip_in_place(&mut self, n: usize) -> usize {
        let num_elements = usize::min(self.length, n);
        self.length -= num_elements;
        num_elements
    }
}

impl<'a> HybridRleBuffered<'a> {
    pub fn gather_limited_into<O: Clone, G: HybridRleGatherer<O>>(
        &mut self,
        target: &mut G::Target,
        limit: usize,
        gatherer: &G,
    ) -> ParquetResult<usize> {
        let start_target_length = gatherer.target_num_elements(target);
        let start_length = self.len();

        let num_processed = match self {
            HybridRleBuffered::Bitpacked(b) => b.gather_limited_into(target, limit, gatherer),
            HybridRleBuffered::Rle(b) => b.gather_limited_into(target, limit, gatherer),
        }?;

        debug_assert!(num_processed <= limit);
        debug_assert_eq!(
            num_processed,
            gatherer.target_num_elements(target) - start_target_length
        );
        debug_assert_eq!(num_processed, start_length - self.len());

        Ok(num_processed)
    }

    pub fn gather_into<O: Clone, G: HybridRleGatherer<O>>(
        self,
        target: &mut G::Target,
        gatherer: &G,
    ) -> ParquetResult<usize> {
        let start_target_length = gatherer.target_num_elements(target);
        let start_length = self.len();

        let num_processed = match self {
            HybridRleBuffered::Bitpacked(b) => b.gather_into(target, gatherer),
            HybridRleBuffered::Rle(b) => b.gather_into(target, gatherer),
        }?;

        debug_assert_eq!(
            num_processed,
            gatherer.target_num_elements(target) - start_target_length
        );
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
        debug_assert_eq!(
            num_skipped,
            start_length - self.len(),
            "{self:?}: {num_skipped} != {start_length} - {}",
            self.len()
        );

        num_skipped
    }
}
