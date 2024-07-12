use crate::parquet::encoding::bitpacked::{Decoder, Unpackable, Unpacked};
use crate::parquet::encoding::hybrid_rle::{BufferedBitpacked, HybridRleBuffered};
use crate::parquet::error::{ParquetError, ParquetResult};

/// Trait that describes what to do with consumed Hybrid-RLE Encoded values.
///
/// This is quite a general trait that provides a lot of open space as to how to handle the
/// Hybrid-RLE encoded values. There is also the [`Translator`] trait that is usually good enough
/// if you want just want to map values to another set of values and collect them into a vector.
///
/// Although, this trait might seem quite over-engineered (it might be), it is very useful for
/// performance. This allows for usage of the properties that [`HybridRleDecoder`] provides and for
/// definition of efficient procedures for collecting slices, chunks, repeated elements and
/// bit-packed elements.
///
/// The [`Translator`] doc-comment has a good description of why this trait is needed.
///
/// [`HybridRleDecoder`]: super::HybridRleDecoder
pub trait HybridRleGatherer<O: Clone> {
    type Target;

    fn target_reserve(&self, target: &mut Self::Target, n: usize);
    fn target_num_elements(&self, target: &Self::Target) -> usize;

    fn hybridrle_to_target(&self, value: u32) -> ParquetResult<O>;
    fn gather_one(&self, target: &mut Self::Target, value: O) -> ParquetResult<()>;
    fn gather_repeated(&self, target: &mut Self::Target, value: O, n: usize) -> ParquetResult<()>;
    fn gather_slice(&self, target: &mut Self::Target, source: &[u32]) -> ParquetResult<()> {
        self.target_reserve(target, source.len());
        for v in source {
            self.gather_one(target, self.hybridrle_to_target(*v)?)?;
        }
        Ok(())
    }
    fn gather_chunk(
        &self,
        target: &mut Self::Target,
        source: &<u32 as Unpackable>::Unpacked,
    ) -> ParquetResult<()> {
        self.gather_slice(target, source)
    }
    fn gather_bitpacked_all(
        &self,
        target: &mut Self::Target,
        mut decoder: Decoder<u32>,
    ) -> ParquetResult<()> {
        self.target_reserve(target, decoder.len());

        let mut chunked = decoder.chunked();

        for unpacked in &mut chunked {
            self.gather_chunk(target, &unpacked)?;
        }

        if let Some((last, last_length)) = chunked.remainder() {
            self.gather_slice(target, &last[..last_length])?;
        }

        Ok(())
    }

    fn gather_bitpacked_limited<'a>(
        &self,
        target: &mut Self::Target,
        mut decoder: Decoder<'a, u32>,
        limit: usize,
    ) -> ParquetResult<BufferedBitpacked<'a>> {
        assert!(limit < decoder.len());

        const CHUNK_SIZE: usize = <u32 as Unpackable>::Unpacked::LENGTH;

        let mut chunked = decoder.chunked();

        let num_full_chunks = limit / CHUNK_SIZE;
        for unpacked in (&mut chunked).take(num_full_chunks) {
            self.gather_chunk(target, &unpacked)?;
        }

        let (unpacked, unpacked_length) = chunked.next_inexact().unwrap();
        let unpacked_offset = limit % CHUNK_SIZE;
        debug_assert!(unpacked_offset < unpacked_length);
        self.gather_slice(target, &unpacked[..unpacked_offset])?;

        Ok(BufferedBitpacked {
            unpacked,

            unpacked_start: unpacked_offset,
            unpacked_end: unpacked_length,
            decoder,
        })
    }
    fn gather_bitpacked<'a>(
        &self,
        target: &mut Self::Target,
        decoder: Decoder<'a, u32>,
        limit: Option<usize>,
    ) -> ParquetResult<(usize, Option<HybridRleBuffered<'a>>)> {
        let length = decoder.len();

        match limit {
            None => self
                .gather_bitpacked_all(target, decoder)
                .map(|_| (length, None)),
            Some(limit) if limit >= length => self
                .gather_bitpacked_all(target, decoder)
                .map(|_| (length, None)),
            Some(limit) => self
                .gather_bitpacked_limited(target, decoder, limit)
                .map(|b| (limit, Some(HybridRleBuffered::Bitpacked(b)))),
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct ZeroCount {
    pub num_zero: usize,
    pub num_nonzero: usize,
}
pub struct ZeroCountGatherer;

impl HybridRleGatherer<ZeroCount> for ZeroCountGatherer {
    type Target = ZeroCount;

    #[inline(always)]
    fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

    #[inline]
    fn target_num_elements(&self, target: &Self::Target) -> usize {
        target.num_zero + target.num_nonzero
    }

    #[inline]
    fn hybridrle_to_target(&self, value: u32) -> ParquetResult<ZeroCount> {
        Ok(ZeroCount {
            num_zero: usize::from(value == 0),
            num_nonzero: usize::from(value != 0),
        })
    }

    #[inline]
    fn gather_one(&self, target: &mut Self::Target, value: ZeroCount) -> ParquetResult<()> {
        target.num_zero += value.num_zero;
        target.num_nonzero += value.num_nonzero;
        Ok(())
    }

    #[inline]
    fn gather_repeated(
        &self,
        target: &mut Self::Target,
        value: ZeroCount,
        n: usize,
    ) -> ParquetResult<()> {
        target.num_zero += value.num_zero * n;
        target.num_nonzero += value.num_nonzero * n;
        Ok(())
    }

    #[inline]
    fn gather_slice(&self, target: &mut Self::Target, source: &[u32]) -> ParquetResult<()> {
        let mut num_zero = 0;
        let mut num_nonzero = 0;

        for v in source {
            num_zero += usize::from(*v == 0);
            num_nonzero += usize::from(*v != 0);
        }

        target.num_zero += num_zero;
        target.num_nonzero += num_nonzero;

        Ok(())
    }
}

/// A trait to describe a translation from a HybridRLE encoding to an another format.
///
/// In essence, this is one method ([`Translator::translate`]) that maps an `u32` to the desired
/// output type `O`. There are several other methods that may provide optimized routines
/// for slices, chunks and decoders.
///
/// # Motivation
///
/// The [`HybridRleDecoder`] is used extensively during Parquet decoding because it is used for
/// Dremel decoding and dictionary decoding. We want to perform a transformation from this
/// space-efficient encoding to a buffer. Here, items might be skipped, might be mapped and only a
/// few items might be needed. There are 3 main ways to do this.
///
/// 1. Element-by-element translation using iterator `map`, `filter`, `skip`, etc. This suffers
///    from the problem that is difficult to SIMD the translation and that a `collect` might need
///    to constantly poll the `next` function. Next to that monomorphization might need to generate
///    many, many variants.
/// 2. Buffer most everything, filter and translate later. This has high memory-consumption and
///    might suffer from cache-eviction problems. This is computationally the most efficient, but
///    probably still has a high runtime. Also, this fails to utilize run-length information and
///    needs to retranslate all repeated elements.
/// 3. Batched operations. Here, we try to utilize the run-length information and utilize SIMD to
///    process many bitpacked items. This can provide the best of both worlds.
///
/// The [`HybridRleDecoder`][super::HybridRleDecoder] decoders utilizing both run-length encoding
/// and bitpacking. In both processes, this [`Translator`] trait allows for translation with (i) no
/// heap allocations and (ii) cheap buffering and can stop and start at any point. Consequently,
/// the memory consumption while doing these translations can be relatively low while still
/// processing items in batches.
///
/// [`HybridRleDecoder`]: super::HybridRleDecoder
pub trait Translator<O> {
    /// Translate from a decoded value to the output format
    fn translate(&self, value: u32) -> ParquetResult<O>;

    /// Translate from a slice of decoded values to the output format and write them to a `target`.
    ///
    /// This can overwritten to be more optimized.
    fn translate_slice(&self, target: &mut Vec<O>, source: &[u32]) -> ParquetResult<()> {
        target.reserve(source.len());
        for v in source {
            target.push(self.translate(*v)?);
        }
        Ok(())
    }

    /// Translate from a chunk of unpacked items to the output format and write them to a `target`.
    ///
    /// This is the same as [`Translator::translate_slice`] but with a known slice size. This can
    /// allow SIMD routines to better optimize the procedure.
    ///
    /// This can overwritten to be more optimized.
    fn translate_chunk(
        &self,
        target: &mut Vec<O>,
        source: &<u32 as Unpackable>::Unpacked,
    ) -> ParquetResult<()> {
        self.translate_slice(target, &source[..])
    }

    /// Translate and collect all the items in a [`Decoder`] to a `target`.
    ///
    /// This can overwritten to be more optimized.
    fn translate_bitpacked_all(
        &self,
        target: &mut Vec<O>,
        mut decoder: Decoder<u32>,
    ) -> ParquetResult<()> {
        target.reserve(decoder.len());

        let mut chunked = decoder.chunked();

        for unpacked in &mut chunked {
            self.translate_chunk(target, &unpacked)?;
        }

        if let Some((last, last_length)) = chunked.remainder() {
            self.translate_slice(target, &last[..last_length])?;
        }

        Ok(())
    }

    /// Translate and collect a limited number of items in a [`Decoder`] to a `target`.
    ///
    /// This can overwritten to be more optimized.
    ///
    /// # Panics
    ///
    /// This method panics when `limit` is larger than the `decoder` length.
    fn translate_bitpacked_limited<'a>(
        &self,
        target: &mut Vec<O>,
        mut decoder: Decoder<'a, u32>,
        limit: usize,
    ) -> ParquetResult<BufferedBitpacked<'a>> {
        assert!(limit < decoder.len());

        const CHUNK_SIZE: usize = <u32 as Unpackable>::Unpacked::LENGTH;

        let mut chunked = decoder.chunked();

        let num_full_chunks = limit / CHUNK_SIZE;
        for unpacked in (&mut chunked).take(num_full_chunks) {
            self.translate_chunk(target, &unpacked)?;
        }

        let (unpacked, unpacked_length) = chunked.next_inexact().unwrap();
        let unpacked_offset = limit % CHUNK_SIZE;
        debug_assert!(unpacked_offset < unpacked_length);
        self.translate_slice(target, &unpacked[..unpacked_offset])?;

        Ok(BufferedBitpacked {
            unpacked,

            unpacked_start: unpacked_offset,
            unpacked_end: unpacked_length,
            decoder,
        })
    }

    /// Translate and collect items in a [`Decoder`] to a `target`.
    ///
    /// This can overwritten to be more optimized.
    fn translate_bitpacked<'a>(
        &self,
        target: &mut Vec<O>,
        decoder: Decoder<'a, u32>,
        limit: Option<usize>,
    ) -> ParquetResult<(usize, Option<HybridRleBuffered<'a>>)> {
        let length = decoder.len();

        match limit {
            None => self
                .translate_bitpacked_all(target, decoder)
                .map(|_| (length, None)),
            Some(limit) if limit >= length => self
                .translate_bitpacked_all(target, decoder)
                .map(|_| (length, None)),
            Some(limit) => self
                .translate_bitpacked_limited(target, decoder, limit)
                .map(|b| (limit, Some(HybridRleBuffered::Bitpacked(b)))),
        }
    }
}

impl<O: Clone, T: Translator<O>> HybridRleGatherer<O> for T {
    type Target = Vec<O>;

    #[inline(always)]
    fn target_reserve(&self, target: &mut Self::Target, n: usize) {
        target.reserve(n);
    }
    #[inline(always)]
    fn target_num_elements(&self, target: &Self::Target) -> usize {
        target.len()
    }

    #[inline(always)]
    fn hybridrle_to_target(&self, value: u32) -> ParquetResult<O> {
        self.translate(value)
    }

    #[inline(always)]
    fn gather_one(&self, target: &mut Self::Target, value: O) -> ParquetResult<()> {
        target.push(value);
        Ok(())
    }

    #[inline(always)]
    fn gather_repeated(&self, target: &mut Self::Target, value: O, n: usize) -> ParquetResult<()> {
        target.resize(target.len() + n, value);
        Ok(())
    }

    #[inline(always)]
    fn gather_slice(&self, target: &mut Self::Target, source: &[u32]) -> ParquetResult<()> {
        self.translate_slice(target, source)
    }

    #[inline(always)]
    fn gather_chunk(
        &self,
        target: &mut Self::Target,
        source: &<u32 as Unpackable>::Unpacked,
    ) -> ParquetResult<()> {
        self.translate_chunk(target, source)
    }

    #[inline(always)]
    fn gather_bitpacked_all(
        &self,
        target: &mut Self::Target,
        decoder: Decoder<u32>,
    ) -> ParquetResult<()> {
        self.translate_bitpacked_all(target, decoder)
    }

    #[inline(always)]
    fn gather_bitpacked_limited<'a>(
        &self,
        target: &mut Self::Target,
        decoder: Decoder<'a, u32>,
        limit: usize,
    ) -> ParquetResult<BufferedBitpacked<'a>> {
        self.translate_bitpacked_limited(target, decoder, limit)
    }

    #[inline(always)]
    fn gather_bitpacked<'a>(
        &self,
        target: &mut Self::Target,
        decoder: Decoder<'a, u32>,
        limit: Option<usize>,
    ) -> ParquetResult<(usize, Option<HybridRleBuffered<'a>>)> {
        self.translate_bitpacked(target, decoder, limit)
    }
}

/// This is a unit translation variant of [`Translator`]. This just maps all encoded values from a
/// [`HybridRleDecoder`] to themselves.
///
/// [`HybridRleDecoder`]: super::HybridRleDecoder
pub struct UnitTranslator;

impl Translator<u32> for UnitTranslator {
    fn translate(&self, value: u32) -> ParquetResult<u32> {
        Ok(value)
    }

    fn translate_slice(&self, target: &mut Vec<u32>, source: &[u32]) -> ParquetResult<()> {
        target.extend_from_slice(source);
        Ok(())
    }
    fn translate_chunk(
        &self,
        target: &mut Vec<u32>,
        source: &<u32 as Unpackable>::Unpacked,
    ) -> ParquetResult<()> {
        target.extend_from_slice(&source[..]);
        Ok(())
    }
    fn translate_bitpacked_all(
        &self,
        target: &mut Vec<u32>,
        decoder: Decoder<u32>,
    ) -> ParquetResult<()> {
        decoder.collect_into(target);
        Ok(())
    }
}

/// This is a dictionary translation variant of [`Translator`].
///
/// All the [`HybridRleDecoder`] values are regarded as a offset into a dictionary.
///
/// [`HybridRleDecoder`]: super::HybridRleDecoder
pub struct DictionaryTranslator<'a, T>(pub &'a [T]);

impl<'a, T: Copy> Translator<T> for DictionaryTranslator<'a, T> {
    fn translate(&self, value: u32) -> ParquetResult<T> {
        self.0
            .get(value as usize)
            .cloned()
            .ok_or(ParquetError::oos("Dictionary index is out of range"))
    }

    fn translate_slice(&self, target: &mut Vec<T>, source: &[u32]) -> ParquetResult<()> {
        let Some(source_max) = source.iter().copied().max() else {
            return Ok(());
        };

        if source_max as usize >= self.0.len() {
            return Err(ParquetError::oos("Dictionary index is out of range"));
        }

        // Safety: We have checked before that source only has indexes that are smaller than the
        // dictionary length.
        target.extend(
            source
                .iter()
                .map(|&src_idx| unsafe { *self.0.get_unchecked(src_idx as usize) }),
        );

        Ok(())
    }

    fn translate_chunk(
        &self,
        target: &mut Vec<T>,
        source: &<u32 as Unpackable>::Unpacked,
    ) -> ParquetResult<()> {
        let source_max: u32 = source.iter().copied().max().unwrap();

        if source_max as usize >= self.0.len() {
            return Err(ParquetError::oos("Dictionary index is out of range"));
        }

        // Safety: We have checked before that source only has indexes that are smaller than the
        // dictionary length.
        target.extend(
            source
                .iter()
                .map(|&src_idx| unsafe { *self.0.get_unchecked(src_idx as usize) }),
        );

        Ok(())
    }
}

/// A closure-based translator
pub struct FnTranslator<O, F: Fn(u32) -> ParquetResult<O>>(pub F);

impl<O, F: Fn(u32) -> ParquetResult<O>> Translator<O> for FnTranslator<O, F> {
    fn translate(&self, value: u32) -> ParquetResult<O> {
        (self.0)(value)
    }
}

#[derive(Default)]
pub struct TryFromUsizeTranslator<O: TryFrom<usize>>(std::marker::PhantomData<O>);

impl<O: TryFrom<usize>> Translator<O> for TryFromUsizeTranslator<O> {
    fn translate(&self, value: u32) -> ParquetResult<O> {
        O::try_from(value as usize).map_err(|_| ParquetError::oos("Invalid cast in translation"))
    }
}

pub struct SliceDictionaryTranslator<'a, T> {
    dict: &'a [T],
    size: usize,
}

impl<'a, T> SliceDictionaryTranslator<'a, T> {
    pub fn new(dict: &'a [T], size: usize) -> Self {
        debug_assert_eq!(dict.len() % size, 0);
        Self { dict, size }
    }
}

impl<'a, T> Translator<&'a [T]> for SliceDictionaryTranslator<'a, T> {
    fn translate(&self, value: u32) -> ParquetResult<&'a [T]> {
        let idx = value as usize;

        if idx >= self.dict.len() / self.size {
            return Err(ParquetError::oos("Dictionary slice index is out of range"));
        }

        Ok(&self.dict[idx * self.size..(idx + 1) * self.size])
    }

    fn translate_slice(&self, target: &mut Vec<&'a [T]>, source: &[u32]) -> ParquetResult<()> {
        let Some(source_max) = source.iter().copied().max() else {
            return Ok(());
        };

        if source_max as usize >= self.dict.len() / self.size {
            return Err(ParquetError::oos("Dictionary index is out of range"));
        }

        // Safety: We have checked before that source only has indexes that are smaller than the
        // dictionary length.
        target.extend(source.iter().map(|&src_idx| unsafe {
            self.dict
                .get_unchecked((src_idx as usize) * self.size..(src_idx as usize + 1) * self.size)
        }));

        Ok(())
    }
}
