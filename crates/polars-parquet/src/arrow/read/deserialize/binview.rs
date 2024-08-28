use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::{
    Array, BinaryViewArray, DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray,
    Utf8ViewArray, View,
};
use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::{ArrowDataType, PhysicalType};

use super::utils::{dict_indices_decoder, freeze_validity, BatchableCollector};
use crate::parquet::encoding::delta_bitpacked::{lin_natural_sum, DeltaGatherer};
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::{delta_byte_array, delta_length_byte_array, hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::{self, extend_from_decoder, Decoder, PageValidity};
use crate::read::PrimitiveLogicalType;

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> utils::StateTranslation<'a, BinViewDecoder> for StateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        decoder: &BinViewDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
    ) -> ParquetResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let values = BinaryIter::new(values, page.num_values());

                Ok(Self::Plain(values))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values = dict_indices_decoder(page)?;
                Ok(Self::Dictionary(values))
            },
            (Encoding::DeltaLengthByteArray, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::DeltaLengthByteArray(
                    delta_length_byte_array::Decoder::try_new(values)?,
                    Vec::new(),
                ))
            },
            (Encoding::DeltaByteArray, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::DeltaBytes(delta_byte_array::Decoder::try_new(
                    values,
                )?))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(v) => v.len_when_not_nullable(),
            Self::Dictionary(v) => v.len(),
            Self::DeltaLengthByteArray(v, _) => v.len(),
            Self::DeltaBytes(v) => v.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(t) => _ = t.by_ref().nth(n - 1),
            Self::Dictionary(t) => t.skip_in_place(n)?,
            Self::DeltaLengthByteArray(t, _) => t.skip_in_place(n)?,
            Self::DeltaBytes(t) => t.skip_in_place(n)?,
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut BinViewDecoder,
        decoded: &mut <BinViewDecoder as utils::Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<utils::PageValidity<'a>>,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        additional: usize,
    ) -> ParquetResult<()> {
        let views_offset = decoded.0.views().len();
        let buffer_offset = decoded.0.completed_buffers().len();

        let mut validate_utf8 = decoder.check_utf8.load(Ordering::Relaxed);

        match self {
            Self::Plain(page_values) => {
                decoder.decode_plain_encoded(
                    decoded,
                    page_values,
                    is_optional,
                    page_validity.as_mut(),
                    additional,
                )?;

                // Already done in decode_plain_encoded
                validate_utf8 = false;
            },
            Self::Dictionary(ref mut page) => {
                let dict = dict.unwrap();

                decoder.decode_dictionary_encoded(
                    decoded,
                    page,
                    is_optional,
                    page_validity.as_mut(),
                    dict,
                    additional,
                )?;

                // Already done in decode_plain_encoded
                validate_utf8 = false;
            },
            Self::DeltaLengthByteArray(ref mut page_values, ref mut lengths) => {
                let (values, validity) = decoded;

                let mut collector = DeltaCollector {
                    gatherer: &mut StatGatherer::default(),
                    pushed_lengths: lengths,
                    decoder: page_values,
                };

                match page_validity {
                    None => {
                        (&mut collector).push_n(values, additional)?;

                        if is_optional {
                            validity.extend_constant(additional, true);
                        }
                    },
                    Some(page_validity) => extend_from_decoder(
                        validity,
                        page_validity,
                        Some(additional),
                        values,
                        &mut collector,
                    )?,
                }

                collector.flush(values);
            },
            Self::DeltaBytes(ref mut page_values) => {
                let (values, validity) = decoded;

                let mut collector = DeltaBytesCollector {
                    decoder: page_values,
                };

                match page_validity {
                    None => {
                        collector.push_n(values, additional)?;

                        if is_optional {
                            validity.extend_constant(additional, true);
                        }
                    },
                    Some(page_validity) => extend_from_decoder(
                        validity,
                        page_validity,
                        Some(additional),
                        values,
                        collector,
                    )?,
                }
            },
        }

        if validate_utf8 {
            decoded
                .0
                .validate_utf8(buffer_offset, views_offset)
                .map_err(|_| ParquetError::oos("Binary view contained invalid UTF-8"))?
        }

        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct BinViewDecoder {
    check_utf8: AtomicBool,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    DeltaLengthByteArray(delta_length_byte_array::Decoder<'a>, Vec<u32>),
    DeltaBytes(delta_byte_array::Decoder<'a>),
}

impl utils::ExactSize for DecodedStateTuple {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl utils::ExactSize for (Vec<View>, Vec<Buffer<u8>>) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

pub(crate) struct DeltaCollector<'a, 'b> {
    // We gatherer the decoded lengths into `pushed_lengths`. Then, we `flush` those to the
    // `BinView` This allows us to group many memcopies into one and take better potential fast
    // paths for inlineable views and such.
    pub(crate) gatherer: &'b mut StatGatherer,
    pub(crate) pushed_lengths: &'b mut Vec<u32>,

    pub(crate) decoder: &'b mut delta_length_byte_array::Decoder<'a>,
}

pub(crate) struct DeltaBytesCollector<'a, 'b> {
    pub(crate) decoder: &'b mut delta_byte_array::Decoder<'a>,
}

/// A [`DeltaGatherer`] that gathers the minimum, maximum and summation of the values as `usize`s.
pub(crate) struct StatGatherer {
    min: usize,
    max: usize,
    sum: usize,
}

impl Default for StatGatherer {
    fn default() -> Self {
        Self {
            min: usize::MAX,
            max: usize::MIN,
            sum: 0,
        }
    }
}

impl DeltaGatherer for StatGatherer {
    type Target = Vec<u32>;

    fn target_len(&self, target: &Self::Target) -> usize {
        target.len()
    }

    fn target_reserve(&self, target: &mut Self::Target, n: usize) {
        target.reserve(n);
    }

    fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
        if v < 0 {
            return Err(ParquetError::oos("DELTA_LENGTH_BYTE_ARRAY length < 0"));
        }

        if v > i64::from(u32::MAX) {
            return Err(ParquetError::not_supported(
                "DELTA_LENGTH_BYTE_ARRAY length > u32::MAX",
            ));
        }

        let v = v as usize;

        self.min = self.min.min(v);
        self.max = self.max.max(v);
        self.sum += v;

        target.push(v as u32);

        Ok(())
    }

    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        let mut is_invalid = false;
        let mut is_too_large = false;

        target.extend(slice.iter().map(|&v| {
            is_invalid |= v < 0;
            is_too_large |= v > i64::from(u32::MAX);

            let v = v as usize;

            self.min = self.min.min(v);
            self.max = self.max.max(v);
            self.sum += v;

            v as u32
        }));

        if is_invalid {
            target.truncate(target.len() - slice.len());
            return Err(ParquetError::oos("DELTA_LENGTH_BYTE_ARRAY length < 0"));
        }

        if is_too_large {
            return Err(ParquetError::not_supported(
                "DELTA_LENGTH_BYTE_ARRAY length > u32::MAX",
            ));
        }

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
            return Err(ParquetError::oos("DELTA_LENGTH_BYTE_ARRAY length < 0"));
        }

        if v > i64::from(u32::MAX) || v + ((num_repeats - 1) as i64) * delta > i64::from(u32::MAX) {
            return Err(ParquetError::not_supported(
                "DELTA_LENGTH_BYTE_ARRAY length > u32::MAX",
            ));
        }

        target.extend((0..num_repeats).map(|i| (v + (i as i64) * delta) as u32));

        let vstart = v;
        let vend = v + (num_repeats - 1) as i64 * delta;

        let (min, max) = if delta < 0 {
            (vend, vstart)
        } else {
            (vstart, vend)
        };

        let sum = lin_natural_sum(v, delta, num_repeats) as usize;

        #[cfg(debug_assertions)]
        {
            assert_eq!(
                (0..num_repeats)
                    .map(|i| (v + (i as i64) * delta) as usize)
                    .sum::<usize>(),
                sum
            );
        }

        self.min = self.min.min(min as usize);
        self.max = self.max.max(max as usize);
        self.sum += sum;

        Ok(())
    }
}

impl<'a, 'b> BatchableCollector<(), MutableBinaryViewArray<[u8]>> for &mut DeltaCollector<'a, 'b> {
    fn reserve(target: &mut MutableBinaryViewArray<[u8]>, n: usize) {
        target.reserve(n);
    }

    fn push_n(
        &mut self,
        _target: &mut MutableBinaryViewArray<[u8]>,
        n: usize,
    ) -> ParquetResult<()> {
        self.decoder
            .lengths
            .gather_n_into(self.pushed_lengths, n, self.gatherer)?;

        Ok(())
    }

    fn push_n_nulls(
        &mut self,
        target: &mut MutableBinaryViewArray<[u8]>,
        n: usize,
    ) -> ParquetResult<()> {
        self.flush(target);
        target.extend_constant(n, <Option<&[u8]>>::None);
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.decoder.skip_in_place(n)
    }
}

impl<'a, 'b> DeltaCollector<'a, 'b> {
    pub fn flush(&mut self, target: &mut MutableBinaryViewArray<[u8]>) {
        if !self.pushed_lengths.is_empty() {
            let start_bytes_len = target.total_bytes_len();
            let start_buffer_len = target.total_buffer_len();
            unsafe {
                target.extend_from_lengths_with_stats(
                    &self.decoder.values[self.decoder.offset..],
                    self.pushed_lengths.iter().map(|&v| v as usize),
                    self.gatherer.min,
                    self.gatherer.max,
                    self.gatherer.sum,
                )
            };
            debug_assert_eq!(
                target.total_bytes_len() - start_bytes_len,
                self.gatherer.sum,
            );
            debug_assert_eq!(
                target.total_buffer_len() - start_buffer_len,
                self.pushed_lengths
                    .iter()
                    .map(|&v| v as usize)
                    .filter(|&v| v > View::MAX_INLINE_SIZE as usize)
                    .sum::<usize>(),
            );

            self.decoder.offset += self.gatherer.sum;
            self.pushed_lengths.clear();
            *self.gatherer = StatGatherer::default();
        }
    }
}

impl<'a, 'b> BatchableCollector<(), MutableBinaryViewArray<[u8]>> for DeltaBytesCollector<'a, 'b> {
    fn reserve(target: &mut MutableBinaryViewArray<[u8]>, n: usize) {
        target.reserve(n);
    }

    fn push_n(&mut self, target: &mut MutableBinaryViewArray<[u8]>, n: usize) -> ParquetResult<()> {
        struct MaybeUninitCollector(usize);

        impl DeltaGatherer for MaybeUninitCollector {
            type Target = [MaybeUninit<usize>; BATCH_SIZE];

            fn target_len(&self, _target: &Self::Target) -> usize {
                self.0
            }

            fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

            fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
                target[self.0] = MaybeUninit::new(v as usize);
                self.0 += 1;
                Ok(())
            }
        }

        let decoder_len = self.decoder.len();
        let mut n = usize::min(n, decoder_len);

        if n == 0 {
            return Ok(());
        }

        let mut buffer = Vec::new();
        target.reserve(n);

        const BATCH_SIZE: usize = 4096;

        let mut prefix_lengths = [const { MaybeUninit::<usize>::uninit() }; BATCH_SIZE];
        let mut suffix_lengths = [const { MaybeUninit::<usize>::uninit() }; BATCH_SIZE];

        while n > 0 {
            let num_elems = usize::min(n, BATCH_SIZE);
            n -= num_elems;

            self.decoder.prefix_lengths.gather_n_into(
                &mut prefix_lengths,
                num_elems,
                &mut MaybeUninitCollector(0),
            )?;
            self.decoder.suffix_lengths.gather_n_into(
                &mut suffix_lengths,
                num_elems,
                &mut MaybeUninitCollector(0),
            )?;

            for i in 0..num_elems {
                let prefix_length = unsafe { prefix_lengths[i].assume_init() };
                let suffix_length = unsafe { suffix_lengths[i].assume_init() };

                buffer.clear();

                buffer.extend_from_slice(&self.decoder.last[..prefix_length]);
                buffer.extend_from_slice(
                    &self.decoder.values[self.decoder.offset..self.decoder.offset + suffix_length],
                );

                target.push_value(&buffer);

                self.decoder.last.clear();
                std::mem::swap(&mut self.decoder.last, &mut buffer);

                self.decoder.offset += suffix_length;
            }
        }

        Ok(())
    }

    fn push_n_nulls(
        &mut self,
        target: &mut MutableBinaryViewArray<[u8]>,
        n: usize,
    ) -> ParquetResult<()> {
        target.extend_constant(n, <Option<&[u8]>>::None);
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.decoder.skip_in_place(n)
    }
}

impl utils::Decoder for BinViewDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = (Vec<View>, Vec<Buffer<u8>>);
    type DecodedState = DecodedStateTuple;
    type Output = Box<dyn Array>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn apply_dictionary(
        &mut self,
        (values, _): &mut Self::DecodedState,
        dict: &Self::Dict,
    ) -> ParquetResult<()> {
        if values.completed_buffers().len() < dict.1.len() {
            for buffer in &dict.1 {
                values.push_buffer(buffer.clone());
            }
        }

        assert!(values.completed_buffers().len() == dict.1.len());

        Ok(())
    }

    fn deserialize_dict(&self, page: DictPage) -> ParquetResult<Self::Dict> {
        let values = &page.buffer;
        let num_values = page.num_values;

        // Each value is prepended by the length which is 4 bytes.
        let num_bytes = values.len() - 4 * num_values;

        let mut views = Vec::with_capacity(num_values);
        let mut buffer = Vec::with_capacity(num_bytes);

        let mut buffers = Vec::with_capacity(1);

        let mut offset = 0;
        let mut max_length = 0;
        views.extend(BinaryIter::new(values, num_values).map(|v| {
            let length = v.len();
            max_length = usize::max(length, max_length);
            if length <= View::MAX_INLINE_SIZE as usize {
                View::new_inline(v)
            } else {
                if offset >= u32::MAX as usize {
                    let full_buffer = std::mem::take(&mut buffer);
                    let num_bytes = full_buffer.capacity() - full_buffer.len();
                    buffers.push(Buffer::from(full_buffer));
                    buffer.reserve(num_bytes);
                    offset = 0;
                }

                buffer.extend_from_slice(v);
                let view = View::new_from_bytes(v, buffers.len() as u32, offset as u32);
                offset += v.len();
                view
            }
        }));

        buffers.push(Buffer::from(buffer));

        if self.check_utf8.load(Ordering::Relaxed) {
            // This is a small trick that allows us to check the Parquet buffer instead of the view
            // buffer. Batching the UTF-8 verification is more performant. For this to be allowed,
            // all the interleaved lengths need to be valid UTF-8.
            //
            // Every strings prepended by 4 bytes (L, 0, 0, 0), since we check here L < 128. L is
            // only a valid first byte of a UTF-8 code-point and (L, 0, 0, 0) is valid UTF-8.
            // Consequently, it is valid to just check the whole buffer.
            if max_length < 128 {
                simdutf8::basic::from_utf8(values)
                    .map_err(|_| ParquetError::oos("String data contained invalid UTF-8"))?;
            } else {
                arrow::array::validate_utf8_view(&views, &buffers)
                    .map_err(|_| ParquetError::oos("String data contained invalid UTF-8"))?;
            }
        }

        Ok((views, buffers))
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        let views_offset = values.views().len();
        let buffer_offset = values.completed_buffers().len();

        struct Collector<'a, 'b> {
            iter: &'b mut BinaryIter<'a>,
            max_length: &'b mut usize,
        }

        impl<'a, 'b> BatchableCollector<(), MutableBinaryViewArray<[u8]>> for Collector<'a, 'b> {
            fn reserve(target: &mut MutableBinaryViewArray<[u8]>, n: usize) {
                target.reserve(n);
            }

            fn push_n(
                &mut self,
                target: &mut MutableBinaryViewArray<[u8]>,
                n: usize,
            ) -> ParquetResult<()> {
                for x in self.iter.take(n) {
                    *self.max_length = usize::max(*self.max_length, x.len());
                    target.push_value(x);
                }
                Ok(())
            }

            fn push_n_nulls(
                &mut self,
                target: &mut MutableBinaryViewArray<[u8]>,
                n: usize,
            ) -> ParquetResult<()> {
                target.extend_constant(n, <Option<&[u8]>>::None);
                Ok(())
            }

            fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
                if n > 0 {
                    _ = self.iter.nth(n - 1);
                }
                Ok(())
            }
        }

        let mut max_length = 0;
        let buffer = page_values.values;
        let mut collector = Collector {
            iter: page_values,
            max_length: &mut max_length,
        };

        match page_validity {
            None => {
                collector.push_n(values, limit)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(limit), values, collector)?
            },
        }

        let buffer = &buffer[..buffer.len() - page_values.values.len()];

        if self.check_utf8.load(Ordering::Relaxed) {
            // This is a small trick that allows us to check the Parquet buffer instead of the view
            // buffer. Batching the UTF-8 verification is more performant. For this to be allowed,
            // all the interleaved lengths need to be valid UTF-8.
            //
            // Every strings prepended by 4 bytes (L, 0, 0, 0), since we check here L < 128. L is
            // only a valid first byte of a UTF-8 code-point and (L, 0, 0, 0) is valid UTF-8.
            // Consequently, it is valid to just check the whole buffer.
            if max_length < 128 {
                simdutf8::basic::from_utf8(buffer)
                    .map_err(|_| ParquetError::oos("String data contained invalid UTF-8"))?;
            } else {
                values
                    .validate_utf8(buffer_offset, views_offset)
                    .map_err(|_| ParquetError::oos("String data contained invalid UTF-8"))?
            }
        }

        Ok(())
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut hybrid_rle::HybridRleDecoder<'a>,
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        struct DictionaryTranslator<'a>(&'a [View]);

        impl<'a> HybridRleGatherer<View> for DictionaryTranslator<'a> {
            type Target = MutableBinaryViewArray<[u8]>;

            fn target_reserve(&self, target: &mut Self::Target, n: usize) {
                target.reserve(n);
            }

            fn target_num_elements(&self, target: &Self::Target) -> usize {
                target.len()
            }

            fn hybridrle_to_target(&self, value: u32) -> ParquetResult<View> {
                self.0
                    .get(value as usize)
                    .cloned()
                    .ok_or(ParquetError::oos("Dictionary index is out of range"))
            }

            fn gather_one(&self, target: &mut Self::Target, value: View) -> ParquetResult<()> {
                // SAFETY:
                // - All the dictionary values are already buffered
                // - We keep the `total_bytes_len` in-sync with the views
                unsafe {
                    target.views_mut().push(value);
                    target.set_total_bytes_len(target.total_bytes_len() + value.length as usize);
                }

                Ok(())
            }

            fn gather_repeated(
                &self,
                target: &mut Self::Target,
                value: View,
                n: usize,
            ) -> ParquetResult<()> {
                // SAFETY:
                // - All the dictionary values are already buffered
                // - We keep the `total_bytes_len` in-sync with the views
                unsafe {
                    let length = target.views_mut().len();
                    target.views_mut().resize(length + n, value);
                    target
                        .set_total_bytes_len(target.total_bytes_len() + n * value.length as usize);
                }

                Ok(())
            }

            fn gather_slice(&self, target: &mut Self::Target, source: &[u32]) -> ParquetResult<()> {
                let Some(source_max) = source.iter().copied().max() else {
                    return Ok(());
                };

                if source_max as usize >= self.0.len() {
                    return Err(ParquetError::oos("Dictionary index is out of range"));
                }

                let mut view_length_sum = 0usize;
                // Safety: We have checked before that source only has indexes that are smaller than the
                // dictionary length.
                //
                // Safety:
                // - All the dictionary values are already buffered
                // - We keep the `total_bytes_len` in-sync with the views
                unsafe {
                    target.views_mut().extend(source.iter().map(|&src_idx| {
                        let v = *self.0.get_unchecked(src_idx as usize);
                        view_length_sum += v.length as usize;
                        v
                    }));
                    target.set_total_bytes_len(target.total_bytes_len() + view_length_sum);
                }

                Ok(())
            }
        }

        let translator = DictionaryTranslator(&dict.0);

        match page_validity {
            None => {
                page_values.gather_n_into(values, limit, &translator)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                struct Collector<'a, 'b> {
                    decoder: &'b mut hybrid_rle::HybridRleDecoder<'a>,
                    translator: DictionaryTranslator<'b>,
                }

                impl<'a, 'b> BatchableCollector<(), MutableBinaryViewArray<[u8]>> for Collector<'a, 'b> {
                    fn reserve(target: &mut MutableBinaryViewArray<[u8]>, n: usize) {
                        target.reserve(n);
                    }

                    fn push_n(
                        &mut self,
                        target: &mut MutableBinaryViewArray<[u8]>,
                        n: usize,
                    ) -> ParquetResult<()> {
                        self.decoder.gather_n_into(target, n, &self.translator)?;
                        Ok(())
                    }

                    fn push_n_nulls(
                        &mut self,
                        target: &mut MutableBinaryViewArray<[u8]>,
                        n: usize,
                    ) -> ParquetResult<()> {
                        target.extend_constant(n, <Option<&[u8]>>::None);
                        Ok(())
                    }

                    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
                        self.decoder.skip_in_place(n)
                    }
                }
                let collector = Collector {
                    decoder: page_values,
                    translator,
                };
                extend_from_decoder(validity, page_validity, Some(limit), values, collector)?;
            },
        }

        Ok(())
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        let mut array: BinaryViewArray = values.freeze();

        let validity = freeze_validity(validity);
        array = array.with_validity(validity);

        match data_type.to_physical_type() {
            PhysicalType::BinaryView => Ok(array.boxed()),
            PhysicalType::Utf8View => {
                // SAFETY: we already checked utf8
                unsafe {
                    Ok(Utf8ViewArray::new_unchecked(
                        data_type,
                        array.views().clone(),
                        array.data_buffers().clone(),
                        array.validity().cloned(),
                        array.total_bytes_len(),
                        array.total_buffer_len(),
                    )
                    .boxed())
                }
            },
            _ => unreachable!(),
        }
    }
}

impl utils::DictDecodable for BinViewDecoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_data_type = match &data_type {
            ArrowDataType::Dictionary(_, values, _) => values.as_ref().clone(),
            _ => data_type.clone(),
        };

        let mut view_dict = MutableBinaryViewArray::with_capacity(dict.0.len());
        for buffer in dict.1 {
            view_dict.push_buffer(buffer);
        }
        unsafe { view_dict.views_mut().extend(dict.0.iter()) };
        unsafe { view_dict.set_total_bytes_len(dict.0.iter().map(|v| v.length as usize).sum()) };
        let view_dict = view_dict.freeze();

        let dict = match value_data_type.to_physical_type() {
            PhysicalType::Utf8View => view_dict.to_utf8view().unwrap().boxed(),
            PhysicalType::BinaryView => view_dict.boxed(),
            _ => unreachable!(),
        };

        Ok(DictionaryArray::try_new(data_type, keys, dict).unwrap())
    }
}

impl utils::NestedDecoder for BinViewDecoder {
    fn validity_extend(
        _: &mut utils::State<'_, Self>,
        (_, validity): &mut Self::DecodedState,
        value: bool,
        n: usize,
    ) {
        validity.extend_constant(n, value);
    }

    fn values_extend_nulls(
        _: &mut utils::State<'_, Self>,
        (values, _): &mut Self::DecodedState,
        n: usize,
    ) {
        values.extend_constant(n, <Option<&[u8]>>::None);
    }
}

#[derive(Debug)]
pub struct BinaryIter<'a> {
    values: &'a [u8],

    /// A maximum number of items that this [`BinaryIter`] may produce.
    ///
    /// This equal the length of the iterator i.f.f. the data encoded by the [`BinaryIter`] is not
    /// nullable.
    max_num_values: usize,
}

impl<'a> BinaryIter<'a> {
    pub fn new(values: &'a [u8], max_num_values: usize) -> Self {
        Self {
            values,
            max_num_values,
        }
    }

    /// Return the length of the iterator when the data is not nullable.
    pub fn len_when_not_nullable(&self) -> usize {
        self.max_num_values
    }
}

impl<'a> Iterator for BinaryIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.max_num_values == 0 {
            assert!(self.values.is_empty());
            return None;
        }

        let (length, remaining) = self.values.split_at(4);
        let length: [u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(length) as usize;
        let (result, remaining) = remaining.split_at(length);
        self.max_num_values -= 1;
        self.values = remaining;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.max_num_values))
    }
}
