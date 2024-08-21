use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::{
    Array, BinaryViewArray, DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray,
    Utf8ViewArray, View,
};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};

use super::binary::decoders::*;
use super::utils::{freeze_validity, BatchableCollector};
use crate::parquet::encoding::delta_bitpacked::{lin_natural_sum, DeltaGatherer};
use crate::parquet::encoding::hybrid_rle::{self, DictionaryTranslator};
use crate::parquet::encoding::{delta_byte_array, delta_length_byte_array};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::binary::utils::BinaryIter;
use crate::read::deserialize::utils::{
    self, binary_views_dict, extend_from_decoder, Decoder, PageValidity, StateTranslation,
    TranslatedHybridRle,
};
use crate::read::PrimitiveLogicalType;

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> StateTranslation<'a, BinViewDecoder> for BinaryStateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        decoder: &BinViewDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&PageValidity<'a>>,
    ) -> ParquetResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
        Self::new(page, dict, page_validity, is_string)
    }

    fn len_when_not_nullable(&self) -> usize {
        Self::len_when_not_nullable(self)
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        Self::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut BinViewDecoder,
        decoded: &mut <BinViewDecoder as utils::Decoder>::DecodedState,
        page_validity: &mut Option<utils::PageValidity<'a>>,
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
                    page_validity.as_mut(),
                    additional,
                )?;

                // Already done in decode_plain_encoded
                validate_utf8 = false;
            },
            Self::Dictionary(page) => {
                decoder.decode_dictionary_encoded(
                    decoded,
                    &mut page.values,
                    page_validity.as_mut(),
                    page.dict,
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
                    None => (&mut collector).push_n(values, additional)?,
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
                    None => collector.push_n(values, additional)?,
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
    views_dict: Option<Vec<View>>,
}

impl utils::ExactSize for DecodedStateTuple {
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
        target.views_mut().reserve(n);
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
}

impl<'a, 'b> DeltaCollector<'a, 'b> {
    pub fn flush(&mut self, target: &mut MutableBinaryViewArray<[u8]>) {
        if !self.pushed_lengths.is_empty() {
            unsafe {
                target.extend_from_lengths_with_stats(
                    &self.decoder.values[self.decoder.offset..],
                    self.pushed_lengths.iter().map(|&v| v as usize),
                    self.gatherer.min,
                    self.gatherer.max,
                    self.gatherer.sum,
                )
            };

            self.decoder.offset += self.gatherer.sum;
            self.pushed_lengths.clear();
            *self.gatherer = StatGatherer::default();
        }
    }
}

impl<'a, 'b> BatchableCollector<(), MutableBinaryViewArray<[u8]>> for DeltaBytesCollector<'a, 'b> {
    fn reserve(target: &mut MutableBinaryViewArray<[u8]>, n: usize) {
        target.views_mut().reserve(n);
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
        target.views_mut().reserve(n);

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
}

impl utils::Decoder for BinViewDecoder {
    type Translation<'a> = BinaryStateTranslation<'a>;
    type Dict = BinaryDict;
    type DecodedState = DecodedStateTuple;
    type Output = Box<dyn Array>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as StateTranslation<'a, Self>>::PlainDecoder,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        let views_offset = values.views().len();
        let buffer_offset = values.completed_buffers().len();

        match page_validity {
            None => {
                for x in page_values.by_ref().take(limit) {
                    values.push_value_ignore_validity(x)
                }
            },
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(limit), values, page_values)?
            },
        }

        if self.check_utf8.load(Ordering::Relaxed) {
            // @TODO: Better error message
            values
                .validate_utf8(buffer_offset, views_offset)
                .map_err(|_| ParquetError::oos("Binary view contained invalid UTF-8"))?
        }

        Ok(())
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut hybrid_rle::HybridRleDecoder<'a>,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        let validate_utf8 = self.check_utf8.load(Ordering::Relaxed);

        if validate_utf8 && simdutf8::basic::from_utf8(dict.values()).is_err() {
            return Err(ParquetError::oos(
                "Binary view dictionary contained invalid UTF-8",
            ));
        }

        let views_dict = self
            .views_dict
            .get_or_insert_with(|| binary_views_dict(values, dict));
        let translator = DictionaryTranslator(views_dict);

        match page_validity {
            None => {
                page_values.translate_and_collect_n_into(values.views_mut(), limit, &translator)?;
                if let Some(validity) = values.validity() {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                let collector = TranslatedHybridRle::new(page_values, &translator);
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

        let mut view_dict = MutableBinaryViewArray::with_capacity(dict.len());
        for v in dict.iter() {
            view_dict.push(v);
        }
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
