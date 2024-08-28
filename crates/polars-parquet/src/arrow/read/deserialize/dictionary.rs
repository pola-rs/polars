use std::sync::atomic::AtomicUsize;

use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;

use super::utils::{
    self, dict_indices_decoder, extend_from_decoder, freeze_validity, BatchableCollector, Decoder,
    DictDecodable, ExactSize, PageValidity, StateTranslation,
};
use super::ParquetError;
use crate::parquet::encoding::hybrid_rle::{self, HybridRleDecoder, Translator};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

impl<'a, K: DictionaryKey, D: utils::DictDecodable> StateTranslation<'a, DictionaryDecoder<K, D>>
    for HybridRleDecoder<'a>
{
    type PlainDecoder = HybridRleDecoder<'a>;

    fn new(
        _decoder: &DictionaryDecoder<K, D>,
        page: &'a DataPage,
        _dict: Option<&'a <DictionaryDecoder<K, D> as Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
    ) -> ParquetResult<Self> {
        if !matches!(
            page.encoding(),
            Encoding::PlainDictionary | Encoding::RleDictionary
        ) {
            return Err(utils::not_implemented(page));
        }

        dict_indices_decoder(page)
    }

    fn len_when_not_nullable(&self) -> usize {
        self.len()
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        HybridRleDecoder::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut DictionaryDecoder<K, D>,
        decoded: &mut <DictionaryDecoder<K, D> as Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<PageValidity<'a>>,
        _: Option<&'a <DictionaryDecoder<K, D> as Decoder>::Dict>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        let dict_size = decoder.dict_size.load(std::sync::atomic::Ordering::Relaxed);

        if dict_size == usize::MAX {
            panic!("Dictionary not set for dictionary array");
        }

        let mut collector = DictArrayCollector {
            values: self,
            dict_size,
        };

        match page_validity {
            None => {
                collector.push_n(&mut decoded.0, additional)?;

                if is_optional {
                    validity.extend_constant(additional, true);
                }
            },
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(additional), values, collector)?
            },
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct DictionaryDecoder<K: DictionaryKey, D: utils::DictDecodable> {
    dict_size: AtomicUsize,
    decoder: D,
    _pd: std::marker::PhantomData<K>,
}

impl<K: DictionaryKey, D: utils::DictDecodable> DictionaryDecoder<K, D> {
    pub fn new(decoder: D) -> Self {
        Self {
            dict_size: AtomicUsize::new(usize::MAX),
            decoder,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<K: DictionaryKey, D: utils::DictDecodable> utils::Decoder for DictionaryDecoder<K, D> {
    type Translation<'a> = HybridRleDecoder<'a>;
    type Dict = D::Dict;
    type DecodedState = (Vec<K>, MutableBitmap);
    type Output = DictionaryArray<K>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<K>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> ParquetResult<Self::Dict> {
        let dict = self.decoder.deserialize_dict(page)?;
        self.dict_size
            .store(dict.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(dict)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<DictionaryArray<K>> {
        let validity = freeze_validity(validity);
        let dict = dict.unwrap();
        let keys = PrimitiveArray::new(K::PRIMITIVE.into(), values.into(), validity);

        self.decoder.finalize_dict_array(data_type, dict, keys)
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut <Self::Translation<'a> as StateTranslation<'a, Self>>::PlainDecoder,
        _is_optional: bool,
        _page_validity: Option<&mut PageValidity<'a>>,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut HybridRleDecoder<'a>,
        _is_optional: bool,
        _page_validity: Option<&mut PageValidity<'a>>,
        _dict: &Self::Dict,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
    }
}

impl<K: DictionaryKey, D: DictDecodable> utils::NestedDecoder for DictionaryDecoder<K, D> {
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
        values.resize(values.len() + n, K::default());
    }
}

pub(crate) struct DictArrayCollector<'a, 'b> {
    values: &'b mut hybrid_rle::HybridRleDecoder<'a>,
    dict_size: usize,
}

pub(crate) struct DictArrayTranslator {
    dict_size: usize,
}

impl<'a, 'b, K: DictionaryKey> BatchableCollector<(), Vec<K>> for DictArrayCollector<'a, 'b> {
    fn reserve(target: &mut Vec<K>, n: usize) {
        target.reserve(n);
    }

    fn push_n(&mut self, target: &mut Vec<K>, n: usize) -> ParquetResult<()> {
        let translator = DictArrayTranslator {
            dict_size: self.dict_size,
        };
        self.values
            .translate_and_collect_n_into(target, n, &translator)
    }

    fn push_n_nulls(&mut self, target: &mut Vec<K>, n: usize) -> ParquetResult<()> {
        target.resize(target.len() + n, K::default());
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.values.skip_in_place(n)
    }
}

impl<K: DictionaryKey> Translator<K> for DictArrayTranslator {
    fn translate(&self, value: u32) -> ParquetResult<K> {
        let value = value as usize;

        if value >= self.dict_size || value > K::MAX_USIZE_VALUE {
            return Err(ParquetError::oos("Dictionary index out-of-range"));
        }

        // SAFETY: value for sure fits in K
        Ok(unsafe { K::from_usize_unchecked(value) })
    }

    fn translate_slice(&self, target: &mut Vec<K>, source: &[u32]) -> ParquetResult<()> {
        let Some(max) = source.iter().max() else {
            return Ok(());
        };

        let max = *max as usize;

        if max >= self.dict_size || max > K::MAX_USIZE_VALUE {
            return Err(ParquetError::oos("Dictionary index out-of-range"));
        }

        // SAFETY: value for sure fits in K
        target.extend(
            source
                .iter()
                .map(|v| unsafe { K::from_usize_unchecked(*v as usize) }),
        );

        Ok(())
    }
}
