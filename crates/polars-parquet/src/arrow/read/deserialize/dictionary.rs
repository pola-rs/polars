use arrow::array::{Array, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::utils::filter::Filter;
use super::utils::{
    self, dict_indices_decoder, extend_from_decoder, BatchableCollector, Decoder, PageValidity,
    StateTranslation,
};
use super::ParquetError;
use crate::parquet::encoding::hybrid_rle::{self, HybridRleDecoder, Translator};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

impl<'a, K: DictionaryKey> StateTranslation<'a, DictionaryDecoder<K>> for HybridRleDecoder<'a> {
    type PlainDecoder = HybridRleDecoder<'a>;

    fn new(
        _decoder: &DictionaryDecoder<K>,
        page: &'a DataPage,
        _dict: Option<&'a <DictionaryDecoder<K> as Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
        _filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
        if !matches!(
            page.encoding(),
            Encoding::PlainDictionary | Encoding::RleDictionary
        ) {
            return Err(utils::not_implemented(page));
        }

        Ok(dict_indices_decoder(page)?)
    }

    fn len_when_not_nullable(&self) -> usize {
        self.len()
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        HybridRleDecoder::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut DictionaryDecoder<K>,
        decoded: &mut <DictionaryDecoder<K> as Decoder>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        let mut collector = DictArrayCollector {
            values: self,
            dict_size: decoder.dict_size,
        };

        match page_validity {
            None => collector.push_n(&mut decoded.0, additional)?,
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(additional), values, collector)?
            },
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct DictionaryDecoder<K: DictionaryKey> {
    dict_size: usize,
    _pd: std::marker::PhantomData<K>,
}

impl<K: DictionaryKey> DictionaryDecoder<K> {
    pub fn new(dict_size: usize) -> Self {
        Self {
            dict_size,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<K: DictionaryKey> utils::Decoder for DictionaryDecoder<K> {
    type Translation<'a> = HybridRleDecoder<'a>;
    type Dict = ();
    type DecodedState = (Vec<K>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<K>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, _: DictPage) -> Self::Dict {}

    fn finalize(
        &self,
        _data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        Ok(Box::new(PrimitiveArray::new(
            K::PRIMITIVE.into(),
            values.into(),
            validity.into(),
        )))
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut <Self::Translation<'a> as StateTranslation<'a, Self>>::PlainDecoder,
        _page_validity: Option<&mut PageValidity<'a>>,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut HybridRleDecoder<'a>,
        _page_validity: Option<&mut PageValidity<'a>>,
        _dict: &Self::Dict,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
    }

    fn finalize_dict_array<K2: DictionaryKey>(
        &self,
        _data_type: ArrowDataType,
        _dict: Self::Dict,
        _decoded: (Vec<K2>, Option<arrow::bitmap::Bitmap>),
    ) -> ParquetResult<arrow::array::DictionaryArray<K2>> {
        unimplemented!()
    }
}

impl<K: DictionaryKey> utils::NestedDecoder for DictionaryDecoder<K> {
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
