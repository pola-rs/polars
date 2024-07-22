use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::nested_utils::{InitNested, NestedState};
use super::super::utils::MaybeNext;
use super::basic::deserialize_plain;
use super::DecoderFunction;
use crate::parquet::encoding::hybrid_rle::Translator;
use crate::parquet::encoding::{hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::parquet::read::BasicDecompressor;
use crate::parquet::types::NativeType as ParquetNativeType;
use crate::read::deserialize::utils::{self, BatchableCollector, DictArrayDecoder};
use crate::read::CompressedPagesIter;

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
        Ok(unsafe { K::from_usize_unchecked(value as usize) })
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

pub(crate) struct StateTranslation<'a, K: DictionaryKey, D: DictArrayDecoder<K>> {
    values: hybrid_rle::HybridRleDecoder<'a>,
    _pd: std::marker::PhantomData<(K, D)>,
}

impl<'a, K: DictionaryKey, D: DictArrayDecoder<K>> utils::DictArrayStateTranslation<'a, K, D>
    for StateTranslation<'a, K, D>
{
    fn new(
        _decoder: &D,
        page: &'a DataPage,
        _dict: &'a <D as DictArrayDecoder<K>>::Dict,
        _page_validity: Option<&utils::PageValidity<'a>>,
        _filter: Option<&utils::filter::Filter<'a>>,
    ) -> ParquetResult<Self> {
        if !matches!(
            page.encoding(),
            Encoding::PlainDictionary | Encoding::RleDictionary
        ) {
            return Err(ParquetError::FeatureNotSupported(
                "Dictionary array with data pages that does not reference dictionary page"
                    .to_string(),
            ));
        }

        Ok(StateTranslation {
            values: utils::dict_indices_decoder(page)?,
            _pd: std::marker::PhantomData,
        })
    }

    fn len_when_not_nullable(&self) -> usize {
        self.values.len()
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.values.skip_in_place(n)
    }

    fn extend_from_state(
        &mut self,
        _decoder: &D,
        decoded: &mut (Vec<K>, MutableBitmap),
        dict: &'a <D as DictArrayDecoder<K>>::Dict,
        page_validity: &mut Option<utils::PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        use utils::ExactSize;

        match page_validity {
            Some(page_validity) => {
                let collector = DictArrayCollector {
                    values: &mut self.values,
                    dict_size: dict.len(),
                };

                utils::extend_from_decoder(
                    &mut decoded.1,
                    page_validity,
                    Some(additional),
                    &mut decoded.0,
                    collector,
                )?;
            },
            None => {
                let translator = DictArrayTranslator {
                    dict_size: dict.len(),
                };

                self.values.translate_and_collect_n_into(
                    &mut decoded.0,
                    additional,
                    &translator,
                )?;
            },
        }

        Ok(())
    }
}

impl<T> utils::ExactSize for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

pub(crate) struct PrimitiveDictArrayDecoder<
    K: DictionaryKey,
    P: ParquetNativeType,
    T: NativeType,
    F: DecoderFunction<P, T>,
> {
    dfn: F,
    _pd: std::marker::PhantomData<(K, P, T)>,
}

impl<K, P, T, F> PrimitiveDictArrayDecoder<K, P, T, F>
where
    K: DictionaryKey,
    P: ParquetNativeType,
    T: NativeType,
    F: DecoderFunction<P, T>,
{
    pub(crate) fn new(dfn: F) -> Self {
        Self {
            dfn,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<K, P, T, F> DictArrayDecoder<K> for PrimitiveDictArrayDecoder<K, P, T, F>
where
    K: DictionaryKey,
    P: ParquetNativeType,
    T: NativeType,
    F: DecoderFunction<P, T>,
{
    type Translation<'a> = StateTranslation<'a, K, Self>;
    type Dict = Vec<T>;

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, self.dfn)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        (values, validity): (Vec<K>, Option<Bitmap>),
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_type = match &data_type {
            ArrowDataType::Dictionary(_, value, _) => value.as_ref().clone(),
            _ => T::PRIMITIVE.into(),
        };

        let array = PrimitiveArray::<K>::new(K::PRIMITIVE.into(), values.into(), validity);
        let dict = Box::new(PrimitiveArray::new(value_type, dict.into(), None));

        Ok(DictionaryArray::try_new(data_type, array, dict).unwrap())
    }
}

fn read_dict<P, T, D>(data_type: ArrowDataType, dict: &DictPage, decoder: D) -> Box<dyn Array>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };
    let values = deserialize_plain::<P, T, D>(&dict.buffer, decoder);
    Box::new(PrimitiveArray::new(data_type, values.into(), None))
}

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`]
pub struct NestedDictIter<K, T, I, P, D>
where
    I: CompressedPagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    iter: BasicDecompressor<I>,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: D,
    phantom: std::marker::PhantomData<(P, T)>,
}

impl<K, T, I, P, D> NestedDictIter<K, T, I, P, D>
where
    K: DictionaryKey,
    I: CompressedPagesIter,
    T: NativeType,

    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub fn new(
        iter: BasicDecompressor<I>,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        decoder: D,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            values: None,
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
            decoder,
            phantom: Default::default(),
        }
    }
}

impl<K, T, I, P, D> Iterator for NestedDictIter<K, T, I, P, D>
where
    I: CompressedPagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type Item = PolarsResult<(NestedState, DictionaryArray<K>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = nested_next_dict(
                &mut self.iter,
                &mut self.items,
                &mut self.remaining,
                &self.init,
                &mut self.values,
                self.data_type.clone(),
                self.chunk_size,
                |dict| read_dict(self.data_type.clone(), dict, self.decoder),
            );
            match maybe_state {
                MaybeNext::Some(Ok(dict)) => return Some(Ok(dict)),
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
