use std::sync::atomic::AtomicUsize;

use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;

use super::utils::{
    self, dict_indices_decoder, freeze_validity, unspecialized_decode, Decoder, ExactSize,
    StateTranslation,
};
use super::ParquetError;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
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
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        if !matches!(
            page.encoding(),
            Encoding::PlainDictionary | Encoding::RleDictionary
        ) {
            return Err(utils::not_implemented(page));
        }

        dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))
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

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let dict = self.decoder.deserialize_dict(page)?;
        self.dict_size
            .store(dict.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(dict)
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<DictionaryArray<K>> {
        let validity = freeze_validity(validity);
        let dict = dict.unwrap();
        let keys = PrimitiveArray::new(K::PRIMITIVE.into(), values.into(), validity);

        self.decoder.finalize_dict_array(dtype, dict, keys)
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<super::Filter>,
    ) -> ParquetResult<()> {
<<<<<<< HEAD
        unreachable!()
    }
}

pub(crate) struct DictArrayCollector<'a, 'b> {
    values: &'b mut hybrid_rle::HybridRleDecoder<'a>,
    dict_size: usize,
}

pub(crate) struct DictArrayTranslator {
    dict_size: usize,
}

impl<K: DictionaryKey> BatchableCollector<(), Vec<K>> for DictArrayCollector<'_, '_> {
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
=======
        let keys = state.translation.collect()?;
        let num_rows = keys.len();
        let mut iter = keys.into_iter();

        let dict_size = self.dict_size.load(std::sync::atomic::Ordering::Relaxed);

        unspecialized_decode(
            num_rows,
            || {
                let value = iter.next().unwrap();

                let value = value as usize;

                if value >= dict_size || value > K::MAX_USIZE_VALUE {
                    return Err(ParquetError::oos("Dictionary index out-of-range"));
                }

                // SAFETY: value for sure fits in K
                Ok(unsafe { K::from_usize_unchecked(value) })
            },
            filter,
            state.page_validity,
            state.is_optional,
            &mut decoded.1,
            &mut decoded.0,
        )
>>>>>>> 9f6aea944d (remove a whole load of unused code)
    }
}
