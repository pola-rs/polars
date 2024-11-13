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
    dict_size: usize,
    decoder: D,
    _pd: std::marker::PhantomData<K>,
}

impl<K: DictionaryKey, D: utils::DictDecodable> DictionaryDecoder<K, D> {
    pub fn new(decoder: D) -> Self {
        Self {
            dict_size: usize::MAX,
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
        self.dict_size = dict.len();
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
        let keys = state.translation.collect()?;
        let num_rows = keys.len();
        let mut iter = keys.into_iter();

        let dict_size = self.dict_size;

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
    }
}
