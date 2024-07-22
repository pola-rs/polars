mod nested;

use arrow::array::{Array, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
pub use nested::next_dict as nested_next_dict;
use polars_error::PolarsResult;
use polars_utils::iter::FallibleIterator;

use super::utils::filter::Filter;
use super::utils::{
    self, dict_indices_decoder, extend_from_decoder, Decoder, PageValidity, StateTranslation,
};
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

impl<'a, K: DictionaryKey> StateTranslation<'a, PrimitiveDecoder<K>> for HybridRleDecoder<'a> {
    fn new(
        _decoder: &PrimitiveDecoder<K>,
        page: &'a DataPage,
        _dict: Option<&'a <PrimitiveDecoder<K> as Decoder>::Dict>,
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
        _decoder: &PrimitiveDecoder<K>,
        decoded: &mut <PrimitiveDecoder<K> as Decoder>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        match page_validity {
            None => {
                values.extend(
                    self.by_ref()
                        .map(|x| {
                            let x: K = match (x as usize).try_into() {
                                Ok(key) => key,
                                // todo: convert this to an error.
                                Err(_) => {
                                    panic!("The maximum key is too small")
                                },
                            };
                            x
                        })
                        .take(additional),
                );
                self.get_result()?;
            },
            Some(page_validity) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut self.by_ref().map(|x| {
                        match (x as usize).try_into() {
                            Ok(key) => key,
                            // todo: convert this to an error.
                            Err(_) => panic!("The maximum key is too small"),
                        }
                    }),
                )?;
                self.get_result()?;
            },
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct PrimitiveDecoder<K: DictionaryKey> {
    _pd: std::marker::PhantomData<K>,
}

impl<K> Default for PrimitiveDecoder<K>
where
    K: DictionaryKey,
{
    #[inline]
    fn default() -> Self {
        Self {
            _pd: std::marker::PhantomData,
        }
    }
}

impl<K: DictionaryKey> utils::Decoder for PrimitiveDecoder<K> {
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
}

fn finish_key<K: DictionaryKey>(values: Vec<K>, validity: MutableBitmap) -> PrimitiveArray<K> {
    PrimitiveArray::new(K::PRIMITIVE.into(), values.into(), validity.into())
}
