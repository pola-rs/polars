mod nested;

use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
pub use nested::next_dict as nested_next_dict;
use polars_error::{polars_err, PolarsResult};
use polars_utils::iter::FallibleIterator;

use super::utils::filter::Filter;
use super::utils::{
    self, dict_indices_decoder, extend_from_decoder, DecodedState, Decoder, MaybeNext,
    PageValidity, StateTranslation,
};
use super::PagesIter;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage, Page};

impl<'a, K: DictionaryKey> StateTranslation<'a, PrimitiveDecoder<K>> for HybridRleDecoder<'a> {
    fn new(
        _decoder: &PrimitiveDecoder<K>,
        page: &'a DataPage,
        _dict: Option<&'a <PrimitiveDecoder<K> as Decoder<'a>>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
        _filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
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
        _decoder: &PrimitiveDecoder<K>,
        decoded: &mut <PrimitiveDecoder<K> as Decoder<'a>>::DecodedState,
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

impl<'a, K: DictionaryKey> utils::Decoder<'a> for PrimitiveDecoder<K> {
    type Translation = HybridRleDecoder<'a>;
    type Dict = ();
    type DecodedState = (Vec<K>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<K>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, _: &DictPage) -> Self::Dict {}
}

fn finish_key<K: DictionaryKey>(values: Vec<K>, validity: MutableBitmap) -> PrimitiveArray<K> {
    PrimitiveArray::new(K::PRIMITIVE.into(), values.into(), validity.into())
}

#[inline]
pub(super) fn next_dict<K: DictionaryKey, I: PagesIter, F: Fn(&DictPage) -> Box<dyn Array>>(
    iter: &mut I,
    items: &mut VecDeque<(Vec<K>, MutableBitmap)>,
    dict: &mut Option<Box<dyn Array>>,
    data_type: ArrowDataType,
    remaining: &mut usize,
    chunk_size: Option<usize>,
    read_dict: F,
) -> MaybeNext<PolarsResult<DictionaryArray<K>>> {
    if items.len() > 1 {
        let (values, validity) = items.pop_front().unwrap();
        let keys = finish_key(values, validity);
        return MaybeNext::Some(DictionaryArray::try_new(
            data_type,
            keys,
            dict.clone().unwrap(),
        ));
    }
    match iter.next() {
        Err(e) => MaybeNext::Some(Err(e.into())),
        Ok(Some(page)) => {
            let (page, dict) = match (&dict, page) {
                (None, Page::Data(_)) => {
                    return MaybeNext::Some(Err(polars_err!(ComputeError:
                        "not implemented: dictionary arrays from non-dict-encoded pages",
                    )));
                },
                (_, Page::Dict(dict_page)) => {
                    *dict = Some(read_dict(dict_page));
                    return next_dict(
                        iter, items, dict, data_type, remaining, chunk_size, read_dict,
                    );
                },
                (Some(dict), Page::Data(page)) => (page, dict),
            };

            // there is a new page => consume the page from the start
            let maybe_page = utils::State::new(&PrimitiveDecoder::<K>::default(), page, None);
            let page = match maybe_page {
                Ok(page) => page,
                Err(e) => return MaybeNext::Some(Err(e)),
            };

            if let Err(e) = utils::extend_from_new_page(
                page,
                chunk_size,
                items,
                remaining,
                &PrimitiveDecoder::<K>::default(),
            ) {
                return MaybeNext::Some(Err(e));
            }

            if items.front().unwrap().len() < chunk_size.unwrap_or(usize::MAX) {
                MaybeNext::More
            } else {
                let (values, validity) = items.pop_front().unwrap();
                let keys = finish_key(values, validity);
                MaybeNext::Some(DictionaryArray::try_new(data_type, keys, dict.clone()))
            }
        },
        Ok(None) => {
            if let Some((values, validity)) = items.pop_front() {
                // we have a populated item and no more pages
                // the only case where an item's length may be smaller than chunk_size
                debug_assert!(values.len() <= chunk_size.unwrap_or(usize::MAX));

                let keys = finish_key(values, validity);
                MaybeNext::Some(DictionaryArray::try_new(
                    data_type,
                    keys,
                    dict.clone().unwrap(),
                ))
            } else {
                MaybeNext::None
            }
        },
    }
}
