use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::{polars_err, PolarsResult};

use super::super::super::PagesIter;
use super::super::nested_utils::*;
use super::super::utils::{dict_indices_decoder, not_implemented, MaybeNext, PageState};
use super::finish_key;
use crate::parquet::encoding::hybrid_rle::{HybridRleDecoder, TryFromUsizeTranslator};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage, Page};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub struct State<'a> {
    is_optional: bool,
    values: HybridRleDecoder<'a>,
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug)]
pub struct DictionaryDecoder<K>
where
    K: DictionaryKey,
{
    phantom_k: std::marker::PhantomData<K>,
}

impl<K> Default for DictionaryDecoder<K>
where
    K: DictionaryKey,
{
    #[inline]
    fn default() -> Self {
        Self {
            phantom_k: std::marker::PhantomData,
        }
    }
}

impl<'a, K: DictionaryKey> NestedDecoder<'a> for DictionaryDecoder<K> {
    type State = State<'a>;
    type Dictionary = ();
    type DecodedState = (Vec<K>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        _: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        if is_filtered {
            return Err(not_implemented(page));
        }

        let values = match page.encoding() {
            Encoding::RleDictionary | Encoding::PlainDictionary => dict_indices_decoder(page)?,
            _ => return Err(not_implemented(page)),
        };

        Ok(State {
            is_optional,
            values,
        })
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_n_valid(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        if state.len() < n {
            return Err(ParquetError::oos("No values in page left"));
        }

        let translator = <TryFromUsizeTranslator<K>>::default();
        state
            .values
            .translate_and_collect_n_into(values, n, &translator)?;

        if state.is_optional {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        let (values, validity) = decoded;
        values.resize(values.len() + n, K::default());
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, _: &DictPage) -> Self::Dictionary {}
}

#[allow(clippy::too_many_arguments)]
pub fn next_dict<K: DictionaryKey, I: PagesIter, F: Fn(&DictPage) -> Box<dyn Array>>(
    iter: &mut I,
    items: &mut VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: &mut usize,
    init: &[InitNested],
    dict: &mut Option<Box<dyn Array>>,
    data_type: ArrowDataType,
    chunk_size: Option<usize>,
    read_dict: F,
) -> MaybeNext<PolarsResult<(NestedState, DictionaryArray<K>)>> {
    if items.len() > 1 {
        let (nested, (values, validity)) = items.pop_front().unwrap();
        let keys = finish_key(values, validity);
        let dict = DictionaryArray::try_new(data_type, keys, dict.clone().unwrap());
        return MaybeNext::Some(dict.map(|dict| (nested, dict)));
    }
    match iter.next() {
        Err(e) => MaybeNext::Some(Err(e.into())),
        Ok(Some(page)) => {
            let (page, dict) = match (&dict, &page) {
                (None, Page::Data(_)) => {
                    return MaybeNext::Some(Err(polars_err!(ComputeError:
                        "not implemented: dictionary arrays from non-dict-encoded pages",
                    )));
                },
                (_, Page::Dict(dict_page)) => {
                    *dict = Some(read_dict(dict_page));
                    return next_dict(
                        iter, items, remaining, init, dict, data_type, chunk_size, read_dict,
                    );
                },
                (Some(dict), Page::Data(page)) => (page, dict),
            };

            let error = extend(
                page,
                init,
                items,
                None,
                remaining,
                &DictionaryDecoder::<K>::default(),
                chunk_size,
            );
            match error {
                Ok(_) => {},
                Err(e) => return MaybeNext::Some(Err(e)),
            };

            if items.front().unwrap().0.len() < chunk_size.unwrap_or(usize::MAX) {
                MaybeNext::More
            } else {
                let (nested, (values, validity)) = items.pop_front().unwrap();
                let keys = finish_key(values, validity);
                let dict = DictionaryArray::try_new(data_type, keys, dict.clone());
                MaybeNext::Some(dict.map(|dict| (nested, dict)))
            }
        },
        Ok(None) => {
            if let Some((nested, (values, validity))) = items.pop_front() {
                // we have a populated item and no more pages
                // the only case where an item's length may be smaller than chunk_size
                debug_assert!(values.len() <= chunk_size.unwrap_or(usize::MAX));

                let keys = finish_key(values, validity);
                let dict = DictionaryArray::try_new(data_type, keys, dict.clone().unwrap());
                MaybeNext::Some(dict.map(|dict| (nested, dict)))
            } else {
                MaybeNext::None
            }
        },
    }
}
