use std::collections::VecDeque;

use parquet2::{
    encoding::{hybrid_rle::HybridRleDecoder, Encoding},
    page::{DataPage, DictPage, Page},
    schema::Repetition,
};

use crate::datatypes::DataType;
use crate::{
    array::{Array, DictionaryArray, DictionaryKey},
    bitmap::MutableBitmap,
    error::{Error, Result},
};

use super::{
    super::super::Pages,
    super::nested_utils::*,
    super::utils::{dict_indices_decoder, not_implemented, MaybeNext, PageState},
    finish_key,
};

// The state of a required DataPage with a boolean physical type
#[derive(Debug)]
pub struct Required<'a> {
    values: HybridRleDecoder<'a>,
    length: usize,
}

impl<'a> Required<'a> {
    fn try_new(page: &'a DataPage) -> Result<Self> {
        let values = dict_indices_decoder(page)?;
        let length = page.num_values();
        Ok(Self { values, length })
    }
}

// The state of a `DataPage` of a `Dictionary` type
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum State<'a> {
    Optional(HybridRleDecoder<'a>),
    Required(Required<'a>),
}

impl<'a> State<'a> {
    pub fn len(&self) -> usize {
        match self {
            State::Optional(page) => page.len(),
            State::Required(page) => page.length,
        }
    }
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        self.len()
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
    ) -> Result<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        match (page.encoding(), is_optional, is_filtered) {
            (Encoding::RleDictionary | Encoding::PlainDictionary, true, false) => {
                dict_indices_decoder(page).map(State::Optional)
            }
            (Encoding::RleDictionary | Encoding::PlainDictionary, false, false) => {
                Required::try_new(page).map(State::Required)
            }
            _ => Err(not_implemented(page)),
        }
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_valid(&self, state: &mut Self::State, decoded: &mut Self::DecodedState) -> Result<()> {
        let (values, validity) = decoded;
        match state {
            State::Optional(page_values) => {
                let key = page_values.next().transpose()?;
                // todo: convert unwrap to error
                let key = match K::try_from(key.unwrap_or_default() as usize) {
                    Ok(key) => key,
                    Err(_) => todo!(),
                };
                values.push(key);
                validity.push(true);
            }
            State::Required(page_values) => {
                let key = page_values.values.next().transpose()?;
                let key = match K::try_from(key.unwrap_or_default() as usize) {
                    Ok(key) => key,
                    Err(_) => todo!(),
                };
                values.push(key);
            }
        }
        Ok(())
    }

    fn push_null(&self, decoded: &mut Self::DecodedState) {
        let (values, validity) = decoded;
        values.push(K::default());
        validity.push(false)
    }

    fn deserialize_dict(&self, _: &DictPage) -> Self::Dictionary {}
}

#[allow(clippy::too_many_arguments)]
pub fn next_dict<K: DictionaryKey, I: Pages, F: Fn(&DictPage) -> Box<dyn Array>>(
    iter: &mut I,
    items: &mut VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: &mut usize,
    init: &[InitNested],
    dict: &mut Option<Box<dyn Array>>,
    data_type: DataType,
    chunk_size: Option<usize>,
    read_dict: F,
) -> MaybeNext<Result<(NestedState, DictionaryArray<K>)>> {
    if items.len() > 1 {
        let (nested, (values, validity)) = items.pop_front().unwrap();
        let keys = finish_key(values, validity);
        let dict = DictionaryArray::try_new(data_type, keys, dict.clone().unwrap());
        return MaybeNext::Some(dict.map(|dict| (nested, dict)));
    }
    match iter.next() {
        Err(e) => MaybeNext::Some(Err(e.into())),
        Ok(Some(page)) => {
            let (page, dict) = match (&dict, page) {
                (None, Page::Data(_)) => {
                    return MaybeNext::Some(Err(Error::nyi(
                        "dictionary arrays from non-dict-encoded pages",
                    )));
                }
                (_, Page::Dict(dict_page)) => {
                    *dict = Some(read_dict(dict_page));
                    return next_dict(
                        iter, items, remaining, init, dict, data_type, chunk_size, read_dict,
                    );
                }
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
                Ok(_) => {}
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
        }
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
        }
    }
}
