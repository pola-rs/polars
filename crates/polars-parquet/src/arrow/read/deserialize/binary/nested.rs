use std::collections::VecDeque;

use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use arrow::pushable::Pushable;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils::MaybeNext;
use super::basic::finish;
use super::decoders::*;
use super::utils::*;
use crate::arrow::read::PagesIter;
use crate::parquet::encoding::hybrid_rle::{DictionaryTranslator, Translator};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::{
    not_implemented, page_is_filtered, page_is_optional, PageState,
};

#[derive(Debug)]
pub struct State<'a> {
    is_optional: bool,
    translation: StateTranslation<'a>,
}

#[derive(Debug)]
pub enum StateTranslation<'a> {
    Unit(BinaryIter<'a>),
    Dictionary(ValuesDictionary<'a>, Option<Vec<&'a [u8]>>),
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Unit(iter) => iter.size_hint().0,
            StateTranslation::Dictionary(values, _) => values.len(),
        }
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder<O: Offset>(std::marker::PhantomData<O>);

impl<'a, O: Offset> NestedDecoder<'a> for BinaryDecoder<O> {
    type State = State<'a>;
    type Dictionary = BinaryDict;
    type DecodedState = (Binary<O>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional = page_is_optional(page);
        let is_filtered = page_is_filtered(page);

        if is_filtered {
            return Err(not_implemented(page));
        }

        let translation = match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                StateTranslation::Dictionary(ValuesDictionary::try_new(page, dict)?, None)
            },
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let values = BinaryIter::new(values, page.num_values());
                StateTranslation::Unit(values)
            },
            _ => return Err(not_implemented(page)),
        };

        Ok(State {
            is_optional,
            translation,
        })
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
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

        match &mut state.translation {
            StateTranslation::Unit(page) => {
                // @TODO: This can be optimized to not be a constantly polling
                for value in page.by_ref().take(n) {
                    values.push(value);
                }
            },
            StateTranslation::Dictionary(page, dict) => {
                let dict =
                    dict.get_or_insert_with(|| page.dict.values_iter().collect::<Vec<&[u8]>>());
                let translator = DictionaryTranslator(dict);

                // @TODO: This can be optimized to not be a constantly polling
                for value in page.values.by_ref().take(n) {
                    values.push(translator.translate(value)?);
                }
            },
        }

        if state.is_optional {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        let (values, validity) = decoded;

        values.extend_null_constant(n);
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub struct NestedIter<O: Offset, I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, (Binary<O>, MutableBitmap))>,
    dict: Option<BinaryDict>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<O: Offset, I: PagesIter> NestedIter<O, I> {
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            iter,
            data_type,
            init,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<O: Offset, I: PagesIter> Iterator for NestedIter<O, I> {
    type Item = PolarsResult<(NestedState, Box<dyn Array>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &BinaryDecoder::<O>::default(),
            );
            match maybe_state {
                MaybeNext::Some(Ok((nested, decoded))) => {
                    return Some(
                        finish(&self.data_type, decoded.0, decoded.1).map(|array| (nested, array)),
                    )
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue, // Using continue in a loop instead of calling next helps prevent stack overflow.
            }
        }
    }
}
