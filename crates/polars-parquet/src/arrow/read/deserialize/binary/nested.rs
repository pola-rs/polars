use std::collections::VecDeque;

use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils;
use super::super::utils::MaybeNext;
use super::basic::{finish};
use super::utils::*;
use crate::arrow::read::PagesIter;
use crate::parquet::encoding::Encoding;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;
use super::decoders::*;

#[derive(Debug)]
enum State<'a> {
    Optional(BinaryIter<'a>),
    Required(BinaryIter<'a>),
    RequiredDictionary(ValuesDictionary<'a>),
    OptionalDictionary(ValuesDictionary<'a>),
}

impl<'a> utils::PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match self {
            State::Optional(validity) => validity.size_hint().0,
            State::Required(state) => state.size_hint().0,
            State::RequiredDictionary(required) => required.len(),
            State::OptionalDictionary(optional) => optional.len(),
        }
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder<O: Offset> {
    phantom_o: std::marker::PhantomData<O>,
}

impl<'a, O: Offset> NestedDecoder<'a> for BinaryDecoder<O> {
    type State = State<'a>;
    type Dictionary = Dict;
    type DecodedState = (Binary<O>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        match (page.encoding(), dict, is_optional, is_filtered) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
                ValuesDictionary::try_new(page, dict).map(State::RequiredDictionary)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
                ValuesDictionary::try_new(page, dict).map(State::OptionalDictionary)
            },
            (Encoding::Plain, _, true, false) => {
                let (_, _, values) = split_buffer(page)?;

                let values = BinaryIter::new(values);

                Ok(State::Optional(values))
            },
            (Encoding::Plain, _, false, false) => {
                let (_, _, values) = split_buffer(page)?;

                let values = BinaryIter::new(values);

                Ok(State::Required(values))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_valid(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
    ) -> PolarsResult<()> {
        let (values, validity) = decoded;
        match state {
            State::Optional(page) => {
                let value = page.next().unwrap_or_default();
                values.push(value);
                validity.push(true);
            },
            State::Required(page) => {
                let value = page.next().unwrap_or_default();
                values.push(value);
            },
            State::RequiredDictionary(page) => {
                let dict_values = &page.dict;
                let item = page
                    .values
                    .next()
                    .map(|index| dict_values.value(index.unwrap() as usize))
                    .unwrap_or_default();
                values.push(item);
            },
            State::OptionalDictionary(page) => {
                let dict_values = &page.dict;
                let item = page
                    .values
                    .next()
                    .map(|index| dict_values.value(index.unwrap() as usize))
                    .unwrap_or_default();
                values.push(item);
                validity.push(true);
            },
        }
        Ok(())
    }

    fn push_null(&self, decoded: &mut Self::DecodedState) {
        let (values, validity) = decoded;
        values.push(&[]);
        validity.push(false);
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
    dict: Option<Dict>,
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
