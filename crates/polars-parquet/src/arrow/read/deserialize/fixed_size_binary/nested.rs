use std::collections::VecDeque;

use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::utils::{not_implemented, MaybeNext, PageState};
use super::utils::FixedSizeBinary;
use crate::arrow::read::deserialize::fixed_size_binary::basic::{
    finish, Dict, Optional, OptionalDictionary, Required, RequiredDictionary,
};
use crate::arrow::read::deserialize::nested_utils::{next, NestedDecoder};
use crate::arrow::read::deserialize::utils::Pushable;
use crate::arrow::read::{InitNested, NestedState, PagesIter};
use crate::parquet::encoding::Encoding;
use crate::parquet::page::{DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
enum State<'a> {
    Optional(Optional<'a>),
    Required(Required<'a>),
    RequiredDictionary(RequiredDictionary<'a>),
    OptionalDictionary(OptionalDictionary<'a>),
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match self {
            State::Optional(state) => state.validity.len(),
            State::Required(state) => state.len(),
            State::RequiredDictionary(state) => state.len(),
            State::OptionalDictionary(state) => state.validity.len(),
        }
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder {
    size: usize,
}

impl<'a> NestedDecoder<'a> for BinaryDecoder {
    type State = State<'a>;
    type Dictionary = Dict;
    type DecodedState = (FixedSizeBinary, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        match (page.encoding(), dict, is_optional, is_filtered) {
            (Encoding::Plain, _, true, false) => {
                Ok(State::Optional(Optional::try_new(page, self.size)?))
            },
            (Encoding::Plain, _, false, false) => {
                Ok(State::Required(Required::new(page, self.size)))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
                RequiredDictionary::try_new(page, dict).map(State::RequiredDictionary)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
                OptionalDictionary::try_new(page, dict).map(State::OptionalDictionary)
            },
            _ => Err(not_implemented(page)),
        }
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            FixedSizeBinary::with_capacity(capacity, self.size),
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
                let value = page.values.by_ref().next().unwrap_or_default();
                values.push(value);
                validity.push(true);
            },
            State::Required(page) => {
                let value = page.values.by_ref().next().unwrap_or_default();
                values.push(value);
            },
            State::RequiredDictionary(page) => {
                let item = page
                    .values
                    .by_ref()
                    .next()
                    .map(|index| {
                        let index = index.unwrap() as usize;
                        &page.dict[index * self.size..(index + 1) * self.size]
                    })
                    .unwrap_or_default();
                values.push(item);
            },
            State::OptionalDictionary(page) => {
                let item = page
                    .values
                    .by_ref()
                    .next()
                    .map(|index| {
                        let index = index.unwrap() as usize;
                        &page.dict[index * self.size..(index + 1) * self.size]
                    })
                    .unwrap_or_default();
                values.push(item);
                validity.push(true);
            },
        }
        Ok(())
    }

    fn push_null(&self, decoded: &mut Self::DecodedState) {
        let (values, validity) = decoded;
        values.push_null();
        validity.push(false);
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        page.buffer.clone()
    }
}

pub struct NestedIter<I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    size: usize,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, (FixedSizeBinary, MutableBitmap))>,
    dict: Option<Dict>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<I: PagesIter> NestedIter<I> {
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        let size = FixedSizeBinaryArray::get_size(&data_type);
        Self {
            iter,
            data_type,
            size,
            init,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<I: PagesIter> Iterator for NestedIter<I> {
    type Item = PolarsResult<(NestedState, FixedSizeBinaryArray)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &BinaryDecoder { size: self.size },
            );
            match maybe_state {
                MaybeNext::Some(Ok((nested, decoded))) => {
                    return Some(Ok((nested, finish(&self.data_type, decoded.0, decoded.1))))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
