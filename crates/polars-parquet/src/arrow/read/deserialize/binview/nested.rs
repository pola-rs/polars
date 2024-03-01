use std::collections::VecDeque;

use arrow::array::{ArrayRef, MutableBinaryViewArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::binary::decoders::{
    build_nested_state, deserialize_plain, BinaryDict, BinaryNestedState,
};
use crate::read::deserialize::binview::basic::finish;
use crate::read::deserialize::nested_utils::{next, NestedDecoder};
use crate::read::deserialize::utils::MaybeNext;
use crate::read::{InitNested, NestedState, PagesIter};

#[derive(Debug, Default)]
struct BinViewDecoder {}

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> NestedDecoder<'a> for BinViewDecoder {
    type State = BinaryNestedState<'a>;
    type Dictionary = BinaryDict;
    type DecodedState = DecodedStateTuple;

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        build_nested_state(page, dict)
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
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
            BinaryNestedState::Optional(page) => {
                let value = page.next().unwrap_or_default();
                values.push_value_ignore_validity(value);
                validity.push(true);
            },
            BinaryNestedState::Required(page) => {
                let value = page.next().unwrap_or_default();
                values.push_value_ignore_validity(value);
            },
            BinaryNestedState::RequiredDictionary(page) => {
                let dict_values = &page.dict;
                let item = page
                    .values
                    .next()
                    .map(|index| dict_values.value(index.unwrap() as usize))
                    .unwrap_or_default();
                values.push_value_ignore_validity(item);
            },
            BinaryNestedState::OptionalDictionary(page) => {
                let dict_values = &page.dict;
                let item = page
                    .values
                    .next()
                    .map(|index| dict_values.value(index.unwrap() as usize))
                    .unwrap_or_default();
                values.push_value_ignore_validity(item);
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
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub struct NestedIter<I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, DecodedStateTuple)>,
    dict: Option<BinaryDict>,
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

impl<I: PagesIter> Iterator for NestedIter<I> {
    type Item = PolarsResult<(NestedState, ArrayRef)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &BinViewDecoder::default(),
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
