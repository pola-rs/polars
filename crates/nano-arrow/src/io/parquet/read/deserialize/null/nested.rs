use std::collections::VecDeque;

use parquet2::page::{DataPage, DictPage};

use crate::array::NullArray;
use crate::io::parquet::read::deserialize::utils::DecodedState;
use crate::{datatypes::DataType, error::Result};

use super::super::nested_utils::*;
use super::super::utils;
use super::super::Pages;

impl<'a> utils::PageState<'a> for usize {
    fn len(&self) -> usize {
        *self
    }
}

#[derive(Debug)]
struct NullDecoder {}

impl DecodedState for usize {
    fn len(&self) -> usize {
        *self
    }
}

impl<'a> NestedDecoder<'a> for NullDecoder {
    type State = usize;
    type Dictionary = usize;
    type DecodedState = usize;

    fn build_state(
        &self,
        _page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> Result<Self::State> {
        if let Some(n) = dict {
            return Ok(*n);
        }
        Ok(1)
    }

    /// Initializes a new state
    fn with_capacity(&self, _capacity: usize) -> Self::DecodedState {
        0
    }

    fn push_valid(&self, state: &mut Self::State, decoded: &mut Self::DecodedState) -> Result<()> {
        *decoded += *state;
        Ok(())
    }

    fn push_null(&self, decoded: &mut Self::DecodedState) {
        let length = decoded;
        *length += 1;
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        page.num_values
    }
}

/// An iterator adapter over [`Pages`] assumed to be encoded as null arrays
#[derive(Debug)]
pub struct NestedIter<I>
where
    I: Pages,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: DataType,
    items: VecDeque<(NestedState, usize)>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: NullDecoder,
}

impl<I> NestedIter<I>
where
    I: Pages,
{
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: DataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            items: VecDeque::new(),
            chunk_size,
            remaining: num_rows,
            decoder: NullDecoder {},
        }
    }
}

impl<I> Iterator for NestedIter<I>
where
    I: Pages,
{
    type Item = Result<(NestedState, NullArray)>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_state = next(
            &mut self.iter,
            &mut self.items,
            &mut None,
            &mut self.remaining,
            &self.init,
            self.chunk_size,
            &self.decoder,
        );
        match maybe_state {
            utils::MaybeNext::Some(Ok((nested, state))) => {
                Some(Ok((nested, NullArray::new(self.data_type.clone(), state))))
            }
            utils::MaybeNext::Some(Err(e)) => Some(Err(e)),
            utils::MaybeNext::None => None,
            utils::MaybeNext::More => self.next(),
        }
    }
}
