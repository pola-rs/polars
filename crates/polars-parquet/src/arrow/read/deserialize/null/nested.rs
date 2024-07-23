use std::collections::VecDeque;

use arrow::array::NullArray;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils;
use super::super::utils::MaybeNext;
use crate::arrow::read::deserialize::utils::DecodedState;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};
use crate::parquet::read::BasicDecompressor;
use crate::read::CompressedPagesIter;

impl<'a> utils::PageState<'a> for usize {
    fn len(&self) -> usize {
        *self
    }
}

#[derive(Debug)]
struct NullDecoder;

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
    ) -> PolarsResult<Self::State> {
        if let Some(n) = dict {
            return Ok(*n);
        }
        Ok(1)
    }

    /// Initializes a new state
    fn with_capacity(&self, _capacity: usize) -> Self::DecodedState {
        0
    }

    fn push_n_valid(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        *decoded += *state * n;
        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        *decoded += n;
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        page.num_values
    }
}

/// An iterator adapter over [`PagesIter`] assumed to be encoded as null arrays
pub struct NestedIter<I: CompressedPagesIter> {
    iter: BasicDecompressor<I>,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    items: VecDeque<(NestedState, usize)>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: NullDecoder,
}

impl<I: CompressedPagesIter> NestedIter<I> {
    pub fn new(
        iter: BasicDecompressor<I>,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
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
            decoder: NullDecoder,
        }
    }
}

impl<I: CompressedPagesIter> Iterator for NestedIter<I> {
    type Item = PolarsResult<(NestedState, NullArray)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
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
                MaybeNext::Some(Ok((nested, state))) => {
                    return Some(Ok((nested, NullArray::new(self.data_type.clone(), state))))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
