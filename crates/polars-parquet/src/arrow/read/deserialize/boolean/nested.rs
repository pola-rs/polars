use std::collections::VecDeque;

use arrow::array::BooleanArray;
use arrow::bitmap::utils::BitmapIter;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils::MaybeNext;
use super::super::{utils, PagesIter};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
struct State<'a> {
    is_optional: bool,
    iterator: BitmapIter<'a>,
}

impl<'a> utils::PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        self.iterator.len()
    }
}

struct BooleanDecoder;

impl<'a> NestedDecoder<'a> for BooleanDecoder {
    type State = State<'a>;
    type Dictionary = ();
    type DecodedState = (MutableBitmap, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        _: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        if is_filtered {
            return Err(utils::not_implemented(page));
        }

        let iterator = match page.encoding() {
            Encoding::Plain => {
                let values = split_buffer(page)?.values;
                BitmapIter::new(values, 0, values.len() * 8)
            },
            _ => return Err(utils::not_implemented(page)),
        };

        Ok(State {
            is_optional,
            iterator,
        })
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBitmap::with_capacity(capacity),
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

        state.iterator.collect_n_into(values, n);

        if state.is_optional {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        let (values, validity) = decoded;
        values.extend_constant(n, false);
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, _: &DictPage) -> Self::Dictionary {}
}

/// An iterator adapter over [`PagesIter`] assumed to be encoded as boolean arrays
#[derive(Debug)]
pub struct NestedIter<I: PagesIter> {
    iter: I,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, (MutableBitmap, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
}

impl<I: PagesIter> NestedIter<I> {
    pub fn new(iter: I, init: Vec<InitNested>, num_rows: usize, chunk_size: Option<usize>) -> Self {
        Self {
            iter,
            init,
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
        }
    }
}

fn finish(
    data_type: &ArrowDataType,
    values: MutableBitmap,
    validity: MutableBitmap,
) -> BooleanArray {
    BooleanArray::new(data_type.clone(), values.into(), validity.into())
}

impl<I: PagesIter> Iterator for NestedIter<I> {
    type Item = PolarsResult<(NestedState, BooleanArray)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut None,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &BooleanDecoder,
            );
            match maybe_state {
                MaybeNext::Some(Ok((nested, (values, validity)))) => {
                    return Some(Ok((
                        nested,
                        finish(&ArrowDataType::Boolean, values, validity),
                    )))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
