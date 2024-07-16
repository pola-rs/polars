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

impl<'pages, 'mmap: 'pages> NestedDecoder<'pages, 'mmap> for BooleanDecoder {
    type State = State<'pages>;
    type Dictionary = ();
    type DecodedState = (MutableBitmap, MutableBitmap);

    fn build_state(
        &self,
        page: &'pages DataPage<'mmap>,
        _: Option<&'pages Self::Dictionary>,
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

pub struct NestedBooleanIter;

impl NestedBooleanIter {
    pub fn new<'pages, 'mmap: 'pages, I: PagesIter<'mmap>>(
        iter: I,
        init: Vec<InitNested>,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> NestedDecodeIter<
        'pages,
        'mmap,
        BooleanArray,
        I,
        BooleanDecoder,
        fn(
            &ArrowDataType,
            NestedState,
            <BooleanDecoder as NestedDecoder<'pages, 'mmap>>::DecodedState,
        ) -> PolarsResult<(NestedState, BooleanArray)>,
    > {
        NestedDecodeIter::new(
            iter,
            ArrowDataType::Boolean,
            init,
            chunk_size,
            num_rows,
            BooleanDecoder,
            |_dt, nested, (values, validity)| {
                Ok((
                    nested,
                    BooleanArray::new(ArrowDataType::Boolean, values.into(), validity.into()),
                ))
            },
        )
    }
}
