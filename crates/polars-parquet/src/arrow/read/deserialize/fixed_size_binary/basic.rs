use std::collections::VecDeque;

use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::utils::{
    dict_indices_decoder, extend_from_decoder, get_selected_rows, next, not_implemented,
    DecodedState, Decoder, FilteredOptionalPageValidity, MaybeNext, OptionalPageValidity,
    PageState, Pushable,
};
use super::super::PagesIter;
use super::utils::FixedSizeBinary;
use crate::parquet::deserialize::SliceFilteredIter;
use crate::parquet::encoding::{hybrid_rle, Encoding};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

pub(super) type Dict = Vec<u8>;

#[derive(Debug)]
pub(super) struct Optional<'a> {
    pub(super) values: std::slice::ChunksExact<'a, u8>,
    pub(super) validity: OptionalPageValidity<'a>,
}

impl<'a> Optional<'a> {
    pub(super) fn try_new(page: &'a DataPage, size: usize) -> PolarsResult<Self> {
        let (_, _, values) = split_buffer(page)?;

        let values = values.chunks_exact(size);

        Ok(Self {
            values,
            validity: OptionalPageValidity::try_new(page)?,
        })
    }
}

#[derive(Debug)]
pub(super) struct Required<'a> {
    pub values: std::slice::ChunksExact<'a, u8>,
}

impl<'a> Required<'a> {
    pub(super) fn new(page: &'a DataPage, size: usize) -> Self {
        let values = page.buffer();
        assert_eq!(values.len() % size, 0);
        let values = values.chunks_exact(size);
        Self { values }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct FilteredRequired<'a> {
    pub values: SliceFilteredIter<std::slice::ChunksExact<'a, u8>>,
}

impl<'a> FilteredRequired<'a> {
    fn new(page: &'a DataPage, size: usize) -> Self {
        let values = page.buffer();
        assert_eq!(values.len() % size, 0);
        let values = values.chunks_exact(size);

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Self { values }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct RequiredDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a Dict,
}

impl<'a> RequiredDictionary<'a> {
    pub(super) fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct OptionalDictionary<'a> {
    pub(super) values: hybrid_rle::HybridRleDecoder<'a>,
    pub(super) validity: OptionalPageValidity<'a>,
    pub(super) dict: &'a Dict,
}

impl<'a> OptionalDictionary<'a> {
    pub(super) fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = dict_indices_decoder(page)?;

        Ok(Self {
            values,
            validity: OptionalPageValidity::try_new(page)?,
            dict,
        })
    }
}

#[derive(Debug)]
enum State<'a> {
    Optional(Optional<'a>),
    Required(Required<'a>),
    RequiredDictionary(RequiredDictionary<'a>),
    OptionalDictionary(OptionalDictionary<'a>),
    FilteredRequired(FilteredRequired<'a>),
    FilteredOptional(
        FilteredOptionalPageValidity<'a>,
        std::slice::ChunksExact<'a, u8>,
    ),
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match self {
            State::Optional(state) => state.validity.len(),
            State::Required(state) => state.len(),
            State::RequiredDictionary(state) => state.len(),
            State::OptionalDictionary(state) => state.validity.len(),
            State::FilteredRequired(state) => state.len(),
            State::FilteredOptional(state, _) => state.len(),
        }
    }
}

struct BinaryDecoder {
    size: usize,
}

impl DecodedState for (FixedSizeBinary, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> Decoder<'a> for BinaryDecoder {
    type State = State<'a>;
    type Dict = Dict;
    type DecodedState = (FixedSizeBinary, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
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
            (Encoding::Plain, None, false, true) => Ok(State::FilteredRequired(
                FilteredRequired::new(page, self.size),
            )),
            (Encoding::Plain, _, true, true) => {
                let (_, _, values) = split_buffer(page)?;

                Ok(State::FilteredOptional(
                    FilteredOptionalPageValidity::try_new(page)?,
                    values.chunks_exact(self.size),
                ))
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

    fn extend_from_state(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,

        remaining: usize,
    ) {
        let (values, validity) = decoded;
        match state {
            State::Optional(page) => extend_from_decoder(
                validity,
                &mut page.validity,
                Some(remaining),
                values,
                &mut page.values,
            ),
            State::Required(page) => {
                for x in page.values.by_ref().take(remaining) {
                    values.push(x)
                }
            },
            State::FilteredRequired(page) => {
                for x in page.values.by_ref().take(remaining) {
                    values.push(x)
                }
            },
            State::OptionalDictionary(page) => extend_from_decoder(
                validity,
                &mut page.validity,
                Some(remaining),
                values,
                page.values.by_ref().map(|index| {
                    let index = index.unwrap() as usize;
                    &page.dict[index * self.size..(index + 1) * self.size]
                }),
            ),
            State::RequiredDictionary(page) => {
                for x in page
                    .values
                    .by_ref()
                    .map(|index| {
                        let index = index.unwrap() as usize;
                        &page.dict[index * self.size..(index + 1) * self.size]
                    })
                    .take(remaining)
                {
                    values.push(x)
                }
            },
            State::FilteredOptional(page_validity, page_values) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(remaining),
                    values,
                    page_values.by_ref(),
                );
            },
        }
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        page.buffer.clone()
    }
}

pub fn finish(
    data_type: &ArrowDataType,
    values: FixedSizeBinary,
    validity: MutableBitmap,
) -> FixedSizeBinaryArray {
    FixedSizeBinaryArray::new(data_type.clone(), values.values.into(), validity.into())
}

pub struct Iter<I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    size: usize,
    items: VecDeque<(FixedSizeBinary, MutableBitmap)>,
    dict: Option<Dict>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<I: PagesIter> Iter<I> {
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        let size = FixedSizeBinaryArray::get_size(&data_type);
        Self {
            iter,
            data_type,
            size,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<I: PagesIter> Iterator for Iter<I> {
    type Item = PolarsResult<FixedSizeBinaryArray>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                self.chunk_size,
                &BinaryDecoder { size: self.size },
            );
            match maybe_state {
                MaybeNext::Some(Ok((values, validity))) => {
                    return Some(Ok(finish(&self.data_type, values, validity)))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
