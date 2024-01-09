use std::cell::Cell;
use std::collections::VecDeque;
use std::default::Default;

use arrow::array::specification::try_check_utf8;
use arrow::array::{Array, BinaryArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::utils::{
    extend_from_decoder, next, DecodedState, FilteredOptionalPageValidity,
    MaybeNext, OptionalPageValidity,
};
use super::super::{utils, PagesIter};
use super::utils::*;
use crate::parquet::encoding::{Encoding};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::{PrimitiveLogicalType};

use super::decoders::*;


impl<O: Offset> DecodedState for (Binary<O>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder<O: Offset> {
    phantom_o: std::marker::PhantomData<O>,
    check_utf8: Cell<bool>,
}

impl<'a, O: Offset> utils::Decoder<'a> for BinaryDecoder<O> {
    type State = BinaryState<'a>;
    type Dict = BinaryDict;
    type DecodedState = (Binary<O>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        self.check_utf8.set(is_string);
        build_binary_state(page, dict, is_string)
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn extend_from_state(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        additional: usize,
    ) -> PolarsResult<()> {
        let (values, validity) = decoded;
        let mut validate_utf8 = self.check_utf8.take();
        let len_before = values.offsets.len();
        match state {
            BinaryState::Optional(page_validity, page_values) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            ),
            BinaryState::Required(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            BinaryState::Delta(page) => {
                values.extend_lengths(page.lengths.by_ref().take(additional), &mut page.values);
            },
            BinaryState::OptionalDelta(page_validity, page_values) => {
                let Binary {
                    offsets,
                    values: values_,
                } = values;

                let last_offset = *offsets.last();
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    offsets,
                    page_values.lengths.by_ref(),
                );

                let length = *offsets.last() - last_offset;

                let (consumed, remaining) = page_values.values.split_at(length.to_usize());
                page_values.values = remaining;
                values_.extend_from_slice(consumed);
            },
            BinaryState::FilteredRequired(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            BinaryState::FilteredDelta(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            BinaryState::OptionalDictionary(page_validity, page_values) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page_values.dict;
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values
                        .values
                        .by_ref()
                        .map(|index| page_dict.value(index.unwrap() as usize)),
                )
            },
            BinaryState::RequiredDictionary(page) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;

                for x in page
                    .values
                    .by_ref()
                    .map(|index| page_dict.value(index.unwrap() as usize))
                    .take(additional)
                {
                    values.push(x)
                }
            },
            BinaryState::FilteredOptional(page_validity, page_values) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values.by_ref(),
                );
            },
            BinaryState::FilteredOptionalDelta(page_validity, page_values) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values.by_ref(),
                );
            },
            BinaryState::FilteredRequiredDictionary(page) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;
                for x in page
                    .values
                    .by_ref()
                    .map(|index| page_dict.value(index.unwrap() as usize))
                    .take(additional)
                {
                    values.push(x)
                }
            },
            BinaryState::FilteredOptionalDictionary(page_validity, page_values) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page_values.dict;
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values
                        .values
                        .by_ref()
                        .map(|index| page_dict.value(index.unwrap() as usize)),
                )
            },
            BinaryState::OptionalDeltaByteArray(page_validity, page_values) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values,
                )
            },
            BinaryState::DeltaByteArray(page_values) => {
                for x in page_values.take(additional) {
                    values.push(x)
                }
            },
        }

        if validate_utf8 {
            let offsets = &values.offsets.as_slice()[len_before..];
            try_check_utf8(offsets, &values.values)
        } else {
            Ok(())
        }
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub(super) fn finish<O: Offset>(
    data_type: &ArrowDataType,
    mut values: Binary<O>,
    mut validity: MutableBitmap,
) -> PolarsResult<Box<dyn Array>> {
    values.offsets.shrink_to_fit();
    values.values.shrink_to_fit();
    validity.shrink_to_fit();

    match data_type.to_physical_type() {
        PhysicalType::Binary | PhysicalType::LargeBinary => BinaryArray::<O>::try_new(
            data_type.clone(),
            values.offsets.into(),
            values.values.into(),
            validity.into(),
        )
        .map(|x| x.boxed()),
        PhysicalType::Utf8 | PhysicalType::LargeUtf8 => unsafe {
            Ok(Utf8Array::<O>::new_unchecked(
                data_type.clone(),
                values.offsets.into(),
                values.values.into(),
                validity.into(),
            )
            .boxed())
        },
        _ => unreachable!(),
    }
}

pub struct Iter<O: Offset, I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    items: VecDeque<(Binary<O>, MutableBitmap)>,
    dict: Option<BinaryDict>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<O: Offset, I: PagesIter> Iter<O, I> {
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        chunk_size: Option<usize>,
        num_rows: usize,
    ) -> Self {
        Self {
            iter,
            data_type,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<O: Offset, I: PagesIter> Iterator for Iter<O, I> {
    type Item = PolarsResult<Box<dyn Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        let decoder = BinaryDecoder::<O>::default();
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                self.chunk_size,
                &decoder,
            );
            match maybe_state {
                MaybeNext::Some(Ok((values, validity))) => {
                    return Some(finish(&self.data_type, values, validity))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}

