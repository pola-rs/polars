use std::cell::Cell;
use std::collections::VecDeque;

use arrow::array::{Array, ArrayRef, BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_error::PolarsResult;

use super::super::binary::decoders::*;
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils;
use crate::read::deserialize::utils::{extend_from_decoder, next, DecodedState, MaybeNext};
use crate::read::{PagesIter, PrimitiveLogicalType};

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

#[derive(Default)]
struct BinViewDecoder {
    check_utf8: Cell<bool>,
}

impl DecodedState for DecodedStateTuple {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> utils::Decoder<'a> for BinViewDecoder {
    type State = BinaryState<'a>;
    type Dict = BinaryDict;
    type DecodedState = DecodedStateTuple;

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
            MutableBinaryViewArray::with_capacity(capacity),
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
                    values.push_value_ignore_validity(x)
                }
            },
            BinaryState::Delta(page) => {
                for value in page {
                    values.push_value_ignore_validity(value)
                }
            },
            BinaryState::OptionalDelta(page_validity, page_values) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values,
                );
            },
            BinaryState::FilteredRequired(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push_value_ignore_validity(x)
                }
            },
            BinaryState::FilteredDelta(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push_value_ignore_validity(x)
                }
            },
            BinaryState::OptionalDictionary(page_validity, page_values) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page_values.dict;
                utils::extend_from_decoder(
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
                    values.push_value_ignore_validity(x)
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
                // TODO! directly set the dict as buffers and only insert the proper views.
                // This will save a lot of memory.
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;
                for x in page
                    .values
                    .by_ref()
                    .map(|index| page_dict.value(index.unwrap() as usize))
                    .take(additional)
                {
                    values.push_value_ignore_validity(x)
                }
            },
            BinaryState::FilteredOptionalDictionary(page_validity, page_values) => {
                // Already done on the dict.
                validate_utf8 = false;
                // TODO! directly set the dict as buffers and only insert the proper views.
                // This will save a lot of memory.
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
            BinaryState::OptionalDeltaByteArray(page_validity, page_values) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            ),
            BinaryState::DeltaByteArray(page_values) => {
                for x in page_values.take(additional) {
                    values.push_value_ignore_validity(x)
                }
            },
        }

        if validate_utf8 {
            values.validate_utf8()
        } else {
            Ok(())
        }
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub struct BinaryViewArrayIter<I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    items: VecDeque<DecodedStateTuple>,
    dict: Option<BinaryDict>,
    chunk_size: Option<usize>,
    remaining: usize,
}
impl<I: PagesIter> BinaryViewArrayIter<I> {
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

impl<I: PagesIter> Iterator for BinaryViewArrayIter<I> {
    type Item = PolarsResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        let decoder = BinViewDecoder::default();
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

pub(super) fn finish(
    data_type: &ArrowDataType,
    values: MutableBinaryViewArray<[u8]>,
    validity: MutableBitmap,
) -> PolarsResult<Box<dyn Array>> {
    let mut array: BinaryViewArray = values.into();
    let validity: Bitmap = validity.into();

    if validity.unset_bits() != validity.len() {
        array = array.with_validity(Some(validity))
    }

    match data_type.to_physical_type() {
        PhysicalType::BinaryView => unsafe {
            Ok(BinaryViewArray::new_unchecked(
                data_type.clone(),
                array.views().clone(),
                array.data_buffers().clone(),
                array.validity().cloned(),
                array.total_bytes_len(),
                array.total_buffer_len(),
            )
            .boxed())
        },
        PhysicalType::Utf8View => {
            // SAFETY: we already checked utf8
            unsafe {
                Ok(Utf8ViewArray::new_unchecked(
                    data_type.clone(),
                    array.views().clone(),
                    array.data_buffers().clone(),
                    array.validity().cloned(),
                    array.total_bytes_len(),
                    array.total_buffer_len(),
                )
                .boxed())
            }
        },
        _ => unreachable!(),
    }
}
