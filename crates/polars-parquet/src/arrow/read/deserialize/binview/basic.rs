
use std::cell::Cell;
use arrow::array::{BinaryViewArray, MutableBinaryArray, MutableBinaryViewArray, ViewType};
use arrow::bitmap::MutableBitmap;
use arrow::pushable::Pushable;
use polars_error::PolarsResult;
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils;
use crate::read::deserialize::utils::{DecodedState, extend_from_decoder, OptionalPageValidity, PageState};
use super::super::binary::{
    decoders::*,
    BinaryIter
};
use crate::read::{ParquetError, PrimitiveLogicalType};

struct BinViewDecoder {
    check_utf8: Cell<bool>
}

impl DecodedState for (MutableBinaryViewArray<[u8]>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> utils::Decoder<'a> for BinViewDecoder {
    type State = BinaryState<'a>;
    type Dict = BinaryDict;
    type DecodedState = (MutableBinaryViewArray<[u8]>, MutableBitmap);

    fn build_state(&self, page: &'a DataPage, dict: Option<&'a Self::Dict>) -> PolarsResult<Self::State> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        self.check_utf8.set(is_string);
        build_binary_state(page, dict, is_string)
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (MutableBinaryViewArray::with_capacity(capacity), MutableBitmap::with_capacity(capacity))
    }

    fn extend_from_state(&self, state: &mut Self::State, decoded: &mut Self::DecodedState, additional: usize) -> PolarsResult<()> {
        let (values, validity) = decoded;
        let mut validate_utf8 = self.check_utf8.take();

        match state {
            BinaryState::Optional(page_validity, page_values)  => {
                extend_from_decoder(validity, page_validity, Some(additional), values, page_values)
            }
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
                    page_values
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
