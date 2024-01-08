use std::cell::Cell;
use arrow::array::{BinaryViewArray, MutableBinaryArray, MutableBinaryViewArray, ViewType};
use polars_error::PolarsResult;
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils;
use crate::read::deserialize::utils::{DecodedState, OptionalPageValidity, PageState};
use super::binary::BinaryIter;
use crate::read::{ParquetError, PrimitiveLogicalType};

struct BinViewDecoder {
    check_utf8: Cell<bool>
}

type BinViewDict = BinaryViewArray;

#[derive(Debug)]
enum BinViewDecoderState<'a> {
    Optional(OptionalPageValidity<'a>, BinaryIter<'a>),
}

impl<'a> PageState<'a> for BinViewDecoderState<'a> {
    fn len(&self) -> usize {
        todo!()
    }
}

impl DecodedState for BinaryViewArray {
    fn len(&self) -> usize {
        Self::len(self)
    }
}

impl<'a> utils::Decoder<'a> for BinViewDecoder {
    type State = BinViewDecoderState<'a>;
    type Dict = BinViewDict;
    type DecodedState = BinaryViewArray;

    fn build_state(&self, page: &'a DataPage, dict: Option<&'a Self::Dict>) -> PolarsResult<Self::State> {
        let is_optional =
            utils::page_is_optional(page);
        let is_filtered = utils::page_is_filtered(page);
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        self.check_utf8.set(is_string);
        todo!()

    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        todo!()
    }

    fn extend_from_state(&self, page: &mut Self::State, decoded: &mut Self::DecodedState, additional: usize) -> PolarsResult<()> {
        todo!()
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        todo!()
    }
}

fn deserialize_plain(values: &[u8], num_values: usize) -> BinViewDict {
    // TODO! Ideally we just allow a lifetime in `Dict`. That way we can keep a `Vec<&[u8]>`.
    // All the slices point to the same buffer, so it is pretty cache friendly
    let mut mutable = MutableBinaryViewArray::with_capacity(num_values);
    let iter = BinaryIter::new(values).take(num_values);
    for value in iter {
        mutable.push_value(value)
    }
    mutable.into()
}
