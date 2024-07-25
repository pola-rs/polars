use arrow::array::NullArray;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::utils;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

#[derive(Debug)]
pub(crate) struct NullDecoder;

impl utils::ExactSize for usize {
    fn len(&self) -> usize {
        *self
    }
}

pub(crate) struct Translation(usize);

impl<'a> utils::StateTranslation<'a, NullDecoder> for Translation {
    type PlainDecoder = ();

    fn new(
        _decoder: &NullDecoder,
        page: &'a DataPage,
        _dict: Option<&'a <NullDecoder as utils::Decoder>::Dict>,
        _page_validity: Option<&utils::PageValidity<'a>>,
        _filter: Option<&utils::filter::Filter<'a>>,
    ) -> PolarsResult<Self> {
        Ok(Self(page.num_values()))
    }

    fn len_when_not_nullable(&self) -> usize {
        self.0
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.0 -= n;

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        _decoder: &mut NullDecoder,
        decoded: &mut <NullDecoder as utils::Decoder>::DecodedState,
        _page_validity: &mut Option<utils::PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        *decoded += additional;
        self.0 -= additional;
        Ok(())
    }
}

impl utils::Decoder for NullDecoder {
    type Translation<'a> = Translation;
    type Dict = usize;
    type DecodedState = usize;

    /// Initializes a new state
    fn with_capacity(&self, _capacity: usize) -> Self::DecodedState {
        0
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        page.num_values
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        _page_validity: Option<&mut utils::PageValidity<'a>>,
        _limit: usize,
    ) -> ParquetResult<()> {
        unimplemented!()
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut crate::parquet::encoding::hybrid_rle::HybridRleDecoder<'a>,
        _page_validity: Option<&mut utils::PageValidity<'a>>,
        _dict: &Self::Dict,
        _limit: usize,
    ) -> ParquetResult<()> {
        unimplemented!()
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Box<dyn arrow::array::Array>> {
        Ok(Box::new(NullArray::new(data_type, decoded)))
    }

    fn finalize_dict_array<K: arrow::array::DictionaryKey>(
        &self,
        _data_type: ArrowDataType,
        _dict: Self::Dict,
        _decoded: (Vec<K>, Option<arrow::bitmap::Bitmap>),
    ) -> ParquetResult<arrow::array::DictionaryArray<K>> {
        unimplemented!()
    }
}

impl utils::NestedDecoder for NullDecoder {
    fn validity_extend(_: &mut Self::DecodedState, _value: bool, _n: usize) {}

    fn values_extend_nulls(values: &mut Self::DecodedState, n: usize) {
        *values += n;
    }
}
