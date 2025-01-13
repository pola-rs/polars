//! This implements the [`Decoder`][utils::Decoder] trait for the `UNKNOWN` or `Null` nested type.
//! The implementation mostly stubs all the function and just keeps track of the length in the
//! `DecodedState`.

use arrow::array::NullArray;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use super::utils;
use super::utils::filter::Filter;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

pub(crate) struct NullDecoder;
pub(crate) struct NullTranslation {
    num_rows: usize,
}

#[derive(Debug)]
pub(crate) struct NullArrayLength {
    length: usize,
}

impl utils::ExactSize for NullArrayLength {
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a> utils::StateTranslation<'a, NullDecoder> for NullTranslation {
    type PlainDecoder = ();

    fn new(
        _decoder: &NullDecoder,
        page: &'a DataPage,
        _dict: Option<&'a <NullDecoder as utils::Decoder>::Dict>,
        _page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        Ok(NullTranslation {
            num_rows: page.num_values(),
        })
    }
}

impl utils::Decoder for NullDecoder {
    type Translation<'a> = NullTranslation;
    type Dict = NullArray;
    type DecodedState = NullArrayLength;
    type Output = NullArray;

    /// Initializes a new state
    fn with_capacity(&self, _: usize) -> Self::DecodedState {
        NullArrayLength { length: 0 }
    }

    fn deserialize_dict(&mut self, _: DictPage) -> ParquetResult<Self::Dict> {
        Ok(NullArray::new_empty(ArrowDataType::Null))
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        Ok(NullArray::new(dtype, decoded.length))
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        decoded.length += Filter::opt_num_rows(&filter, state.translation.num_rows);
        Ok(())
    }
}
