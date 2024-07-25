use arrow::array::NullArray;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

impl<'a> utils::PageState<'a> for usize {
    fn len(&self) -> usize {
        *self
    }
}

#[derive(Debug)]
pub(crate) struct NullDecoder;

impl utils::ExactSize for usize {
    fn len(&self) -> usize {
        *self
    }
}

impl NestedDecoder for NullDecoder {
    type State<'a> = usize;
    type Dict = usize;
    type DecodedState = usize;

    fn build_state<'a>(
        &self,
        _page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State<'a>> {
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
        state: &mut Self::State<'_>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        *decoded += *state * n;
        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        *decoded += n;
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        page.num_values
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Box<dyn arrow::array::Array>> {
        Ok(Box::new(NullArray::new(data_type, decoded)))
    }
}
