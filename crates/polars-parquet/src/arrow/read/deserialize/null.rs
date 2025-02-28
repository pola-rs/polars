//! This implements the [`Decoder`][utils::Decoder] trait for the `UNKNOWN` or `Null` nested type.
//! The implementation mostly stubs all the function and just keeps track of the length in the
//! `DecodedState`.

use arrow::array::NullArray;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;

use super::utils::filter::Filter;
use super::utils::{self};
use super::PredicateFilter;
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

impl utils::Decoded for NullArrayLength {
    fn len(&self) -> usize {
        self.length
    }
    fn extend_nulls(&mut self, n: usize) {
        self.length += n;
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
    fn num_rows(&self) -> usize {
        self.num_rows
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

    fn has_predicate_specialization(
        &self,
        _state: &utils::State<'_, Self>,
        _predicate: &PredicateFilter,
    ) -> ParquetResult<bool> {
        // @TODO: This can be enabled for the fast paths
        Ok(false)
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn arrow::array::Array,
        _is_optional: bool,
    ) -> ParquetResult<()> {
        let additional = additional.as_any().downcast_ref::<NullArray>().unwrap();
        decoded.length += additional.len();

        Ok(())
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
        _pred_true_mask: &mut BitmapBuilder,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        if matches!(filter, Some(Filter::Predicate(_))) {
            todo!()
        }

        let num_rows = match filter {
            Some(f) => f.num_rows(0),
            None => state.translation.num_rows,
        };
        decoded.length += num_rows;

        Ok(())
    }
}
