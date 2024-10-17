//! This implements the [`Decoder`][utils::Decoder] trait for the `UNKNOWN` or `Null` nested type.
//! The implementation mostly stubs all the function and just keeps track of the length in the
//! `DecodedState`.

use arrow::array::{Array, NullArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use super::utils;
use super::utils::filter::Filter;
use crate::parquet::encoding::hybrid_rle;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

pub(crate) struct NullDecoder;
#[derive(Debug)]
pub(crate) struct NullArrayLength {
    length: usize,
}

impl utils::ExactSize for NullArrayLength {
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a> utils::StateTranslation<'a, NullDecoder> for () {
    type PlainDecoder = ();

    fn new(
        _decoder: &NullDecoder,
        _page: &'a DataPage,
        _dict: Option<&'a <NullDecoder as utils::Decoder>::Dict>,
        _page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        Ok(())
    }

    fn len_when_not_nullable(&self) -> usize {
        usize::MAX
    }

    fn skip_in_place(&mut self, _: usize) -> ParquetResult<()> {
        Ok(())
    }

    fn extend_from_state(
        &mut self,
        _decoder: &mut NullDecoder,
        decoded: &mut <NullDecoder as utils::Decoder>::DecodedState,
        _is_optional: bool,
        _page_validity: &mut Option<Bitmap>,
        _: Option<&'a <NullDecoder as utils::Decoder>::Dict>,
        additional: usize,
    ) -> ParquetResult<()> {
        decoded.length += additional;
        Ok(())
    }
}

impl utils::Decoder for NullDecoder {
    type Translation<'a> = ();
    type Dict = ();
    type DecodedState = NullArrayLength;
    type Output = NullArray;

    /// Initializes a new state
    fn with_capacity(&self, _: usize) -> Self::DecodedState {
        NullArrayLength { length: 0 }
    }

    fn deserialize_dict(&mut self, _: DictPage) -> ParquetResult<Self::Dict> {
        Ok(())
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        _is_optional: bool,
        _page_validity: Option<&mut Bitmap>,
        _limit: usize,
    ) -> ParquetResult<()> {
        unimplemented!()
    }

    fn decode_dictionary_encoded(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut hybrid_rle::HybridRleDecoder<'_>,
        _is_optional: bool,
        _page_validity: Option<&mut Bitmap>,
        _dict: &Self::Dict,
        _limit: usize,
    ) -> ParquetResult<()> {
        unimplemented!()
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        Ok(NullArray::new(dtype, decoded.length))
    }
}

use super::BasicDecompressor;

/// Converts [`PagesIter`] to an [`ArrayIter`]
pub fn iter_to_arrays(
    mut iter: BasicDecompressor,
    dtype: ArrowDataType,
    mut filter: Option<Filter>,
) -> ParquetResult<Box<dyn Array>> {
    _ = iter.read_dict_page()?;

    let num_rows = Filter::opt_num_rows(&filter, iter.total_num_values());

    let mut len = 0usize;

    while len < num_rows {
        let Some(page) = iter.next() else {
            break;
        };
        let page = page?;

        let state_filter;
        (state_filter, filter) = Filter::opt_split_at(&filter, page.num_values());

        // Skip the whole page if we don't need any rows from it
        if state_filter.as_ref().is_some_and(|f| f.num_rows() == 0) {
            continue;
        }

        let num_rows = match state_filter {
            None => page.num_values(),
            Some(filter) => filter.num_rows(),
        };

        len = (len + num_rows).min(num_rows);
    }

    Ok(Box::new(NullArray::new(dtype, len)))
}
