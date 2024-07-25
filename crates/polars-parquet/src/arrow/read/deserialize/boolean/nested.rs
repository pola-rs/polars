use arrow::array::BooleanArray;
use arrow::bitmap::utils::BitmapIter;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub(crate) struct State<'a> {
    is_optional: bool,
    iterator: BitmapIter<'a>,
}

impl<'a> utils::PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        self.iterator.len()
    }
}

impl NestedDecoder for super::BooleanDecoder {
    type State<'a> = State<'a>;
    type Dict = ();
    type DecodedState = (MutableBitmap, MutableBitmap);

    fn build_state<'a>(
        &self,
        page: &'a DataPage,
        _: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State<'a>> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        if is_filtered {
            return Err(utils::not_implemented(page));
        }

        let iterator = match page.encoding() {
            Encoding::Plain => {
                let values = split_buffer(page)?.values;
                BitmapIter::new(values, 0, values.len() * 8)
            },
            _ => return Err(utils::not_implemented(page)),
        };

        Ok(State {
            is_optional,
            iterator,
        })
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBitmap::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_n_valid(
        &self,
        state: &mut Self::State<'_>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        state.iterator.collect_n_into(values, n);

        if state.is_optional {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        let (values, validity) = decoded;
        values.extend_constant(n, false);
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, _: DictPage) -> Self::Dict {}

    fn finalize(
        &self,
        data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn arrow::array::Array>> {
        let validity = if validity.is_empty() {
            None
        } else {
            Some(validity.freeze())
        };

        Ok(Box::new(BooleanArray::new(
            data_type,
            values.into(),
            validity,
        )))
    }
}
