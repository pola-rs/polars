use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use arrow::pushable::Pushable;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::decoders::*;
use super::utils::*;
use super::BinaryDecoder;
use crate::parquet::encoding::hybrid_rle::{DictionaryTranslator, Translator};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::{
    not_implemented, page_is_filtered, page_is_optional, PageState,
};

#[derive(Debug)]
pub struct State<'a> {
    is_optional: bool,
    translation: StateTranslation<'a>,
}

#[derive(Debug)]
pub enum StateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(ValuesDictionary<'a>, Option<Vec<&'a [u8]>>),
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Plain(iter) => iter.size_hint().0,
            StateTranslation::Dictionary(values, _) => values.len(),
        }
    }
}

impl<O: Offset> NestedDecoder for BinaryDecoder<O> {
    type State<'a> = State<'a>;
    type Dict = BinaryDict;
    type DecodedState = (Binary<O>, MutableBitmap);

    fn build_state<'a>(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State<'a>> {
        let is_optional = page_is_optional(page);
        let is_filtered = page_is_filtered(page);

        if is_filtered {
            return Err(not_implemented(page));
        }

        let translation = match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                StateTranslation::Dictionary(ValuesDictionary::try_new(page, dict)?, None)
            },
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let values = BinaryIter::new(values, page.num_values());
                StateTranslation::Plain(values)
            },
            _ => return Err(not_implemented(page)),
        };

        Ok(State {
            is_optional,
            translation,
        })
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_n_valid<'a>(
        &self,
        state: &mut Self::State<'a>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        match &mut state.translation {
            StateTranslation::Plain(page) => {
                // @TODO: This can be optimized to not be a constantly polling
                for value in page.by_ref().take(n) {
                    values.push(value);
                }
            },
            StateTranslation::Dictionary(page, dict) => {
                let dict =
                    dict.get_or_insert_with(|| page.dict.values_iter().collect::<Vec<&[u8]>>());
                let translator = DictionaryTranslator(dict);

                // @TODO: This can be optimized to not be a constantly polling
                for value in page.values.by_ref().take(n) {
                    values.push(translator.translate(value)?);
                }
            },
        }

        if state.is_optional {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        let (values, validity) = decoded;

        values.extend_null_constant(n);
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        super::finalize(data_type, values, validity)
    }
}
