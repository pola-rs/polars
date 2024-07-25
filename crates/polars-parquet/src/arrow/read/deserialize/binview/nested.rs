use arrow::array::{MutableBinaryViewArray, View};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::binary::decoders::{deserialize_plain, BinaryDict, ValuesDictionary};
use crate::read::deserialize::binary::utils::BinaryIter;
use crate::read::deserialize::nested_utils::NestedDecoder;
use crate::read::deserialize::utils::{
    self, binary_views_dict, not_implemented, page_is_filtered, page_is_optional, PageState,
};

#[derive(Debug)]
pub(crate) struct State<'a> {
    pub is_optional: bool,
    pub translation: StateTranslation<'a>,
}

#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(ValuesDictionary<'a>, Option<Vec<View>>),
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Plain(iter) => iter.size_hint().0,
            StateTranslation::Dictionary(values, _) => values.len(),
        }
    }
}

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl NestedDecoder for super::BinViewDecoder {
    type State<'a> = State<'a>;
    type Dict = BinaryDict;
    type DecodedState = DecodedStateTuple;

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
            MutableBinaryViewArray::with_capacity(capacity),
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
        match &mut state.translation {
            StateTranslation::Plain(page) => {
                // @TODO: This should probably be optimized to a better loop
                for value in page.by_ref().take(n) {
                    values.push_value_ignore_validity(value);
                }
            },
            StateTranslation::Dictionary(page, views_dict) => {
                let views_dict =
                    views_dict.get_or_insert_with(|| binary_views_dict(values, page.dict));
                let translator = DictionaryTranslator(views_dict);
                page.values
                    .translate_and_collect_n_into(values.views_mut(), n, &translator)?;
                if let Some(validity) = values.validity() {
                    validity.extend_constant(n, true);
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
        values.extend_null(n);
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Box<dyn arrow::array::Array>> {
        <Self as utils::Decoder>::finalize(self, data_type, decoded)
    }
}
