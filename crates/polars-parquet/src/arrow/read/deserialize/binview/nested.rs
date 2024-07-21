use std::collections::VecDeque;

use arrow::array::{ArrayRef, MutableBinaryViewArray, View};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::read::BasicDecompressor;
use crate::read::deserialize::binary::decoders::{deserialize_plain, BinaryDict, ValuesDictionary};
use crate::read::deserialize::binary::utils::BinaryIter;
use crate::read::deserialize::nested_utils::{next, NestedDecoder};
use crate::read::deserialize::utils::{
    self, binary_views_dict, not_implemented, page_is_filtered, page_is_optional, MaybeNext,
    PageState,
};
use crate::read::{CompressedPagesIter, InitNested, NestedState};

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

struct BinViewDecoder;

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> NestedDecoder<'a> for BinViewDecoder {
    type State = State<'a>;
    type Dictionary = BinaryDict;
    type DecodedState = DecodedStateTuple;

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
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
        state: &mut Self::State,
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

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub struct NestedIter<I: CompressedPagesIter> {
    iter: BasicDecompressor<I>,
    data_type: ArrowDataType,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, DecodedStateTuple)>,
    dict: Option<BinaryDict>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<I: CompressedPagesIter> NestedIter<I> {
    pub fn new(
        iter: BasicDecompressor<I>,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            iter,
            data_type,
            init,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<I: CompressedPagesIter> Iterator for NestedIter<I> {
    type Item = PolarsResult<(NestedState, ArrayRef)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            use utils::Decoder;

            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &BinViewDecoder,
            );
            match maybe_state {
                MaybeNext::Some(Ok((nested, decoded))) => {
                    return Some(
                        super::basic::BinViewDecoder::default()
                            .finalize(self.data_type.clone(), decoded)
                            .map(|array| (nested, array))
                            .map_err(Into::into),
                    )
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue, // Using continue in a loop instead of calling next helps prevent stack overflow.
            }
        }
    }
}
