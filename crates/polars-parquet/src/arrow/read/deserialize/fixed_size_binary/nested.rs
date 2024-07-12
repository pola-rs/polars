use std::collections::VecDeque;

use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::utils::{not_implemented, MaybeNext, PageState};
use super::utils::FixedSizeBinary;
use crate::arrow::read::deserialize::fixed_size_binary::basic::finish;
use crate::arrow::read::deserialize::nested_utils::{next, NestedDecoder};
use crate::arrow::read::{InitNested, NestedState, PagesIter};
use crate::parquet::encoding::hybrid_rle::gatherer::{SliceDictionaryTranslator, Translator};
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::parquet::schema::Repetition;
use crate::read::deserialize::utils::dict_indices_decoder;

#[derive(Debug)]
struct State<'a> {
    is_optional: bool,
    translation: StateTranslation<'a>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum StateTranslation<'a> {
    Unit(std::slice::ChunksExact<'a, u8>),
    Dictionary {
        values: HybridRleDecoder<'a>,
        dict: &'a [u8],
    },
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Unit(chunks) => chunks.len(),
            StateTranslation::Dictionary {
                values: decoder, ..
            } => decoder.len(),
        }
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder {
    size: usize,
}

impl<'a> NestedDecoder<'a> for BinaryDecoder {
    type State = State<'a>;
    type Dictionary = Vec<u8>;
    type DecodedState = (FixedSizeBinary, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        let translation = match (page.encoding(), dict, is_filtered) {
            (Encoding::Plain, _, false) => {
                let values = page.buffer();
                assert_eq!(values.len() % self.size, 0);
                let values = values.chunks_exact(self.size);
                StateTranslation::Unit(values)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false) => {
                let values = dict_indices_decoder(page)?;
                StateTranslation::Dictionary { values, dict }
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
            FixedSizeBinary::with_capacity(capacity, self.size),
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

        if PageState::len(state) < n {
            return Err(ParquetError::oos("No values left in page"));
        }

        match &mut state.translation {
            StateTranslation::Unit(page_values) => {
                for value in page_values.by_ref().take(n) {
                    values.push(value);
                }
            },
            StateTranslation::Dictionary {
                values: page_values,
                dict,
            } => {
                let translator = SliceDictionaryTranslator::new(dict, self.size);
                for value in page_values.by_ref().take(n) {
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
        values.extend_constant(n);
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        page.buffer.clone()
    }
}

pub struct NestedIter<I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    size: usize,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, (FixedSizeBinary, MutableBitmap))>,
    dict: Option<Vec<u8>>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<I: PagesIter> NestedIter<I> {
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        let size = FixedSizeBinaryArray::get_size(&data_type);
        Self {
            iter,
            data_type,
            size,
            init,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<I: PagesIter> Iterator for NestedIter<I> {
    type Item = PolarsResult<(NestedState, FixedSizeBinaryArray)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &BinaryDecoder { size: self.size },
            );
            match maybe_state {
                MaybeNext::Some(Ok((nested, decoded))) => {
                    return Some(Ok((nested, finish(&self.data_type, decoded.0, decoded.1))))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
