use std::collections::VecDeque;

use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::utils::{not_implemented, MaybeNext, PageState};
use super::utils::FixedSizeBinary;
use crate::arrow::read::deserialize::nested_utils::{next, NestedDecoder};
use crate::arrow::read::{InitNested, NestedState};
use crate::parquet::encoding::hybrid_rle::gatherer::{SliceDictionaryTranslator, Translator};
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::parquet::read::BasicDecompressor;
use crate::parquet::schema::Repetition;
use crate::read::deserialize::utils::dict_indices_decoder;
use crate::read::CompressedPagesIter;

#[derive(Debug)]
struct State<'a> {
    is_optional: bool,
    translation: StateTranslation<'a>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum StateTranslation<'a> {
    Plain(std::slice::ChunksExact<'a, u8>),
    Dictionary {
        values: HybridRleDecoder<'a>,
        dict: &'a [u8],
    },
}

impl<'a> PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Plain(chunks) => chunks.len(),
            StateTranslation::Dictionary { values, .. } => values.len(),
        }
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder {
    size: usize,
}

impl NestedDecoder for BinaryDecoder {
    type State<'a> = State<'a>;
    type Dict = Vec<u8>;
    type DecodedState = (FixedSizeBinary, MutableBitmap);

    fn build_state<'a>(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State<'a>> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        let translation = match (page.encoding(), dict, is_filtered) {
            (Encoding::Plain, _, false) => {
                let values = page.buffer();
                assert_eq!(values.len() % self.size, 0);
                let values = values.chunks_exact(self.size);
                StateTranslation::Plain(values)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false) => {
                let values = dict_indices_decoder(page)?;
                StateTranslation::Dictionary {
                    values,
                    dict: &dict[..],
                }
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
        state: &mut Self::State<'_>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        if PageState::len(state) < n {
            return Err(ParquetError::oos("No values left in page"));
        }

        match &mut state.translation {
            StateTranslation::Plain(page_values) => {
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

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        page.buffer.into_vec()
    }

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

        Ok(Box::new(FixedSizeBinaryArray::new(
            data_type.clone(),
            values.values.into(),
            validity,
        )))
    }
}

pub struct NestedIter<I: CompressedPagesIter> {
    iter: BasicDecompressor<I>,
    data_type: ArrowDataType,
    size: usize,
    init: Vec<InitNested>,
    items: VecDeque<(NestedState, (FixedSizeBinary, MutableBitmap))>,
    dict: Option<Vec<u8>>,
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

pub fn finish(
    data_type: &ArrowDataType,
    values: FixedSizeBinary,
    validity: MutableBitmap,
) -> FixedSizeBinaryArray {
    FixedSizeBinaryArray::new(data_type.clone(), values.values.into(), validity.into())
}

impl<I: CompressedPagesIter> Iterator for NestedIter<I> {
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
