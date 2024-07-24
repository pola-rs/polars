use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils;
use super::basic::{deserialize_plain, ValuesDictionary};
use super::DecoderFunction;
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;

#[derive(Debug)]
pub(crate) struct State<'a, P: ParquetNativeType, T: NativeType> {
    is_optional: bool,
    translation: super::basic::StateTranslation<'a, P, T>,
}

impl<'a, P: ParquetNativeType, T: NativeType> utils::PageState<'a> for State<'a, P, T> {
    fn len(&self) -> usize {
        use super::basic::StateTranslation as T;
        match &self.translation {
            T::Plain(values) => values.len(),
            T::Dictionary(values) => values.len(),
            T::ByteStreamSplit(decoder) => decoder.len(),
        }
    }
}

impl<P, T, D> NestedDecoder for super::basic::PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type State<'a> = State<'a, P, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn build_state<'a>(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State<'a>> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        if is_filtered {
            return Err(utils::not_implemented(page));
        }

        use super::basic::StateTranslation as T;
        let translation = match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                T::Plain(ArrayChunks::new(values).unwrap())
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                T::Dictionary(ValuesDictionary::try_new(page, dict)?)
            },
            (Encoding::ByteStreamSplit, _) => {
                T::ByteStreamSplit(byte_stream_split::Decoder::try_new(
                    split_buffer(page)?.values,
                    std::mem::size_of::<P>(),
                )?)
            },
            _ => return Err(utils::not_implemented(page)),
        };

        Ok(State {
            is_optional,
            translation,
        })
    }

    /// Initializes a new state
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
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

        if utils::PageState::len(state) < n {
            return Err(ParquetError::oos("No values left in page"));
        }

        use super::basic::StateTranslation as T;
        match &mut state.translation {
            T::Plain(page_values) => {
                for value in page_values.by_ref().take(n) {
                    values.push(self.decoder.decode(P::from_le_bytes(*value)));
                }
            },
            T::Dictionary(page) => {
                let translator = DictionaryTranslator(page.dict);
                page.values
                    .translate_and_collect_n_into(values, n, &translator)?;
            },
            T::ByteStreamSplit(decoder) => {
                for value in decoder.iter_converted(decode).by_ref().take(n) {
                    values.push(self.decoder.decode(value));
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
        values.resize(values.len() + n, T::default());
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, self.decoder)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn arrow::array::Array>> {
        Ok(Box::new(PrimitiveArray::new(
            data_type,
            values.into(),
            validity.into(),
        )))
    }
}
