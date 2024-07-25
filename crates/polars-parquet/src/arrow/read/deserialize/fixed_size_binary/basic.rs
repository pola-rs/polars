use arrow::array::{Array, DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;
use polars_error::PolarsResult;
use polars_utils::iter::FallibleIterator;

use super::super::utils::{dict_indices_decoder, extend_from_decoder, not_implemented, Decoder};
use super::utils::FixedSizeBinary;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{self, PageValidity};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(std::slice::ChunksExact<'a, u8>),
    Dictionary(HybridRleDecoder<'a>, &'a Vec<u8>),
}

impl<'a> utils::StateTranslation<'a, BinaryDecoder> for StateTranslation<'a> {
    type PlainDecoder = std::slice::ChunksExact<'a, u8>;

    fn new(
        decoder: &BinaryDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinaryDecoder as Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
        _filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                if values.len() % decoder.size != 0 {
                    return Err(ParquetError::oos(format!(
                        "Fixed size binary data length {} is not divisible by size {}",
                        values.len(),
                        decoder.size
                    ))
                    .into());
                }
                let values = values.chunks_exact(decoder.size);
                Ok(Self::Plain(values))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                let values = dict_indices_decoder(page)?;
                Ok(Self::Dictionary(values, dict))
            },
            _ => Err(not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(v) => v.len(),
            Self::Dictionary(v, _) => v.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(v) => _ = v.nth(n - 1),
            Self::Dictionary(v, _) => v.skip_in_place(n)?,
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut BinaryDecoder,
        decoded: &mut <BinaryDecoder as Decoder>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        use StateTranslation as T;
        match self {
            T::Plain(page_values) => decoder.decode_plain_encoded(
                decoded,
                page_values,
                page_validity.as_mut(),
                additional,
            )?,
            T::Dictionary(page_values, dict) => decoder.decode_dictionary_encoded(
                decoded,
                page_values,
                page_validity.as_mut(),
                dict,
                additional,
            )?,
        }

        Ok(())
    }
}

pub(crate) struct BinaryDecoder {
    pub(crate) size: usize,
}

impl<T> utils::ExactSize for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl utils::ExactSize for (FixedSizeBinary, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl Decoder for BinaryDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = Vec<u8>;
    type DecodedState = (FixedSizeBinary, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            FixedSizeBinary::with_capacity(capacity, self.size),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        page.buffer.into_vec()
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        match page_validity {
            None => {
                // @TODO: This can be done through a extend
                for x in page_values.by_ref().take(limit) {
                    values.push(x)
                }
            },
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(limit), values, page_values)?
            },
        }

        Ok(())
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut HybridRleDecoder<'a>,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        match page_validity {
            None => {
                // @TODO: Use Gatherer
                for x in page_values
                    .by_ref()
                    .map(|index| {
                        let index = index as usize;
                        &dict[index * self.size..(index + 1) * self.size]
                    })
                    .take(limit)
                {
                    values.push(x)
                }
                page_values.get_result()?;
            },
            Some(page_validity) => {
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(limit),
                    values,
                    page_values.by_ref().map(|index| {
                        let index = index as usize;
                        &dict[index * self.size..(index + 1) * self.size]
                    }),
                )?;
                page_values.get_result()?;
            },
        }

        Ok(())
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        Ok(Box::new(FixedSizeBinaryArray::new(
            data_type,
            values.values.into(),
            validity.into(),
        )))
    }

    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        (values, validity): (Vec<K>, Option<Bitmap>),
    ) -> ParquetResult<DictionaryArray<K>> {
        let dict =
            FixedSizeBinaryArray::new(ArrowDataType::FixedSizeBinary(self.size), dict.into(), None);
        let array = PrimitiveArray::<K>::new(K::PRIMITIVE.into(), values.into(), validity);
        Ok(DictionaryArray::try_new(data_type, array, Box::new(dict)).unwrap())
    }
}

impl utils::NestedDecoder for BinaryDecoder {
    fn validity_extend(_: &mut utils::State<'_, Self>, (_, validity): &mut Self::DecodedState, value: bool, n: usize) {
        validity.extend_constant(n, value);
    }

    fn values_extend_nulls(_: &mut utils::State<'_, Self>, (values, _): &mut Self::DecodedState, n: usize) {
        values.extend_constant(n);
    }
}
