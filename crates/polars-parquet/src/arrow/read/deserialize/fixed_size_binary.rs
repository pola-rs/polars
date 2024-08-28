use arrow::array::{DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;

use super::utils::{dict_indices_decoder, extend_from_decoder, freeze_validity, Decoder};
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::{hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::{self, BatchableCollector, GatheredHybridRle, PageValidity};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(&'a [u8], usize),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
}

#[derive(Debug)]
pub struct FixedSizeBinary {
    pub values: Vec<u8>,
    pub size: usize,
}

impl<'a> utils::StateTranslation<'a, BinaryDecoder> for StateTranslation<'a> {
    type PlainDecoder = &'a [u8];

    fn new(
        decoder: &BinaryDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinaryDecoder as Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                if values.len() % decoder.size != 0 {
                    return Err(ParquetError::oos(format!(
                        "Fixed size binary data length {} is not divisible by size {}",
                        values.len(),
                        decoder.size
                    )));
                }
                Ok(Self::Plain(values, decoder.size))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values = dict_indices_decoder(page)?;
                Ok(Self::Dictionary(values))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(v, size) => v.len() / size,
            Self::Dictionary(v) => v.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(v, size) => *v = &v[usize::min(v.len(), n * *size)..],
            Self::Dictionary(v) => v.skip_in_place(n)?,
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut BinaryDecoder,
        decoded: &mut <BinaryDecoder as Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<PageValidity<'a>>,
        dict: Option<&'a <BinaryDecoder as Decoder>::Dict>,
        additional: usize,
    ) -> ParquetResult<()> {
        use StateTranslation as T;
        match self {
            T::Plain(page_values, _) => decoder.decode_plain_encoded(
                decoded,
                page_values,
                is_optional,
                page_validity.as_mut(),
                additional,
            )?,
            T::Dictionary(page_values) => decoder.decode_dictionary_encoded(
                decoded,
                page_values,
                is_optional,
                page_validity.as_mut(),
                dict.unwrap(),
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
        self.0.values.len() / self.0.size
    }
}

impl Decoder for BinaryDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = Vec<u8>;
    type DecodedState = (FixedSizeBinary, MutableBitmap);
    type Output = FixedSizeBinaryArray;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        let size = self.size;

        (
            FixedSizeBinary {
                values: Vec::with_capacity(capacity * size),
                size,
            },
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> ParquetResult<Self::Dict> {
        Ok(page.buffer.into_vec())
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        struct FixedSizeBinaryCollector<'a, 'b> {
            slice: &'b mut &'a [u8],
            size: usize,
        }

        impl<'a, 'b> BatchableCollector<(), Vec<u8>> for FixedSizeBinaryCollector<'a, 'b> {
            fn reserve(target: &mut Vec<u8>, n: usize) {
                target.reserve(n);
            }

            fn push_n(&mut self, target: &mut Vec<u8>, n: usize) -> ParquetResult<()> {
                let n = usize::min(n, self.slice.len() / self.size);
                target.extend_from_slice(&self.slice[..n * self.size]);
                *self.slice = &self.slice[n * self.size..];
                Ok(())
            }

            fn push_n_nulls(&mut self, target: &mut Vec<u8>, n: usize) -> ParquetResult<()> {
                target.resize(target.len() + n * self.size, 0);
                Ok(())
            }

            fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
                let n = usize::min(n, self.slice.len() / self.size);
                *self.slice = &self.slice[n * self.size..];
                Ok(())
            }
        }

        let mut collector = FixedSizeBinaryCollector {
            slice: page_values,
            size: self.size,
        };

        match page_validity {
            None => {
                collector.push_n(&mut values.values, limit)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => extend_from_decoder(
                validity,
                page_validity,
                Some(limit),
                &mut values.values,
                collector,
            )?,
        }

        Ok(())
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut hybrid_rle::HybridRleDecoder<'a>,
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        struct FixedSizeBinaryGatherer<'a> {
            dict: &'a [u8],
            size: usize,
        }

        impl<'a> HybridRleGatherer<&'a [u8]> for FixedSizeBinaryGatherer<'a> {
            type Target = Vec<u8>;

            fn target_reserve(&self, target: &mut Self::Target, n: usize) {
                target.reserve(n * self.size);
            }

            fn target_num_elements(&self, target: &Self::Target) -> usize {
                target.len() / self.size
            }

            fn hybridrle_to_target(&self, value: u32) -> ParquetResult<&'a [u8]> {
                let value = value as usize;

                if value * self.size >= self.dict.len() {
                    return Err(ParquetError::oos(
                        "Fixed size binary dictionary index out-of-range",
                    ));
                }

                Ok(&self.dict[value * self.size..(value + 1) * self.size])
            }

            fn gather_one(&self, target: &mut Self::Target, value: &'a [u8]) -> ParquetResult<()> {
                // We make the null value length 0, which allows us to do this.
                if value.is_empty() {
                    target.resize(target.len() + self.size, 0);
                    return Ok(());
                }

                target.extend_from_slice(value);
                Ok(())
            }

            fn gather_repeated(
                &self,
                target: &mut Self::Target,
                value: &'a [u8],
                n: usize,
            ) -> ParquetResult<()> {
                // We make the null value length 0, which allows us to do this.
                if value.is_empty() {
                    target.resize(target.len() + n * self.size, 0);
                    return Ok(());
                }

                debug_assert_eq!(value.len(), self.size);
                for _ in 0..n {
                    target.extend(value);
                }

                Ok(())
            }
        }

        let gatherer = FixedSizeBinaryGatherer {
            dict,
            size: self.size,
        };

        // @NOTE:
        // This is a special case in our gatherer. If the length of the value is 0, then we just
        // resize with the appropriate size. Important is that this also works for FSL with size=0.
        let null_value = &[];

        match page_validity {
            None => {
                page_values.gather_n_into(&mut values.values, limit, &gatherer)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                let collector = GatheredHybridRle::new(page_values, &gatherer, null_value);

                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(limit),
                    &mut values.values,
                    collector,
                )?;
            },
        }

        Ok(())
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);
        Ok(FixedSizeBinaryArray::new(
            data_type,
            values.values.into(),
            validity,
        ))
    }
}

impl utils::DictDecodable for BinaryDecoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let dict =
            FixedSizeBinaryArray::new(ArrowDataType::FixedSizeBinary(self.size), dict.into(), None);
        Ok(DictionaryArray::try_new(data_type, keys, Box::new(dict)).unwrap())
    }
}

impl utils::NestedDecoder for BinaryDecoder {
    fn validity_extend(
        _: &mut utils::State<'_, Self>,
        (_, validity): &mut Self::DecodedState,
        value: bool,
        n: usize,
    ) {
        validity.extend_constant(n, value);
    }

    fn values_extend_nulls(
        _: &mut utils::State<'_, Self>,
        (values, _): &mut Self::DecodedState,
        n: usize,
    ) {
        values
            .values
            .resize(values.values.len() + n * values.size, 0);
    }
}
