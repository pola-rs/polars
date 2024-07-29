use std::default::Default;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::specification::try_check_utf8;
use arrow::array::{Array, BinaryArray, DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::utils;
use super::super::utils::extend_from_decoder;
use super::decoders::*;
use super::utils::*;
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils::{Decoder, GatheredHybridRle, StateTranslation};
use crate::read::PrimitiveLogicalType;

impl<O: Offset> utils::ExactSize for (Binary<O>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, O: Offset> StateTranslation<'a, BinaryDecoder<O>> for BinaryStateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        decoder: &BinaryDecoder<O>,
        page: &'a DataPage,
        dict: Option<&'a <BinaryDecoder<O> as utils::Decoder>::Dict>,
        page_validity: Option<&utils::PageValidity<'a>>,
    ) -> PolarsResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
        BinaryStateTranslation::new(page, dict, page_validity, is_string)
    }

    fn len_when_not_nullable(&self) -> usize {
        BinaryStateTranslation::len_when_not_nullable(self)
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        BinaryStateTranslation::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut BinaryDecoder<O>,
        decoded: &mut <BinaryDecoder<O> as utils::Decoder>::DecodedState,
        page_validity: &mut Option<utils::PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let len_before = decoded.0.offsets.len();

        use BinaryStateTranslation as T;
        match self {
            T::Plain(page_values) => decoder.decode_plain_encoded(
                decoded,
                page_values,
                page_validity.as_mut(),
                additional,
            )?,
            T::Dictionary(page) => decoder.decode_dictionary_encoded(
                decoded,
                &mut page.values,
                page_validity.as_mut(),
                page.dict,
                additional,
            )?,
            T::Delta(page) => {
                let (values, validity) = decoded;

                match page_validity {
                    None => values
                        .extend_lengths(page.lengths.by_ref().take(additional), &mut page.values),
                    Some(page_validity) => {
                        let Binary {
                            offsets,
                            values: values_,
                        } = values;

                        let last_offset = *offsets.last();
                        extend_from_decoder(
                            validity,
                            page_validity,
                            Some(additional),
                            offsets,
                            page.lengths.by_ref(),
                        )?;

                        let length = *offsets.last() - last_offset;

                        let (consumed, remaining) = page.values.split_at(length.to_usize());
                        page.values = remaining;
                        values_.extend_from_slice(consumed);
                    },
                }
            },
            T::DeltaBytes(page_values) => {
                let (values, validity) = decoded;

                match page_validity {
                    None => {
                        for x in page_values.take(additional) {
                            values.push(x)
                        }
                    },
                    Some(page_validity) => {
                        extend_from_decoder(
                            validity,
                            page_validity,
                            Some(additional),
                            values,
                            page_values,
                        )?;
                    },
                }
            },
        }

        // @TODO: Double checking
        if decoder.check_utf8.load(Ordering::Relaxed) {
            // @TODO: This can report a better error.
            let offsets = &decoded.0.offsets.as_slice()[len_before..];
            try_check_utf8(offsets, &decoded.0.values)
                .map_err(|_| ParquetError::oos("invalid utf-8"))?;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) struct BinaryDecoder<O: Offset> {
    phantom_o: std::marker::PhantomData<O>,
    check_utf8: AtomicBool,
}

impl utils::ExactSize for BinaryDict {
    fn len(&self) -> usize {
        BinaryDict::len(self)
    }
}

impl<O: Offset> utils::Decoder for BinaryDecoder<O> {
    type Translation<'a> = BinaryStateTranslation<'a>;
    type Dict = BinaryDict;
    type DecodedState = (Binary<O>, MutableBitmap);
    type Output = Box<dyn Array>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as StateTranslation<'a, Self>>::PlainDecoder,
        page_validity: Option<&mut utils::PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        let len_before = values.offsets.len();

        match page_validity {
            None => {
                for x in page_values.by_ref().take(limit) {
                    values.push(x);
                }
            },
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(limit), values, page_values)?
            },
        }

        if self.check_utf8.load(Ordering::Relaxed) {
            // @TODO: This can report a better error.
            let offsets = &values.offsets.as_slice()[len_before..];
            try_check_utf8(offsets, &values.values).map_err(|_| ParquetError::oos("invalid utf-8"))
        } else {
            Ok(())
        }
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut HybridRleDecoder<'a>,
        page_validity: Option<&mut utils::PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        struct BinaryGatherer<'a, O> {
            dict: &'a BinaryDict,
            _pd: std::marker::PhantomData<O>,
        }

        impl<'a, O: Offset> HybridRleGatherer<&'a [u8]> for BinaryGatherer<'a, O> {
            type Target = Binary<O>;

            fn target_reserve(&self, target: &mut Self::Target, n: usize) {
                // @NOTE: This is an estimation for the reservation. It will probably not be
                // accurate, but then it is a lot better than not allocating.
                target.offsets.reserve(n);
                target.values.reserve(n);
            }

            fn target_num_elements(&self, target: &Self::Target) -> usize {
                target.offsets.len_proxy()
            }

            fn hybridrle_to_target(&self, value: u32) -> ParquetResult<&'a [u8]> {
                let value = value as usize;

                if value >= self.dict.len() {
                    return Err(ParquetError::oos("Binary dictionary index out-of-range"));
                }

                Ok(self.dict.value(value))
            }

            fn gather_one(&self, target: &mut Self::Target, value: &'a [u8]) -> ParquetResult<()> {
                target.push(value);
                Ok(())
            }

            fn gather_repeated(
                &self,
                target: &mut Self::Target,
                value: &'a [u8],
                n: usize,
            ) -> ParquetResult<()> {
                for _ in 0..n {
                    target.push(value);
                }
                Ok(())
            }
        }

        let gatherer = BinaryGatherer {
            dict,
            _pd: std::marker::PhantomData,
        };

        match page_validity {
            None => {
                page_values.gather_n_into(values, limit, &gatherer)?;
            },
            Some(page_validity) => {
                let collector = GatheredHybridRle::new(page_values, &gatherer, &[]);

                extend_from_decoder(validity, page_validity, Some(limit), values, collector)?;
            },
        }

        Ok(())
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        super::finalize(data_type, values, validity)
    }
}

impl<O: Offset> utils::DictDecodable for BinaryDecoder<O> {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_data_type = match data_type.clone() {
            ArrowDataType::Dictionary(_, values, _) => *values,
            v => v,
        };

        let (_, dict_offsets, dict_values, _) = dict.into_inner();
        let dict = match value_data_type.to_physical_type() {
            PhysicalType::Utf8 | PhysicalType::LargeUtf8 => {
                Utf8Array::new(value_data_type, dict_offsets, dict_values, None).boxed()
            },
            PhysicalType::Binary | PhysicalType::LargeBinary => {
                BinaryArray::new(value_data_type, dict_offsets, dict_values, None).boxed()
            },
            _ => unreachable!(),
        };

        // @TODO: Is this datatype correct?
        Ok(DictionaryArray::try_new(data_type, keys, dict).unwrap())
    }
}

impl<O: Offset> utils::NestedDecoder for BinaryDecoder<O> {
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
        values.extend_constant(n);
    }
}
