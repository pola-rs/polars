use std::default::Default;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::specification::try_check_utf8;
use arrow::array::{Array, BinaryArray, DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::Offset;
use polars_error::PolarsResult;
use polars_utils::iter::FallibleIterator;

use super::super::utils;
use super::super::utils::extend_from_decoder;
use super::decoders::*;
use super::utils::*;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils::{Decoder, StateTranslation};
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
        filter: Option<&utils::filter::Filter<'a>>,
    ) -> PolarsResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
        BinaryStateTranslation::new(page, dict, page_validity, filter, is_string)
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
        match page_validity {
            None => {
                // @TODO: Make this into a gatherer
                for x in page_values
                    .by_ref()
                    .map(|index| dict.value(index as usize))
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
                    &mut page_values.by_ref().map(|index| dict.value(index as usize)),
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
        super::finalize(data_type, values, validity)
    }

    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        (values, validity): (Vec<K>, Option<Bitmap>),
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

        let indices = PrimitiveArray::new(K::PRIMITIVE.into(), values.into(), validity);

        // @TODO: Is this datatype correct?
        Ok(DictionaryArray::try_new(data_type, indices, dict).unwrap())
    }
}

impl<O: Offset> utils::NestedDecoder for BinaryDecoder<O> {
    fn validity_extend((_, validity): &mut Self::DecodedState, value: bool, n: usize) {
        validity.extend_constant(n, value);
    }

    fn values_extend_nulls((values, _): &mut Self::DecodedState, n: usize) {
        values.extend_constant(n);
    }
}
