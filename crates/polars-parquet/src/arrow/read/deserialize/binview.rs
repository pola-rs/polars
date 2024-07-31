use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::{
    Array, BinaryViewArray, DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray,
    Utf8ViewArray, View,
};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_error::PolarsResult;

use super::binary::decoders::*;
use super::utils::freeze_validity;
use crate::parquet::encoding::hybrid_rle::{self, DictionaryTranslator};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::binary::utils::BinaryIter;
use crate::read::deserialize::utils::{
    self, binary_views_dict, extend_from_decoder, Decoder, PageValidity, StateTranslation,
    TranslatedHybridRle,
};
use crate::read::PrimitiveLogicalType;

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> StateTranslation<'a, BinViewDecoder> for BinaryStateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        decoder: &BinViewDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&PageValidity<'a>>,
    ) -> PolarsResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
        Self::new(page, dict, page_validity, is_string)
    }

    fn len_when_not_nullable(&self) -> usize {
        Self::len_when_not_nullable(self)
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        Self::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut BinViewDecoder,
        decoded: &mut <BinViewDecoder as utils::Decoder>::DecodedState,
        page_validity: &mut Option<utils::PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let views_offset = decoded.0.views().len();
        let buffer_offset = decoded.0.completed_buffers().len();

        let mut validate_utf8 = decoder.check_utf8.load(Ordering::Relaxed);

        match self {
            Self::Plain(page_values) => {
                decoder.decode_plain_encoded(
                    decoded,
                    page_values,
                    page_validity.as_mut(),
                    additional,
                )?;

                // Already done in decode_plain_encoded
                validate_utf8 = false;
            },
            Self::Dictionary(page) => {
                decoder.decode_dictionary_encoded(
                    decoded,
                    &mut page.values,
                    page_validity.as_mut(),
                    page.dict,
                    additional,
                )?;

                // Already done in decode_plain_encoded
                validate_utf8 = false;
            },
            Self::Delta(page_values) => {
                let (values, validity) = decoded;
                match page_validity {
                    None => {
                        for value in page_values.by_ref().take(additional) {
                            values.push_value_ignore_validity(value)
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
            Self::DeltaBytes(page_values) => {
                let (values, validity) = decoded;
                match page_validity {
                    None => {
                        for x in page_values.take(additional) {
                            values.push_value_ignore_validity(x)
                        }
                    },
                    Some(page_validity) => extend_from_decoder(
                        validity,
                        page_validity,
                        Some(additional),
                        values,
                        page_values,
                    )?,
                }
            },
        }

        if validate_utf8 {
            decoded
                .0
                .validate_utf8(buffer_offset, views_offset)
                .map_err(|_| ParquetError::oos("Binary view contained invalid UTF-8"))?
        }

        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct BinViewDecoder {
    check_utf8: AtomicBool,
    views_dict: Option<Vec<View>>,
}

impl utils::ExactSize for DecodedStateTuple {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl utils::Decoder for BinViewDecoder {
    type Translation<'a> = BinaryStateTranslation<'a>;
    type Dict = BinaryDict;
    type DecodedState = DecodedStateTuple;
    type Output = Box<dyn Array>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
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
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        let views_offset = values.views().len();
        let buffer_offset = values.completed_buffers().len();

        match page_validity {
            None => {
                for x in page_values.by_ref().take(limit) {
                    values.push_value_ignore_validity(x)
                }
            },
            Some(page_validity) => {
                extend_from_decoder(validity, page_validity, Some(limit), values, page_values)?
            },
        }

        if self.check_utf8.load(Ordering::Relaxed) {
            // @TODO: Better error message
            values
                .validate_utf8(buffer_offset, views_offset)
                .map_err(|_| ParquetError::oos("Binary view contained invalid UTF-8"))?
        }

        Ok(())
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut hybrid_rle::HybridRleDecoder<'a>,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        let validate_utf8 = self.check_utf8.load(Ordering::Relaxed);

        if validate_utf8 && simdutf8::basic::from_utf8(dict.values()).is_err() {
            return Err(ParquetError::oos(
                "Binary view dictionary contained invalid UTF-8",
            ));
        }

        let views_dict = self
            .views_dict
            .get_or_insert_with(|| binary_views_dict(values, dict));
        let translator = DictionaryTranslator(views_dict);

        match page_validity {
            None => {
                page_values.translate_and_collect_n_into(values.views_mut(), limit, &translator)?;
                if let Some(validity) = values.validity() {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                let collector = TranslatedHybridRle::new(page_values, &translator);
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
        let mut array: BinaryViewArray = values.freeze();

        let validity = freeze_validity(validity);
        array = array.with_validity(validity);

        match data_type.to_physical_type() {
            PhysicalType::BinaryView => Ok(array.boxed()),
            PhysicalType::Utf8View => {
                // SAFETY: we already checked utf8
                unsafe {
                    Ok(Utf8ViewArray::new_unchecked(
                        data_type,
                        array.views().clone(),
                        array.data_buffers().clone(),
                        array.validity().cloned(),
                        array.total_bytes_len(),
                        array.total_buffer_len(),
                    )
                    .boxed())
                }
            },
            _ => unreachable!(),
        }
    }
}

impl utils::DictDecodable for BinViewDecoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_data_type = match &data_type {
            ArrowDataType::Dictionary(_, values, _) => values.as_ref().clone(),
            _ => data_type.clone(),
        };

        let mut view_dict = MutableBinaryViewArray::with_capacity(dict.len());
        for v in dict.iter() {
            view_dict.push(v);
        }
        let view_dict = view_dict.freeze();

        let dict = match value_data_type.to_physical_type() {
            PhysicalType::Utf8View => view_dict.to_utf8view().unwrap().boxed(),
            PhysicalType::BinaryView => view_dict.boxed(),
            _ => unreachable!(),
        };

        Ok(DictionaryArray::try_new(data_type, keys, dict).unwrap())
    }
}

impl utils::NestedDecoder for BinViewDecoder {
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
        values.extend_constant(n, <Option<&[u8]>>::None);
    }
}
