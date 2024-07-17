use std::cell::Cell;
use std::collections::VecDeque;

use arrow::array::{Array, ArrayRef, BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_error::PolarsResult;

use super::super::binary::decoders::*;
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{
    self, binary_views_dict, extend_from_decoder, next, DecodedState, MaybeNext, PageValidity,
    StateTranslation, TranslatedHybridRle,
};
use crate::read::{PagesIter, PrimitiveLogicalType};

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> StateTranslation<'a, BinViewDecoder> for BinaryStateTranslation<'a> {
    fn new(
        decoder: &BinViewDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&PageValidity<'a>>,
        filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.set(is_string);
        Self::new(page, dict, page_validity, filter, is_string)
    }

    fn len_when_not_nullable(&self) -> usize {
        Self::len_when_not_nullable(self)
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        Self::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &BinViewDecoder,
        decoded: &mut <BinViewDecoder as utils::Decoder>::DecodedState,
        page_validity: &mut Option<utils::PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;
        let views_offset = values.views().len();
        let buffer_offset = values.completed_buffers().len();

        let mut validate_utf8 = decoder.check_utf8.take();

        match (self, page_validity) {
            (Self::Unit(page_values), None) => {
                for x in page_values.by_ref().take(additional) {
                    values.push_value_ignore_validity(x)
                }
            },
            (Self::Unit(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
            (Self::Dictionary(page, views_dict), None) => {
                // Already done on the dict.
                validate_utf8 = false;

                let views_dict =
                    views_dict.get_or_insert_with(|| binary_views_dict(values, page.dict));
                let translator = DictionaryTranslator(views_dict);

                page.values.translate_and_collect_n_into(
                    values.views_mut(),
                    additional,
                    &translator,
                )?;
                if let Some(validity) = values.validity() {
                    validity.extend_constant(additional, true);
                }
            },
            (Self::Dictionary(page, views_dict), Some(page_validity)) => {
                // Already done on the dict.
                validate_utf8 = false;

                let views_dict =
                    views_dict.get_or_insert_with(|| binary_views_dict(values, page.dict));
                let translator = DictionaryTranslator(views_dict);
                let collector = TranslatedHybridRle::new(&mut page.values, &translator);

                extend_from_decoder(validity, page_validity, Some(additional), values, collector)?;
            },
            (Self::Delta(page_values), None) => {
                for value in page_values.by_ref().take(additional) {
                    values.push_value_ignore_validity(value)
                }
            },
            (Self::Delta(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
            (Self::DeltaBytes(page_values), None) => {
                for x in page_values.take(additional) {
                    values.push_value_ignore_validity(x)
                }
            },
            (Self::DeltaBytes(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
        }

        if validate_utf8 {
            // @TODO: Better error message
            values
                .validate_utf8(buffer_offset, views_offset)
                .map_err(|_| ParquetError::oos("invalid UTF-8"))
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
struct BinViewDecoder {
    check_utf8: Cell<bool>,
}

impl DecodedState for DecodedStateTuple {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> utils::Decoder<'a> for BinViewDecoder {
    type Translation = BinaryStateTranslation<'a>;
    type Dict = BinaryDict;
    type DecodedState = DecodedStateTuple;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub struct BinaryViewArrayIter<I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    items: VecDeque<DecodedStateTuple>,
    dict: Option<BinaryDict>,
    chunk_size: Option<usize>,
    remaining: usize,
}
impl<I: PagesIter> BinaryViewArrayIter<I> {
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        chunk_size: Option<usize>,
        num_rows: usize,
    ) -> Self {
        Self {
            iter,
            data_type,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<I: PagesIter> Iterator for BinaryViewArrayIter<I> {
    type Item = PolarsResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        let decoder = BinViewDecoder::default();
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                self.chunk_size,
                &decoder,
            );
            match maybe_state {
                MaybeNext::Some(Ok((values, validity))) => {
                    return Some(finish(&self.data_type, values, validity))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}

pub(super) fn finish(
    data_type: &ArrowDataType,
    values: MutableBinaryViewArray<[u8]>,
    validity: MutableBitmap,
) -> PolarsResult<Box<dyn Array>> {
    let mut array: BinaryViewArray = values.into();
    let validity: Bitmap = validity.into();

    if validity.unset_bits() != validity.len() {
        array = array.with_validity(Some(validity))
    }

    match data_type.to_physical_type() {
        PhysicalType::BinaryView => Ok(array.boxed()),
        PhysicalType::Utf8View => {
            // SAFETY: we already checked utf8
            unsafe {
                Ok(Utf8ViewArray::new_unchecked(
                    data_type.clone(),
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
