use arrow::array::{Array, BinaryArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;
use polars_compute::filter::filter_with_bitmap;

use super::utils::dict_indices_decoder;
use super::{Filter, PredicateFilter};
use crate::parquet::encoding::{Encoding, hybrid_rle};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage, split_buffer};
use crate::read::deserialize::utils::{self};
use crate::read::expr::{ParquetScalar, SpecializedParquetColumnExpr};

mod dictionary;
mod plain;

type DecodedStateTuple = (Vec<u8>, Vec<i64>);

impl<'a> utils::StateTranslation<'a, BinaryDecoder> for StateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        _decoder: &BinaryDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinaryDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let values = BinaryIter::new(values, page.num_values());

                Ok(Self::Plain(values))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values =
                    dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))?;
                Ok(Self::Dictionary(values))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn num_rows(&self) -> usize {
        match self {
            StateTranslation::Plain(i) => i.max_num_values,
            StateTranslation::Dictionary(i) => i.len(),
        }
    }
}

pub struct BinaryDecoder;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
}

impl utils::Decoded for DecodedStateTuple {
    fn len(&self) -> usize {
        self.1.len().saturating_sub(1)
    }

    fn extend_nulls(&mut self, n: usize) {
        let last = *self.1.last().unwrap();
        self.1.extend(std::iter::repeat_n(last, n));
    }

    fn remaining_capacity(&self) -> usize {
        self.1.capacity() - self.1.len()
    }
}

impl utils::Decoder for BinaryDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = BinaryArray<i64>;
    type DecodedState = DecodedStateTuple;
    type Output = BinaryArray<i64>;

    const CHUNKED: bool = true;

    fn with_capacity(&self, _capacity: usize) -> Self::DecodedState {
        // Handled in extend
        (Vec::new(), Vec::new())
    }

    fn evaluate_dict_predicate(
        &self,
        dict: &Self::Dict,
        predicate: &PredicateFilter,
    ) -> ParquetResult<Bitmap> {
        Ok(predicate.predicate.evaluate(dict as &dyn Array))
    }

    fn evaluate_predicate(
        &mut self,
        _state: &utils::State<'_, Self>,
        _predicate: Option<&SpecializedParquetColumnExpr>,
        _pred_true_mask: &mut BitmapBuilder,
        _dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        Ok(false)
    }

    fn apply_dictionary(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _dict: &Self::Dict,
    ) -> ParquetResult<()> {
        Ok(())
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let values = &page.buffer;
        let num_values = page.num_values;
        let mut target = Vec::<u8>::new();
        let mut offsets = Vec::<i64>::with_capacity(page.num_values + 1);

        plain::decode_plain(values, num_values, &mut target, &mut offsets)?;

        let values = target.into();
        let offsets = offsets.into();
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets) };
        let arr = BinaryArray::new(ArrowDataType::LargeBinary, offsets, values, None);

        Ok(arr)
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        assert!(!is_optional);

        let array = additional
            .as_any()
            .downcast_ref::<BinaryArray<i64>>()
            .unwrap();

        let offsets = array.offsets();
        let fst = *offsets.first();
        let lst = *offsets.last();
        let current_lst = *decoded.1.last().unwrap();
        decoded
            .0
            .extend_from_slice(&array.values()[fst as usize..lst as usize]);
        decoded
            .1
            .extend(offsets.iter().map(|o| o + current_lst - fst));

        Ok(())
    }

    fn extend_filtered_with_state(
        &mut self,
        mut state: utils::State<'_, Self>,
        _decoded: &mut Self::DecodedState,
        filter: Option<super::Filter>,
        chunks: &mut Vec<Self::Output>,
    ) -> ParquetResult<()> {
        assert!(state.page_validity.is_none());

        let mut target = Vec::new();
        let mut offsets =
            Vec::with_capacity(utils::StateTranslation::num_rows(&state.translation) + 1);

        match state.translation {
            StateTranslation::Plain(iter) => {
                plain::decode_plain(iter.values, iter.max_num_values, &mut target, &mut offsets)?
            },
            StateTranslation::Dictionary(ref mut indexes) => {
                let dict = state.dict.unwrap();
                dictionary::decode_dictionary(indexes.clone(), &mut target, &mut offsets, dict)?;
            },
        }

        if let Some(Filter::Range(slice)) = &filter
            && (slice.start > 0 || slice.end < offsets.len() - 1)
        {
            let mut new_target = Vec::new();
            let mut new_offsets = Vec::new();

            let buffer_start = offsets[slice.start] as usize;
            let buffer_end = offsets[slice.end.min(offsets.len() - 1)] as usize;

            new_target.extend(target.drain(buffer_start..buffer_end));
            new_offsets.extend(offsets.drain(slice.start..slice.end.min(offsets.len() - 1)));

            target = new_target;
            offsets = new_offsets;
        }

        let values = target.into();
        let offsets = offsets.into();
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets) };

        let mut array = BinaryArray::new(ArrowDataType::LargeBinary, offsets, values, None);

        if let Some(Filter::Mask(mask)) = &filter {
            array = filter_with_bitmap(&array, mask)
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .unwrap()
                .clone();
        }

        chunks.push(array);

        Ok(())
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, mut offsets): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        assert!(values.is_empty());
        assert!(offsets.is_empty());
        offsets.push(0);
        let values: Buffer<u8> = values.into();
        let offsets: OffsetsBuffer<i64> = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
        Ok(BinaryArray::<i64>::new(dtype, offsets, values, None))
    }

    fn extend_constant(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _length: usize,
        _value: &ParquetScalar,
    ) -> ParquetResult<()> {
        todo!()
    }
}

#[derive(Debug)]
pub struct BinaryIter<'a> {
    values: &'a [u8],

    /// A maximum number of items that this [`BinaryIter`] may produce.
    ///
    /// This equal the length of the iterator i.f.f. the data encoded by the [`BinaryIter`] is not
    /// nullable.
    max_num_values: usize,
}

impl<'a> BinaryIter<'a> {
    pub fn new(values: &'a [u8], max_num_values: usize) -> Self {
        Self {
            values,
            max_num_values,
        }
    }
}

impl<'a> Iterator for BinaryIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.max_num_values == 0 {
            assert!(self.values.is_empty());
            return None;
        }

        let (length, remaining) = self.values.split_at(4);
        let length: [u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(length) as usize;
        let (result, remaining) = remaining.split_at(length);
        self.max_num_values -= 1;
        self.values = remaining;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.max_num_values))
    }
}
