use arrow::array::{FixedSizeBinaryArray, Splitable};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::storage::SharedStorage;
use arrow::types::{
    Bytes12Alignment4, Bytes16Alignment16, Bytes1Alignment1, Bytes2Alignment2, Bytes32Alignment16,
    Bytes4Alignment4, Bytes8Alignment8,
};
use bytemuck::Zeroable;

use super::dictionary_encoded::append_validity;
use super::utils::array_chunks::ArrayChunks;
use super::utils::{dict_indices_decoder, freeze_validity, Decoder};
use super::{Filter, PredicateFilter};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::encoding::{hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::dictionary_encoded::constrain_page_validity;
use crate::read::deserialize::utils;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(&'a [u8], usize),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
}

impl<'a> utils::StateTranslation<'a, BinaryDecoder> for StateTranslation<'a> {
    type PlainDecoder = &'a [u8];

    fn new(
        decoder: &BinaryDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinaryDecoder as Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
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
                let values =
                    dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))?;
                Ok(Self::Dictionary(values))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn num_rows(&self) -> usize {
        match self {
            StateTranslation::Plain(v, n) => v.len() / n,
            StateTranslation::Dictionary(i) => i.len(),
        }
    }
}

pub(crate) struct BinaryDecoder {
    pub(crate) size: usize,
}

pub(crate) enum FSBVec {
    Size1(Vec<Bytes1Alignment1>),
    Size2(Vec<Bytes2Alignment2>),
    Size4(Vec<Bytes4Alignment4>),
    Size8(Vec<Bytes8Alignment8>),
    Size12(Vec<Bytes12Alignment4>),
    Size16(Vec<Bytes16Alignment16>),
    Size32(Vec<Bytes32Alignment16>),
    Other(Vec<u8>, usize),
}

impl FSBVec {
    pub fn new(size: usize) -> FSBVec {
        match size {
            1 => Self::Size1(Vec::new()),
            2 => Self::Size2(Vec::new()),
            4 => Self::Size4(Vec::new()),
            8 => Self::Size8(Vec::new()),
            12 => Self::Size12(Vec::new()),
            16 => Self::Size16(Vec::new()),
            32 => Self::Size32(Vec::new()),
            _ => Self::Other(Vec::new(), size),
        }
    }

    fn size(&self) -> usize {
        match self {
            FSBVec::Size1(_) => 1,
            FSBVec::Size2(_) => 2,
            FSBVec::Size4(_) => 4,
            FSBVec::Size8(_) => 8,
            FSBVec::Size12(_) => 12,
            FSBVec::Size16(_) => 16,
            FSBVec::Size32(_) => 32,
            FSBVec::Other(_, size) => *size,
        }
    }

    pub fn into_bytes_buffer(self) -> Buffer<u8> {
        Buffer::from_storage(match self {
            FSBVec::Size1(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Size2(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Size4(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Size8(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Size12(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Size16(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Size32(vec) => SharedStorage::bytes_from_pod_vec(vec),
            FSBVec::Other(vec, _) => SharedStorage::from_vec(vec),
        })
    }

    pub fn extend_from_byte_slice(&mut self, slice: &[u8]) {
        let size = self.size();
        if size == 0 {
            assert_eq!(slice.len(), 0);
            return;
        }

        assert_eq!(slice.len() % size, 0);

        macro_rules! extend_from_slice {
            ($v:expr) => {{
                $v.reserve(slice.len() / size);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        slice.as_ptr(),
                        $v.as_mut_ptr().add($v.len()) as *mut _,
                        slice.len(),
                    )
                }
                let new_len = $v.len() + slice.len() / size;
                unsafe { $v.set_len(new_len) };
            }};
        }

        match self {
            FSBVec::Size1(v) => extend_from_slice!(v),
            FSBVec::Size2(v) => extend_from_slice!(v),
            FSBVec::Size4(v) => extend_from_slice!(v),
            FSBVec::Size8(v) => extend_from_slice!(v),
            FSBVec::Size12(v) => extend_from_slice!(v),
            FSBVec::Size16(v) => extend_from_slice!(v),
            FSBVec::Size32(v) => extend_from_slice!(v),
            FSBVec::Other(v, _) => v.extend_from_slice(slice),
        }
    }

    fn len(&self) -> usize {
        match self {
            FSBVec::Size1(vec) => vec.len(),
            FSBVec::Size2(vec) => vec.len(),
            FSBVec::Size4(vec) => vec.len(),
            FSBVec::Size8(vec) => vec.len(),
            FSBVec::Size12(vec) => vec.len(),
            FSBVec::Size16(vec) => vec.len(),
            FSBVec::Size32(vec) => vec.len(),
            FSBVec::Other(vec, size) => vec.len() / size,
        }
    }
}

impl utils::Decoded for (FSBVec, BitmapBuilder) {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn extend_nulls(&mut self, n: usize) {
        match &mut self.0 {
            FSBVec::Size1(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Size2(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Size4(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Size8(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Size12(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Size16(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Size32(v) => v.resize(v.len() + n, Zeroable::zeroed()),
            FSBVec::Other(v, size) => v.resize(v.len() + n * *size, Zeroable::zeroed()),
        }
        self.1.extend_constant(n, false);
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_fsb_plain(
    size: usize,
    values: &[u8],
    target: &mut FSBVec,
    pred_true_mask: &mut BitmapBuilder,
    validity: &mut BitmapBuilder,
    is_optional: bool,
    filter: Option<Filter>,
    page_validity: Option<&Bitmap>,
) -> ParquetResult<()> {
    assert_ne!(size, 0);
    assert_eq!(values.len() % size, 0);

    macro_rules! decode_static_size {
        ($target:ident) => {{
            let values = ArrayChunks::new(values).ok_or_else(|| {
                ParquetError::oos("Page content does not align with expected element size")
            })?;
            super::primitive::plain::decode_aligned_bytes_dispatch(
                values,
                is_optional,
                page_validity,
                filter,
                validity,
                $target,
                pred_true_mask,
            )
        }};
    }

    use FSBVec as T;
    match target {
        T::Size1(target) => decode_static_size!(target),
        T::Size2(target) => decode_static_size!(target),
        T::Size4(target) => decode_static_size!(target),
        T::Size8(target) => decode_static_size!(target),
        T::Size12(target) => decode_static_size!(target),
        T::Size16(target) => decode_static_size!(target),
        T::Size32(target) => decode_static_size!(target),
        T::Other(target, _) => {
            // @NOTE: All these kernels are quite slow, but they should be very uncommon and the
            // general case requires arbitrary length memcopies anyway.

            if is_optional {
                append_validity(
                    page_validity,
                    filter.as_ref(),
                    validity,
                    values.len() / size,
                );
            }

            let page_validity =
                constrain_page_validity(values.len() / size, page_validity, filter.as_ref());

            match (page_validity, filter.as_ref()) {
                (None, None) => target.extend_from_slice(values),
                (None, Some(filter)) => match filter {
                    Filter::Range(range) => {
                        target.extend_from_slice(&values[range.start * size..range.end * size])
                    },
                    Filter::Mask(bitmap) => {
                        let mut iter = bitmap.iter();
                        let mut offset = 0;

                        while iter.num_remaining() > 0 {
                            let num_selected = iter.take_leading_ones();
                            target
                                .extend_from_slice(&values[offset * size..][..num_selected * size]);
                            offset += num_selected;

                            let num_filtered = iter.take_leading_zeros();
                            offset += num_filtered;
                        }
                    },
                    Filter::Predicate(_) => todo!(),
                },
                (Some(validity), None) => {
                    let mut iter = validity.iter();
                    let mut offset = 0;

                    while iter.num_remaining() > 0 {
                        let num_valid = iter.take_leading_ones();
                        target.extend_from_slice(&values[offset * size..][..num_valid * size]);
                        offset += num_valid;

                        let num_filtered = iter.take_leading_zeros();
                        target.resize(target.len() + num_filtered * size, 0);
                    }
                },
                (Some(validity), Some(filter)) => match filter {
                    Filter::Range(range) => {
                        let (skipped, active) = validity.split_at(range.start);

                        let active = active.sliced(0, range.len());

                        let mut iter = active.iter();
                        let mut offset = skipped.set_bits();

                        while iter.num_remaining() > 0 {
                            let num_valid = iter.take_leading_ones();
                            target.extend_from_slice(&values[offset * size..][..num_valid * size]);
                            offset += num_valid;

                            let num_filtered = iter.take_leading_zeros();
                            target.resize(target.len() + num_filtered * size, 0);
                        }
                    },
                    Filter::Mask(filter) => {
                        let mut offset = 0;
                        for (is_selected, is_valid) in filter.iter().zip(validity.iter()) {
                            if is_selected {
                                if is_valid {
                                    target.extend_from_slice(&values[offset * size..][..size]);
                                } else {
                                    target.resize(target.len() + size, 0);
                                }
                            }

                            offset += usize::from(is_valid);
                        }
                    },
                    Filter::Predicate(_) => todo!(),
                },
            }

            Ok(())
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_fsb_dict(
    size: usize,
    values: HybridRleDecoder<'_>,
    dict: &FixedSizeBinaryArray,
    dict_mask: Option<&Bitmap>,
    target: &mut FSBVec,
    pred_true_mask: &mut BitmapBuilder,
    validity: &mut BitmapBuilder,
    is_optional: bool,
    filter: Option<Filter>,
    page_validity: Option<&Bitmap>,
) -> ParquetResult<()> {
    assert_ne!(size, 0);

    macro_rules! decode_static_size {
        ($dict:ident, $target:ident) => {{
            let dict = $dict.values().as_slice();
            // @NOTE: We initialize the dict with the right alignment for this to work.
            let dict = bytemuck::cast_slice(dict);

            super::dictionary_encoded::decode_dict_dispatch(
                values,
                dict,
                dict_mask,
                is_optional,
                page_validity,
                filter,
                validity,
                $target,
                pred_true_mask,
            )
        }};
    }

    use FSBVec as T;
    match target {
        T::Size1(target) => decode_static_size!(dict, target),
        T::Size2(target) => decode_static_size!(dict, target),
        T::Size4(target) => decode_static_size!(dict, target),
        T::Size8(target) => decode_static_size!(dict, target),
        T::Size12(target) => decode_static_size!(dict, target),
        T::Size16(target) => decode_static_size!(dict, target),
        T::Size32(target) => decode_static_size!(dict, target),
        T::Other(target, _) => {
            // @NOTE: All these kernels are quite slow, but they should be very uncommon and the
            // general case requires arbitrary length memcopies anyway.

            let dict = dict.values().as_slice();

            if is_optional {
                append_validity(
                    page_validity,
                    filter.as_ref(),
                    validity,
                    values.len() / size,
                );
            }

            let page_validity =
                constrain_page_validity(values.len() / size, page_validity, filter.as_ref());

            let mut indexes = Vec::with_capacity(values.len());

            for chunk in values.into_chunk_iter() {
                match chunk? {
                    HybridRleChunk::Rle(value, repeats) => {
                        indexes.resize(indexes.len() + repeats, value)
                    },
                    HybridRleChunk::Bitpacked(decoder) => decoder.collect_into(&mut indexes),
                }
            }

            match (page_validity, filter.as_ref()) {
                (None, None) => target.extend(
                    indexes
                        .into_iter()
                        .flat_map(|v| &dict[(v as usize) * size..][..size]),
                ),
                (None, Some(filter)) => match filter {
                    Filter::Range(range) => target.extend(
                        indexes[range.start..range.end]
                            .iter()
                            .flat_map(|v| &dict[(*v as usize) * size..][..size]),
                    ),
                    Filter::Mask(bitmap) => {
                        let mut iter = bitmap.iter();
                        let mut offset = 0;

                        while iter.num_remaining() > 0 {
                            let num_selected = iter.take_leading_ones();
                            target.extend(
                                indexes[offset..][..num_selected]
                                    .iter()
                                    .flat_map(|v| &dict[(*v as usize) * size..][..size]),
                            );
                            offset += num_selected;

                            let num_filtered = iter.take_leading_zeros();
                            offset += num_filtered;
                        }
                    },
                    Filter::Predicate(_) => todo!(),
                },
                (Some(validity), None) => {
                    let mut iter = validity.iter();
                    let mut offset = 0;

                    while iter.num_remaining() > 0 {
                        let num_valid = iter.take_leading_ones();
                        target.extend(
                            indexes[offset..][..num_valid]
                                .iter()
                                .flat_map(|v| &dict[(*v as usize) * size..][..size]),
                        );
                        offset += num_valid;

                        let num_filtered = iter.take_leading_zeros();
                        target.resize(target.len() + num_filtered * size, 0);
                    }
                },
                (Some(validity), Some(filter)) => match filter {
                    Filter::Range(range) => {
                        let (skipped, active) = validity.split_at(range.start);

                        let active = active.sliced(0, range.len());

                        let mut iter = active.iter();
                        let mut offset = skipped.set_bits();

                        while iter.num_remaining() > 0 {
                            let num_valid = iter.take_leading_ones();
                            target.extend(
                                indexes[offset..][..num_valid]
                                    .iter()
                                    .flat_map(|v| &dict[(*v as usize) * size..][..size]),
                            );
                            offset += num_valid;

                            let num_filtered = iter.take_leading_zeros();
                            target.resize(target.len() + num_filtered * size, 0);
                        }
                    },
                    Filter::Mask(filter) => {
                        let mut offset = 0;
                        for (is_selected, is_valid) in filter.iter().zip(validity.iter()) {
                            if is_selected {
                                if is_valid {
                                    target.extend_from_slice(
                                        &dict[(indexes[offset] as usize) * size..][..size],
                                    );
                                } else {
                                    target.resize(target.len() + size, 0);
                                }
                            }

                            offset += usize::from(is_valid);
                        }
                    },
                    Filter::Predicate(_) => todo!(),
                },
            }

            Ok(())
        },
    }
}

impl Decoder for BinaryDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = FixedSizeBinaryArray;
    type DecodedState = (FSBVec, BitmapBuilder);
    type Output = FixedSizeBinaryArray;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        let size = self.size;

        let values = match size {
            1 => FSBVec::Size1(Vec::with_capacity(capacity)),
            2 => FSBVec::Size2(Vec::with_capacity(capacity)),
            4 => FSBVec::Size4(Vec::with_capacity(capacity)),
            8 => FSBVec::Size8(Vec::with_capacity(capacity)),
            12 => FSBVec::Size12(Vec::with_capacity(capacity)),
            16 => FSBVec::Size16(Vec::with_capacity(capacity)),
            32 => FSBVec::Size32(Vec::with_capacity(capacity)),
            _ => FSBVec::Other(Vec::with_capacity(capacity * size), size),
        };

        (values, BitmapBuilder::with_capacity(capacity))
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let mut target = FSBVec::new(self.size);
        decode_fsb_plain(
            self.size,
            page.buffer.as_ref(),
            &mut target,
            &mut BitmapBuilder::new(),
            &mut BitmapBuilder::new(),
            false,
            None,
            None,
        )?;

        Ok(FixedSizeBinaryArray::new(
            ArrowDataType::FixedSizeBinary(self.size),
            target.into_bytes_buffer(),
            None,
        ))
    }

    fn has_predicate_specialization(
        &self,
        _state: &utils::State<'_, Self>,
        _predicate: &PredicateFilter,
    ) -> ParquetResult<bool> {
        // @TODO: This can be enabled for the fast paths
        Ok(false)
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn arrow::array::Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        let additional = additional
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        decoded
            .0
            .extend_from_byte_slice(additional.values().as_slice());
        match additional.validity() {
            Some(v) => decoded.1.extend_from_bitmap(v),
            None if is_optional => decoded.1.extend_constant(additional.len(), true),
            None => {},
        }

        Ok(())
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);

        Ok(FixedSizeBinaryArray::new(
            dtype,
            values.into_bytes_buffer(),
            validity,
        ))
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(values, size) => decode_fsb_plain(
                size,
                values,
                &mut decoded.0,
                pred_true_mask,
                &mut decoded.1,
                state.is_optional,
                filter,
                state.page_validity.as_ref(),
            ),
            StateTranslation::Dictionary(values) => decode_fsb_dict(
                self.size,
                values,
                state.dict.unwrap(),
                state.dict_mask,
                &mut decoded.0,
                pred_true_mask,
                &mut decoded.1,
                state.is_optional,
                filter,
                state.page_validity.as_ref(),
            ),
        }
    }
}
