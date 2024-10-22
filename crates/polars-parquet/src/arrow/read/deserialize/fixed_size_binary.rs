use arrow::array::{
    DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray, Splitable,
};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::storage::SharedStorage;
use arrow::types::{
    Bytes12Alignment4, Bytes16Alignment16, Bytes1Alignment1, Bytes2Alignment2, Bytes32Alignment16,
    Bytes4Alignment4, Bytes8Alignment8,
};

use super::utils::array_chunks::ArrayChunks;
use super::utils::dict_encoded::append_validity;
use super::utils::{dict_indices_decoder, freeze_validity, Decoder};
use super::Filter;
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::encoding::{hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils;
use crate::read::deserialize::utils::dict_encoded::constrain_page_validity;

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
        _decoder: &mut BinaryDecoder,
        _decoded: &mut <BinaryDecoder as Decoder>::DecodedState,
        _is_optional: bool,
        _page_validity: &mut Option<Bitmap>,
        _dict: Option<&'a <BinaryDecoder as Decoder>::Dict>,
        _additional: usize,
    ) -> ParquetResult<()> {
        unreachable!()
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

    pub fn into_bytes_buffer(self) -> Buffer<u8> {
        Buffer::from_storage(match self {
            FSBVec::Size1(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Size2(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Size4(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Size8(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Size12(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Size16(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Size32(vec) => SharedStorage::bytes_from_aligned_bytes(vec),
            FSBVec::Other(vec, _) => SharedStorage::from_vec(vec),
        })
    }
}

impl<T> utils::ExactSize for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl utils::ExactSize for FSBVec {
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

impl utils::ExactSize for (FSBVec, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

fn decode_fsb_plain(
    size: usize,
    values: &[u8],
    target: &mut FSBVec,
    validity: &mut MutableBitmap,
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
    dict: &FSBVec,
    target: &mut FSBVec,
    validity: &mut MutableBitmap,
    is_optional: bool,
    filter: Option<Filter>,
    page_validity: Option<&Bitmap>,
) -> ParquetResult<()> {
    assert_ne!(size, 0);

    macro_rules! decode_static_size {
        ($dict:ident, $target:ident) => {{
            super::utils::dict_encoded::decode_dict_dispatch(
                values,
                $dict,
                is_optional,
                page_validity,
                filter,
                validity,
                $target,
            )
        }};
    }

    use FSBVec as T;
    match (dict, target) {
        (T::Size1(dict), T::Size1(target)) => decode_static_size!(dict, target),
        (T::Size2(dict), T::Size2(target)) => decode_static_size!(dict, target),
        (T::Size4(dict), T::Size4(target)) => decode_static_size!(dict, target),
        (T::Size8(dict), T::Size8(target)) => decode_static_size!(dict, target),
        (T::Size12(dict), T::Size12(target)) => decode_static_size!(dict, target),
        (T::Size16(dict), T::Size16(target)) => decode_static_size!(dict, target),
        (T::Size32(dict), T::Size32(target)) => decode_static_size!(dict, target),
        (T::Other(dict, _), T::Other(target, _)) => {
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
                },
            }

            Ok(())
        },
        _ => unreachable!(),
    }
}

impl Decoder for BinaryDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = FSBVec;
    type DecodedState = (FSBVec, MutableBitmap);
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

        (values, MutableBitmap::with_capacity(capacity))
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let mut target = FSBVec::new(self.size);
        decode_fsb_plain(
            self.size,
            page.buffer.as_ref(),
            &mut target,
            &mut MutableBitmap::new(),
            false,
            None,
            None,
        )?;
        Ok(target)
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        _is_optional: bool,
        _page_validity: Option<&mut Bitmap>,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
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
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(values, size) => decode_fsb_plain(
                size,
                values,
                &mut decoded.0,
                &mut decoded.1,
                state.is_optional,
                filter,
                state.page_validity.as_ref(),
            ),
            StateTranslation::Dictionary(values) => decode_fsb_dict(
                self.size,
                values,
                state.dict.unwrap(),
                &mut decoded.0,
                &mut decoded.1,
                state.is_optional,
                filter,
                state.page_validity.as_ref(),
            ),
        }
    }
}

impl utils::DictDecodable for BinaryDecoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        dtype: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let dict = FixedSizeBinaryArray::new(
            ArrowDataType::FixedSizeBinary(self.size),
            dict.into_bytes_buffer(),
            None,
        );
        Ok(DictionaryArray::try_new(dtype, keys, Box::new(dict)).unwrap())
    }
}
