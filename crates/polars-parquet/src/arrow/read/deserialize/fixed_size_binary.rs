use arrow::array::{DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::{
    Bytes12Alignment4, Bytes16Alignment16, Bytes1Alignment1, Bytes2Alignment2, Bytes32Alignment16,
    Bytes4Alignment4, Bytes8Alignment8,
};

use super::utils::array_chunks::ArrayChunks;
use super::utils::dict_encoded::append_validity;
use super::utils::{dict_indices_decoder, extend_from_decoder, freeze_validity, Decoder};
use super::Filter;
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::{hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::dict_encoded::constrain_page_validity;
use crate::read::deserialize::utils::{self, BatchableCollector, GatheredHybridRle};

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
        decoder: &mut BinaryDecoder,
        decoded: &mut <BinaryDecoder as Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<Bitmap>,
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

enum FSBTarget {
    Size1(Vec<Bytes1Alignment1>),
    Size2(Vec<Bytes2Alignment2>),
    Size4(Vec<Bytes4Alignment4>),
    Size8(Vec<Bytes8Alignment8>),
    Size12(Vec<Bytes12Alignment4>),
    Size16(Vec<Bytes16Alignment16>),
    Size32(Vec<Bytes32Alignment16>),
    Other(Vec<u8>, usize),
}

impl<T> utils::ExactSize for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl utils::ExactSize for FSBTarget {
    fn len(&self) -> usize {
        match self {
            FSBTarget::Size1(vec) => vec.len(),
            FSBTarget::Size2(vec) => vec.len(),
            FSBTarget::Size4(vec) => vec.len(),
            FSBTarget::Size8(vec) => vec.len(),
            FSBTarget::Size16(vec) => vec.len(),
            FSBTarget::Size32(vec) => vec.len(),
            FSBTarget::Other(vec, size) => vec.len() / size,
        }
    }
}

impl utils::ExactSize for (FSBTarget, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

fn decode_fsb_plain(
    size: usize,
    values: &[u8],
    target: &mut FSBTarget,
    validity: &mut MutableBitmap,
    is_optional: bool,
    filter: Option<Filter>,
    page_validity: Option<&Bitmap>,
) -> ParquetResult<()> {
    assert_ne!(size, 0);
    assert_eq!(values.len() % size, 0);

    if is_optional {
        append_validity(
            page_validity,
            filter.as_ref(),
            validity,
            values.len() / size,
        );
    }

    let page_validity = constrain_page_validity(values.len() / size, page_validity, filter.as_ref());

    macro_rules! decode_static_size {
        ($target:ident) => {{
            let values = ArrayChunks::new(values).ok_or_else(|| {
                ParquetError::oos("Page content does not align with expected element size")
            })?;
            super::primitive::plain::decode_aligned_bytes_dispatch(
                values,
                is_optional,
                page_validity.as_ref(),
                filter,
                validity,
                $target,
            )
        }};
    }

    use FSBTarget as T;
    match target {
        T::Size1(target) => decode_static_size!(target),
        T::Size2(target) => decode_static_size!(target),
        T::Size4(target) => decode_static_size!(target),
        T::Size8(target) => decode_static_size!(target),
        T::Size12(target) => decode_static_size!(target),
        T::Size16(target) => decode_static_size!(target),
        T::Size32(target) => decode_static_size!(target),
        T::Other(_target, _) => todo!(),
    }
}

impl Decoder for BinaryDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = Vec<u8>;
    type DecodedState = (FSBTarget, MutableBitmap);
    type Output = FixedSizeBinaryArray;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        let size = self.size;

        let values = match size {
            1 => FSBTarget::Size1(Vec::with_capacity(capacity)),
            2 => FSBTarget::Size2(Vec::with_capacity(capacity)),
            4 => FSBTarget::Size4(Vec::with_capacity(capacity)),
            8 => FSBTarget::Size8(Vec::with_capacity(capacity)),
            12 => FSBTarget::Size12(Vec::with_capacity(capacity)),
            16 => FSBTarget::Size16(Vec::with_capacity(capacity)),
            32 => FSBTarget::Size32(Vec::with_capacity(capacity)),
            _ => FSBTarget::Other(Vec::with_capacity(capacity * size), size),
        };

        (values, MutableBitmap::with_capacity(capacity))
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        Ok(page.buffer.into_vec())
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

    fn decode_dictionary_encoded(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut hybrid_rle::HybridRleDecoder<'_>,
        is_optional: bool,
        page_validity: Option<&mut Bitmap>,
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
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);
        Ok(FixedSizeBinaryArray::new(
            dtype,
            values.values.into(),
            validity,
        ))
    }
}

impl utils::DictDecodable for BinaryDecoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        dtype: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let dict =
            FixedSizeBinaryArray::new(ArrowDataType::FixedSizeBinary(self.size), dict.into(), None);
        Ok(DictionaryArray::try_new(dtype, keys, Box::new(dict)).unwrap())
    }
}
