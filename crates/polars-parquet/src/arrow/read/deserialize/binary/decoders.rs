use arrow::array::specification::try_check_utf8;
use arrow::array::{BinaryArray, MutableBinaryValuesArray};
use arrow::offset::Offsets;
use arrow::types::Offset;
use polars_error::PolarsResult;

use super::super::utils;
use super::utils::*;
use crate::parquet::encoding::{
    delta_bitpacked, delta_byte_array, delta_length_byte_array, hybrid_rle, Encoding,
};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage};
use crate::read::deserialize::utils::PageValidity;

pub(crate) type BinaryDict = BinaryArray<i64>;

#[derive(Debug)]
pub(crate) struct ValuesDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a BinaryDict,
}

impl<'a> ValuesDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a BinaryDict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum BinaryStateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(ValuesDictionary<'a>),
    DeltaLengthByteArray(delta_length_byte_array::Decoder<'a>, Vec<u32>),
    DeltaBytes(delta_byte_array::Decoder<'a>),
}

impl<'a> BinaryStateTranslation<'a> {
    pub(crate) fn new(
        page: &'a DataPage,
        dict: Option<&'a BinaryDict>,
        _page_validity: Option<&PageValidity<'a>>,
        is_string: bool,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                if is_string {
                    try_check_utf8(dict.offsets(), dict.values())?;
                }
                Ok(BinaryStateTranslation::Dictionary(
                    ValuesDictionary::try_new(page, dict)?,
                ))
            },
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let values = BinaryIter::new(values, page.num_values());

                Ok(BinaryStateTranslation::Plain(values))
            },
            (Encoding::DeltaLengthByteArray, _) => {
                let values = split_buffer(page)?.values;
                Ok(BinaryStateTranslation::DeltaLengthByteArray(
                    delta_length_byte_array::Decoder::try_new(values)?,
                    Vec::new(),
                ))
            },
            (Encoding::DeltaByteArray, _) => {
                let values = split_buffer(page)?.values;
                Ok(BinaryStateTranslation::DeltaBytes(
                    delta_byte_array::Decoder::try_new(values)?,
                ))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }
    pub(crate) fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(v) => v.len_when_not_nullable(),
            Self::Dictionary(v) => v.len(),
            Self::DeltaLengthByteArray(v, _) => v.len(),
            Self::DeltaBytes(v) => v.len(),
        }
    }

    pub(crate) fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(t) => _ = t.by_ref().nth(n - 1),
            Self::Dictionary(t) => t.values.skip_in_place(n)?,
            Self::DeltaLengthByteArray(t, _) => t.skip_in_place(n)?,
            Self::DeltaBytes(t) => t.skip_in_place(n)?,
        }

        Ok(())
    }
}

pub(crate) fn deserialize_plain(values: &[u8], num_values: usize) -> BinaryDict {
    // Each value is prepended by the length which is 4 bytes.
    let num_bytes = values.len() - 4 * num_values;

    let mut dict_values = MutableBinaryValuesArray::<i64>::with_capacities(num_values, num_bytes);
    for v in BinaryIter::new(values, num_values) {
        dict_values.push(v)
    }

    dict_values.into()
}

#[derive(Default)]
pub(crate) struct OffsetGatherer<O: Offset> {
    _pd: std::marker::PhantomData<O>,
}

impl<O: Offset> delta_bitpacked::DeltaGatherer for OffsetGatherer<O> {
    type Target = Offsets<O>;

    fn target_len(&self, target: &Self::Target) -> usize {
        target.len()
    }

    fn target_reserve(&self, target: &mut Self::Target, n: usize) {
        target.reserve(n);
    }

    fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
        target.try_push(v.try_into().unwrap()).unwrap();
        Ok(())
    }
    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        target
            .try_extend_from_lengths(slice.iter().copied().map(|i| i.try_into().unwrap()))
            .map_err(|_| ParquetError::oos("Invalid length in delta encoding"))
    }
    fn gather_chunk(&mut self, target: &mut Self::Target, chunk: &[i64; 64]) -> ParquetResult<()> {
        target
            .try_extend_from_lengths(chunk.iter().copied().map(|i| i.try_into().unwrap()))
            .map_err(|_| ParquetError::oos("Invalid length in delta encoding"))
    }
}
