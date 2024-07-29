use arrow::array::specification::try_check_utf8;
use arrow::array::{BinaryArray, MutableBinaryValuesArray};
use polars_error::PolarsResult;

use super::super::utils;
use super::utils::*;
use crate::parquet::encoding::{delta_bitpacked, delta_length_byte_array, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage};
use crate::read::deserialize::utils::PageValidity;

pub(crate) type BinaryDict = BinaryArray<i64>;

#[derive(Debug)]
pub(crate) struct Delta<'a> {
    pub lengths: std::vec::IntoIter<usize>,
    pub values: &'a [u8],
}

impl<'a> Delta<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let values = split_buffer(page)?.values;

        let mut lengths_iter = delta_length_byte_array::Decoder::try_new(values)?;

        #[allow(clippy::needless_collect)] // we need to consume it to get the values
        let lengths = lengths_iter
            .by_ref()
            .map(|x| x.map(|x| x as usize))
            .collect::<ParquetResult<Vec<_>>>()?;

        let values = lengths_iter.into_values();
        Ok(Self {
            lengths: lengths.into_iter(),
            values,
        })
    }

    pub fn len(&self) -> usize {
        self.lengths.size_hint().0
    }
}

impl<'a> Iterator for Delta<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let length = self.lengths.next()?;
        let (item, remaining) = self.values.split_at(length);
        self.values = remaining;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lengths.size_hint()
    }
}

#[derive(Debug)]
pub(crate) struct DeltaBytes<'a> {
    prefix: std::vec::IntoIter<i32>,
    suffix: std::vec::IntoIter<i32>,
    data: &'a [u8],
    data_offset: usize,
    last_value: Vec<u8>,
}

impl<'a> DeltaBytes<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let values = split_buffer(page)?.values;
        let mut decoder = delta_bitpacked::Decoder::try_new(values)?;
        let prefix = (&mut decoder)
            .take(page.num_values())
            .map(|r| r.map(|v| v as i32).unwrap())
            .collect::<Vec<_>>();

        let mut data_offset = decoder.consumed_bytes();
        let mut decoder = delta_bitpacked::Decoder::try_new(&values[decoder.consumed_bytes()..])?;
        let suffix = (&mut decoder)
            .map(|r| r.map(|v| v as i32).unwrap())
            .collect::<Vec<_>>();
        data_offset += decoder.consumed_bytes();

        Ok(Self {
            prefix: prefix.into_iter(),
            suffix: suffix.into_iter(),
            data: values,
            data_offset,
            last_value: vec![],
        })
    }
}

impl<'a> Iterator for DeltaBytes<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let prefix_len = self.prefix.next()? as usize;
        let suffix_len = self.suffix.next()? as usize;

        self.last_value.truncate(prefix_len);
        self.last_value
            .extend_from_slice(&self.data[self.data_offset..self.data_offset + suffix_len]);
        self.data_offset += suffix_len;

        // SAFETY: the consumer will only keep one value around per iteration.
        // We need a different API for this to work with safe code.
        let extend_lifetime =
            unsafe { std::mem::transmute::<&[u8], &'a [u8]>(self.last_value.as_slice()) };
        Some(extend_lifetime)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.prefix.size_hint()
    }
}

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

#[derive(Debug)]
pub(crate) enum BinaryStateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(ValuesDictionary<'a>),
    Delta(Delta<'a>),
    DeltaBytes(DeltaBytes<'a>),
}

impl<'a> BinaryStateTranslation<'a> {
    pub(crate) fn new(
        page: &'a DataPage,
        dict: Option<&'a BinaryDict>,
        _page_validity: Option<&PageValidity<'a>>,
        is_string: bool,
    ) -> PolarsResult<Self> {
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
                Ok(BinaryStateTranslation::Delta(Delta::try_new(page)?))
            },
            (Encoding::DeltaByteArray, _) => Ok(BinaryStateTranslation::DeltaBytes(
                DeltaBytes::try_new(page)?,
            )),
            _ => Err(utils::not_implemented(page)),
        }
    }
    pub(crate) fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(v) => v.len_when_not_nullable(),
            Self::Dictionary(v) => v.len(),
            Self::Delta(v) => v.len(),
            Self::DeltaBytes(v) => v.size_hint().0,
        }
    }

    pub(crate) fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(t) => _ = t.by_ref().nth(n - 1),
            Self::Dictionary(t) => t.values.skip_in_place(n)?,
            Self::Delta(t) => _ = t.by_ref().nth(n - 1),
            Self::DeltaBytes(t) => _ = t.by_ref().nth(n - 1),
        }

        Ok(())
    }
}

pub(crate) fn deserialize_plain(values: &[u8], num_values: usize) -> BinaryDict {
    let all = BinaryIter::new(values, num_values).collect::<Vec<_>>();
    let values_size = all.iter().map(|v| v.len()).sum::<usize>();
    let mut dict_values = MutableBinaryValuesArray::<i64>::with_capacities(all.len(), values_size);
    for v in all {
        dict_values.push(v)
    }

    dict_values.into()
}
