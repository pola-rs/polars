use arrow::array::specification::try_check_utf8;
use arrow::array::{BinaryArray, MutableBinaryValuesArray};
use polars_error::PolarsResult;

use super::super::utils;
use super::super::utils::{get_selected_rows, FilteredOptionalPageValidity, OptionalPageValidity};
use super::utils::*;
use crate::parquet::deserialize::SliceFilteredIter;
use crate::parquet::encoding::{delta_bitpacked, delta_length_byte_array, hybrid_rle, Encoding};
use crate::parquet::page::{split_buffer, DataPage};
use crate::read::deserialize::utils::{page_is_filtered, page_is_optional};
use crate::read::ParquetError;

pub(crate) type BinaryDict = BinaryArray<i64>;

#[derive(Debug)]
pub(crate) struct Required<'a> {
    pub values: std::iter::Take<BinaryIter<'a>>,
}

impl<'a> Required<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let (_, _, values) = split_buffer(page)?;
        let values = BinaryIter::new(values).take(page.num_values());

        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(crate) struct Delta<'a> {
    pub lengths: std::vec::IntoIter<usize>,
    pub values: &'a [u8],
}

impl<'a> Delta<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let (_, _, values) = split_buffer(page)?;

        let mut lengths_iter = delta_length_byte_array::Decoder::try_new(values)?;

        #[allow(clippy::needless_collect)] // we need to consume it to get the values
        let lengths = lengths_iter
            .by_ref()
            .map(|x| x.map(|x| x as usize))
            .collect::<Result<Vec<_>, ParquetError>>()?;

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
        let (_, _, values) = split_buffer(page)?;
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
pub(crate) struct FilteredRequired<'a> {
    pub values: SliceFilteredIter<std::iter::Take<BinaryIter<'a>>>,
}

impl<'a> FilteredRequired<'a> {
    pub fn new(page: &'a DataPage) -> Self {
        let values = BinaryIter::new(page.buffer()).take(page.num_values());

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(crate) struct FilteredDelta<'a> {
    pub values: SliceFilteredIter<Delta<'a>>,
}

impl<'a> FilteredDelta<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let values = Delta::try_new(page)?;

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(crate) struct RequiredDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a BinaryDict,
}

impl<'a> RequiredDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a BinaryDict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(crate) struct FilteredRequiredDictionary<'a> {
    pub values: SliceFilteredIter<hybrid_rle::HybridRleDecoder<'a>>,
    pub dict: &'a BinaryDict,
}

impl<'a> FilteredRequiredDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a BinaryDict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Ok(Self { values, dict })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
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
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(crate) enum BinaryState<'a> {
    Optional(OptionalPageValidity<'a>, BinaryIter<'a>),
    Required(Required<'a>),
    RequiredDictionary(RequiredDictionary<'a>),
    OptionalDictionary(OptionalPageValidity<'a>, ValuesDictionary<'a>),
    Delta(Delta<'a>),
    OptionalDelta(OptionalPageValidity<'a>, Delta<'a>),
    DeltaByteArray(DeltaBytes<'a>),
    OptionalDeltaByteArray(OptionalPageValidity<'a>, DeltaBytes<'a>),
    FilteredRequired(FilteredRequired<'a>),
    FilteredDelta(FilteredDelta<'a>),
    FilteredOptionalDelta(FilteredOptionalPageValidity<'a>, Delta<'a>),
    FilteredOptional(FilteredOptionalPageValidity<'a>, BinaryIter<'a>),
    FilteredRequiredDictionary(FilteredRequiredDictionary<'a>),
    FilteredOptionalDictionary(FilteredOptionalPageValidity<'a>, ValuesDictionary<'a>),
}

impl<'a> utils::PageState<'a> for BinaryState<'a> {
    fn len(&self) -> usize {
        match self {
            BinaryState::Optional(validity, _) => validity.len(),
            BinaryState::Required(state) => state.len(),
            BinaryState::Delta(state) => state.len(),
            BinaryState::OptionalDelta(state, _) => state.len(),
            BinaryState::RequiredDictionary(values) => values.len(),
            BinaryState::OptionalDictionary(optional, _) => optional.len(),
            BinaryState::FilteredRequired(state) => state.len(),
            BinaryState::FilteredOptional(validity, _) => validity.len(),
            BinaryState::FilteredDelta(state) => state.len(),
            BinaryState::FilteredOptionalDelta(state, _) => state.len(),
            BinaryState::FilteredRequiredDictionary(values) => values.len(),
            BinaryState::FilteredOptionalDictionary(optional, _) => optional.len(),
            BinaryState::DeltaByteArray(values) => values.size_hint().0,
            BinaryState::OptionalDeltaByteArray(optional, _) => optional.len(),
        }
    }
}

pub(crate) fn deserialize_plain(values: &[u8], num_values: usize) -> BinaryDict {
    let all = BinaryIter::new(values).take(num_values).collect::<Vec<_>>();
    let values_size = all.iter().map(|v| v.len()).sum::<usize>();
    let mut dict_values = MutableBinaryValuesArray::<i64>::with_capacities(all.len(), values_size);
    for v in all {
        dict_values.push(v)
    }
    dict_values.into()
}

pub(crate) fn build_binary_state<'a>(
    page: &'a DataPage,
    dict: Option<&'a BinaryDict>,
    is_string: bool,
) -> PolarsResult<BinaryState<'a>> {
    let is_optional = utils::page_is_optional(page);
    let is_filtered = utils::page_is_filtered(page);

    match (page.encoding(), dict, is_optional, is_filtered) {
        (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
            if is_string {
                try_check_utf8(dict.offsets(), dict.values())?;
            }
            Ok(BinaryState::RequiredDictionary(
                RequiredDictionary::try_new(page, dict)?,
            ))
        },
        (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
            if is_string {
                try_check_utf8(dict.offsets(), dict.values())?;
            }
            Ok(BinaryState::OptionalDictionary(
                OptionalPageValidity::try_new(page)?,
                ValuesDictionary::try_new(page, dict)?,
            ))
        },
        (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, true) => {
            if is_string {
                try_check_utf8(dict.offsets(), dict.values())?;
            }
            FilteredRequiredDictionary::try_new(page, dict)
                .map(BinaryState::FilteredRequiredDictionary)
        },
        (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, true) => {
            if is_string {
                try_check_utf8(dict.offsets(), dict.values())?;
            }
            Ok(BinaryState::FilteredOptionalDictionary(
                FilteredOptionalPageValidity::try_new(page)?,
                ValuesDictionary::try_new(page, dict)?,
            ))
        },
        (Encoding::Plain, _, true, false) => {
            let (_, _, values) = split_buffer(page)?;

            let values = BinaryIter::new(values);

            Ok(BinaryState::Optional(
                OptionalPageValidity::try_new(page)?,
                values,
            ))
        },
        (Encoding::Plain, _, false, false) => Ok(BinaryState::Required(Required::try_new(page)?)),
        (Encoding::Plain, _, false, true) => {
            Ok(BinaryState::FilteredRequired(FilteredRequired::new(page)))
        },
        (Encoding::Plain, _, true, true) => {
            let (_, _, values) = split_buffer(page)?;

            Ok(BinaryState::FilteredOptional(
                FilteredOptionalPageValidity::try_new(page)?,
                BinaryIter::new(values),
            ))
        },
        (Encoding::DeltaLengthByteArray, _, false, false) => {
            Delta::try_new(page).map(BinaryState::Delta)
        },
        (Encoding::DeltaLengthByteArray, _, true, false) => Ok(BinaryState::OptionalDelta(
            OptionalPageValidity::try_new(page)?,
            Delta::try_new(page)?,
        )),
        (Encoding::DeltaLengthByteArray, _, false, true) => {
            FilteredDelta::try_new(page).map(BinaryState::FilteredDelta)
        },
        (Encoding::DeltaLengthByteArray, _, true, true) => Ok(BinaryState::FilteredOptionalDelta(
            FilteredOptionalPageValidity::try_new(page)?,
            Delta::try_new(page)?,
        )),
        (Encoding::DeltaByteArray, _, true, false) => Ok(BinaryState::OptionalDeltaByteArray(
            OptionalPageValidity::try_new(page)?,
            DeltaBytes::try_new(page)?,
        )),
        (Encoding::DeltaByteArray, _, false, false) => {
            Ok(BinaryState::DeltaByteArray(DeltaBytes::try_new(page)?))
        },
        _ => Err(utils::not_implemented(page)),
    }
}

#[derive(Debug)]
pub(crate) enum BinaryNestedState<'a> {
    Optional(BinaryIter<'a>),
    Required(BinaryIter<'a>),
    RequiredDictionary(ValuesDictionary<'a>),
    OptionalDictionary(ValuesDictionary<'a>),
}

impl<'a> utils::PageState<'a> for BinaryNestedState<'a> {
    fn len(&self) -> usize {
        match self {
            BinaryNestedState::Optional(validity) => validity.size_hint().0,
            BinaryNestedState::Required(state) => state.size_hint().0,
            BinaryNestedState::RequiredDictionary(required) => required.len(),
            BinaryNestedState::OptionalDictionary(optional) => optional.len(),
        }
    }
}

pub(crate) fn build_nested_state<'a>(
    page: &'a DataPage,
    dict: Option<&'a BinaryDict>,
) -> PolarsResult<BinaryNestedState<'a>> {
    let is_optional = page_is_optional(page);
    let is_filtered = page_is_filtered(page);

    match (page.encoding(), dict, is_optional, is_filtered) {
        (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
            ValuesDictionary::try_new(page, dict).map(BinaryNestedState::RequiredDictionary)
        },
        (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
            ValuesDictionary::try_new(page, dict).map(BinaryNestedState::OptionalDictionary)
        },
        (Encoding::Plain, _, true, false) => {
            let (_, _, values) = split_buffer(page)?;

            let values = BinaryIter::new(values);

            Ok(BinaryNestedState::Optional(values))
        },
        (Encoding::Plain, _, false, false) => {
            let (_, _, values) = split_buffer(page)?;

            let values = BinaryIter::new(values);

            Ok(BinaryNestedState::Required(values))
        },
        _ => Err(utils::not_implemented(page)),
    }
}
