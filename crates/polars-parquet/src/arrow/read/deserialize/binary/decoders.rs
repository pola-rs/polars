use arrow::array::{Array, BinaryArray, MutableBinaryValuesArray, };
use polars_error::PolarsResult;

use super::super::utils::{
    get_selected_rows, DecodedState, FilteredOptionalPageValidity,
    OptionalPageValidity,
};
use super::super::{utils};
use super::utils::*;
use crate::parquet::deserialize::SliceFilteredIter;
use crate::parquet::encoding::{delta_bitpacked, delta_length_byte_array, hybrid_rle};
use crate::parquet::page::{split_buffer, DataPage};
use crate::read::{ParquetError};

pub(super) type Dict = BinaryArray<i64>;

#[derive(Debug)]
pub(super) struct Required<'a> {
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
pub(super) struct Delta<'a> {
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
pub(super) struct DeltaBytes<'a> {
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
pub(super) struct FilteredRequired<'a> {
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
pub(super) struct FilteredDelta<'a> {
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
pub(super) struct RequiredDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a Dict,
}

impl<'a> RequiredDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct FilteredRequiredDictionary<'a> {
    pub values: SliceFilteredIter<hybrid_rle::HybridRleDecoder<'a>>,
    pub dict: &'a Dict,
}

impl<'a> FilteredRequiredDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
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
pub(super) struct ValuesDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a Dict,
}

impl<'a> ValuesDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) enum State<'a> {
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

impl<'a> utils::PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match self {
            State::Optional(validity, _) => validity.len(),
            State::Required(state) => state.len(),
            State::Delta(state) => state.len(),
            State::OptionalDelta(state, _) => state.len(),
            State::RequiredDictionary(values) => values.len(),
            State::OptionalDictionary(optional, _) => optional.len(),
            State::FilteredRequired(state) => state.len(),
            State::FilteredOptional(validity, _) => validity.len(),
            State::FilteredDelta(state) => state.len(),
            State::FilteredOptionalDelta(state, _) => state.len(),
            State::FilteredRequiredDictionary(values) => values.len(),
            State::FilteredOptionalDictionary(optional, _) => optional.len(),
            State::DeltaByteArray(values) => values.size_hint().0,
            State::OptionalDeltaByteArray(optional, _) => optional.len(),
        }
    }
}

pub(super) fn deserialize_plain(values: &[u8], num_values: usize) -> Dict {
    let all = BinaryIter::new(values).take(num_values).collect::<Vec<_>>();
    let values_size = all.iter().map(|v| v.len()).sum::<usize>();
    let mut dict_values = MutableBinaryValuesArray::<i64>::with_capacities(all.len(), values_size);
    for v in all {
        dict_values.push(v)
    }
    dict_values.into()
}
